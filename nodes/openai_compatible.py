import base64
import hashlib
import io
import json
import time
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .api_client import LLMClient
except ImportError:
    from api_client import LLMClient

try:
    from .folder_registry import system_prompt_options, load_system_prompt
except ImportError:
    from folder_registry import system_prompt_options, load_system_prompt


class OpenAICompatibleLoader:
    """
    Simplified OpenAI-compatible API adapter for ComfyUI.

    Targets nanoGPT's standard chat-completions shape: the answer arrives in
    choices[0].message.content and the thinking trace in
    choices[0].message.reasoning.  Legacy providers that use
    message.reasoning_content or <think> tags are still accepted.

    Uses system prompt library.
    Does not use conversation_id.
    """

    TAG = "[zx21]"

    # Values accepted by nanoGPT's top-level reasoning_effort parameter.
    REASONING_EFFORTS = [
        "high",
        "medium",
        "low",
        "minimal",
        "xhigh",
        "max",
        "none",
    ]

    THINK_TAG_RE = re.compile(
        r"<think(?:ing)?\b[^>]*>(.*?)</think(?:ing)?>",
        re.DOTALL | re.IGNORECASE,
    )

    # Vision input limits (ComfyUI IMAGE tensor -> JPEG data URLs).
    MAX_IMAGES = 8
    MAX_IMAGE_SIDE = 1024
    JPEG_QUALITY = 85

    # Local ComfyUI memory, not API-side conversation memory.
    _MEMORY_STORE = {}
    MEMORY_LIMIT = 40

    # Persistent, on-disk history.  It is intentionally independent from the
    # optional API conversation memory, so every successful request is saved.
    _PERSISTENT_HISTORY_FILE = Path(__file__).with_name(
        "openai_compatible_history.txt"
    )
    _PERSISTENT_HISTORY_LOCK = threading.Lock()
    PERSISTENT_HISTORY_LIMIT = 5
    # Per-field cap for the on-disk history: prompt, response and reasoning.
    HISTORY_TEXT_LIMIT = 20000

    @classmethod
    def INPUT_TYPES(cls):
        sp_opts = system_prompt_options() or ["None (empty)"]

        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "https://nano-gpt.com/api/v1",
                    "placeholder": "https://nano-gpt.com/api/v1",
                    "tooltip": "OpenAI-compatible API base URL. nanoGPT standard "
                               "endpoint; subscription plans use "
                               "https://nano-gpt.com/api/subscription/v1",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-nano-...",
                    "tooltip": "API key. Can be empty if local API does not require auth.",
                }),
                "model": ("STRING", {
                    "default": "deepseek/deepseek-v4.1-flash",
                    "placeholder": "deepseek/deepseek-v4.1-flash",
                    "tooltip": "Model name, e.g. deepseek/deepseek-v4.1-flash. "
                               "Right-click this widget and 'Convert model to "
                               "input' to drive it from the LLM Model Selector.",
                }),
                "system_prompt": (sp_opts, {
                    "default": sp_opts[0],
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "hello",
                }),
            },
            "optional": {
                "enable_memory": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable Memory",
                    "tooltip": "Conversation memory, isolated per "
                               "model/endpoint/account.",
                }),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "label": "Thinking Mode",
                    "tooltip": "Enable reasoning. Sends reasoning_effort "
                               "(nanoGPT standard) with the request.",
                }),
                "reasoning_effort": (cls.REASONING_EFFORTS, {
                    "default": "high",
                    "tooltip": "How much the model should think. Only used when "
                               "Thinking Mode is enabled. 'none' disables it.",
                }),
                "image": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "Optional image (or image batch, up to 8) sent "
                               "to the model as vision input. Requires a "
                               "multimodal model.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "zx21/LLM"

    # ── Result helpers ───────────────────────────────────────────────

    @staticmethod
    def _success(
        text: str,
        reasoning: str,
        in_tok: int,
        out_tok: int,
        reasoning_tok: int = 0,
    ) -> dict:
        return {
            "ui": {
                "text": [
                    f"Token Usage:\nInput: {in_tok}\nOutput: {out_tok}\n"
                    f"Reasoning: {reasoning_tok}"
                ]
            },
            "result": (text, reasoning),
        }

    @staticmethod
    def _error_result(error_msg: str) -> dict:
        return {
            "ui": {
                "text": [
                    f"⚠ Error:\n{error_msg}"
                ]
            },
            "result": (f"[Error] {error_msg}", ""),
        }

    # ── Memory ───────────────────────────────────────────────────────

    @staticmethod
    def _memory_key(
        unique_id: str,
        base_url: str,
        model: str,
        system_prompt: str,
        api_key: str = "",
    ) -> Tuple[str, str, str, str, str]:
        """Conversation memory is per node AND per provider config, so a
        model, endpoint or account switch never inherits the previous
        context.  The API key is fingerprinted, never stored raw."""
        key_fingerprint = hashlib.sha256(
            (api_key or "").encode("utf-8")
        ).hexdigest()[:16]
        return (unique_id, base_url, model, system_prompt or "", key_fingerprint)

    def _clear_memory(self, unique_id: str) -> None:
        """Drop every stored conversation belonging to this node."""
        if not unique_id:
            return
        for key in [k for k in self._MEMORY_STORE if k[0] == unique_id]:
            self._MEMORY_STORE.pop(key, None)

    def _get_history(
        self, memory_key: Tuple[str, str, str, str] | None
    ) -> List[Dict[str, Any]]:
        if not memory_key:
            return []

        return list(self._MEMORY_STORE.get(memory_key, []))

    def _remember_turn(
        self,
        memory_key: Tuple[str, str, str, str],
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not memory_key:
            return

        history = self._MEMORY_STORE.setdefault(memory_key, [])

        history.append({
            "role": "user",
            "content": user_text,
        })

        history.append({
            "role": "assistant",
            "content": assistant_text,
        })

        if len(history) > self.MEMORY_LIMIT:
            del history[: len(history) - self.MEMORY_LIMIT]

    def _truncate_for_history(self, text: str) -> str:
        if len(text) > self.HISTORY_TEXT_LIMIT:
            return text[: self.HISTORY_TEXT_LIMIT] + "...[truncated]"
        return text

    def _save_persistent_turn(
        self,
        prompt: str,
        response: str,
    ) -> None:
        """Keep the five newest prompt/response pairs on disk (the reasoning
        trace is delivered via the reasoning output, not stored here)."""
        try:
            with self._PERSISTENT_HISTORY_LOCK:
                history_file = self._PERSISTENT_HISTORY_FILE

                if history_file.exists():
                    saved_history = json.loads(
                        history_file.read_text(encoding="utf-8")
                    )
                    if not isinstance(saved_history, list):
                        raise ValueError("History file must contain a JSON list.")
                else:
                    saved_history = []

                saved_history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": self._truncate_for_history(prompt),
                    "response": self._truncate_for_history(response),
                })
                saved_history = saved_history[-self.PERSISTENT_HISTORY_LIMIT:]

                # Replacing a file on the same volume prevents a partially
                # written history file if ComfyUI stops during the write.
                temporary_file = history_file.with_name(
                    f"{history_file.name}.tmp"
                )
                temporary_file.write_text(
                    json.dumps(saved_history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary_file.replace(history_file)
        except (OSError, TypeError, UnicodeError, ValueError) as error:
            # A history-file problem must not discard a successful LLM result.
            print(f"{self.TAG} Could not save prompt history: {error}")

    # ── Response helpers ─────────────────────────────────────────────

    @staticmethod
    def _flatten_text(value: Any) -> str:
        """Turn a provider payload (str, list of parts, dict) into plain text."""
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            parts = []
            for item in value:
                text = OpenAICompatibleLoader._flatten_text(item)
                if text:
                    parts.append(text)
            return "\n\n".join(parts)

        if isinstance(value, dict):
            for key in ("text", "summary", "content", "reasoning"):
                text = OpenAICompatibleLoader._flatten_text(value.get(key))
                if text:
                    return text

        return ""

    @classmethod
    def _extract_reasoning(cls, response_content: Any, data: dict) -> Tuple[str, str]:
        """Split the visible answer from the model's reasoning trace.

        Standard nanoGPT output keeps them apart:
            choices[0].message.content    -> response
            choices[0].message.reasoning  -> reasoning

        Fallbacks cover DeepSeek's native `reasoning_content`, structured
        `reasoning_details`, and <think> tags embedded in the text.
        """
        if not isinstance(response_content, str):
            response_content = cls._flatten_text(response_content)

        choices = data.get("choices") or []
        first_choice = (choices[0] if choices else None) or {}
        message = first_choice.get("message") or {}
        delta = first_choice.get("delta") or {}

        reasoning_content = ""

        for source in (message, delta, data):
            for key in ("reasoning", "reasoning_content", "reasoning_details"):
                candidate = cls._flatten_text(source.get(key))
                if candidate:
                    reasoning_content = candidate
                    break
            if reasoning_content:
                break

        # Strip <think> blocks from the visible text even when the provider
        # already returned them in a dedicated field.
        matches = cls.THINK_TAG_RE.findall(response_content)
        if matches:
            response_content = cls.THINK_TAG_RE.sub("", response_content).strip()
            if not reasoning_content:
                reasoning_content = "\n\n".join(
                    match.strip() for match in matches if match.strip()
                )

        # A model that hit the token limit inside an unclosed think block:
        # keep the partial trace out of the response pin.
        if not reasoning_content:
            open_tag = re.search(
                r"<think(?:ing)?\b[^>]*>",
                response_content,
                re.IGNORECASE,
            )
            if open_tag:
                reasoning_content = response_content[open_tag.end():].strip()
                response_content = response_content[:open_tag.start()].strip()

        return response_content, reasoning_content

    # ── Vision helpers ───────────────────────────────────────────────

    @classmethod
    def _encode_images(cls, image: Any) -> List[str]:
        """Encode a ComfyUI IMAGE batch into base64 JPEG data URLs.

        Frames are downscaled so the long side fits MAX_IMAGE_SIDE, keeping
        the request payload reasonable for OpenAI-compatible vision APIs.
        """
        if image is None:
            return []

        import numpy as np
        from PIL import Image

        try:
            import torch
            if isinstance(image, torch.Tensor):
                frames = image.detach().cpu().numpy()
            else:
                frames = np.asarray(image)
        except ImportError:
            frames = np.asarray(image)

        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.ndim != 4:
            raise ValueError(
                f"Unsupported image shape {tuple(frames.shape)}; "
                "expected a ComfyUI IMAGE tensor (B,H,W,C)."
            )

        data_urls: List[str] = []
        for frame in frames[: cls.MAX_IMAGES]:
            arr = (np.clip(frame, 0.0, 1.0) * 255.0).astype("uint8")

            if arr.ndim == 2:
                pil_image = Image.fromarray(arr, "L").convert("RGB")
            elif arr.shape[-1] == 4:
                pil_image = Image.fromarray(arr, "RGBA").convert("RGB")
            elif arr.shape[-1] == 3:
                pil_image = Image.fromarray(arr, "RGB")
            else:
                raise ValueError(f"Unsupported channel count: {arr.shape}")

            if max(pil_image.size) > cls.MAX_IMAGE_SIDE:
                pil_image.thumbnail(
                    (cls.MAX_IMAGE_SIDE, cls.MAX_IMAGE_SIDE),
                    Image.LANCZOS,
                )

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=cls.JPEG_QUALITY)
            data_urls.append(
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )

        return data_urls

    @staticmethod
    def _build_user_content(prompt: str, image_urls: List[str]) -> Any:
        """Multimodal user message, or the plain prompt string without images."""
        if not image_urls:
            return prompt

        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        ]
        if prompt:
            content.append({"type": "text", "text": prompt})
        return content

    # ── Main ─────────────────────────────────────────────────────────

    def generate(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        prompt: str,
        enable_memory: bool = False,
        thinking: bool = False,
        reasoning_effort: str = "high",
        image: Any = None,
        unique_id: str = "",
    ) -> dict:
        base_url = base_url.strip()
        api_key = api_key.strip()
        model = model.strip()
        prompt = prompt.strip()

        if not base_url:
            return self._error_result("Base URL is missing.")

        if not model:
            return self._error_result("Model name is missing.")

        try:
            image_urls = self._encode_images(image)
        except Exception as error:
            return self._error_result(f"Image encoding failed: {error}")

        if not prompt and not image_urls:
            return self._error_result("Prompt is empty.")

        resolved_system_prompt = load_system_prompt(system_prompt)

        if system_prompt == "None (empty)":
            resolved_system_prompt = None

        if not enable_memory:
            self._clear_memory(unique_id)
            memory_key = None
        else:
            memory_key = self._memory_key(
                unique_id, base_url, model, resolved_system_prompt or "", api_key
            )

        messages: List[Dict[str, Any]] = []

        if resolved_system_prompt:
            messages.append({
                "role": "system",
                "content": resolved_system_prompt,
            })

        if enable_memory:
            messages.extend(self._get_history(memory_key))

        messages.append({
            "role": "user",
            "content": self._build_user_content(prompt, image_urls),
        })

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # nanoGPT's standard switch for thinking.  The legacy `thinking: true`
        # flag was a proxy-specific extension and is ignored by the standard
        # endpoint, which is why the reasoning pin used to stay empty.
        if thinking:
            payload["reasoning_effort"] = (
                (reasoning_effort or "").strip().lower() or "high"
            )

        start = time.time()

        print(
            f"{self.TAG} {time.strftime('%H:%M:%S')} → {model} | "
            f"memory={enable_memory} | thinking={thinking}"
            f" ({payload.get('reasoning_effort', 'off')})"
            + (f" | img(x{len(image_urls)})" if image_urls else "")
        )

        if resolved_system_prompt:
            print(f"   sys│ {resolved_system_prompt[:60].replace(chr(10), ' ')}...")

        print(f"   in │ {prompt[:80].replace(chr(10), ' ')}...")

        try:
            client = LLMClient(base_url, api_key)
            response_content, data = client.chat(payload)

            response_content = response_content or ""
            data = data or {}

            response_content, reasoning_content = self._extract_reasoning(
                response_content,
                data,
            )

            usage = data.get("usage") or {}

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = (
                details.get("reasoning_tokens")
                or usage.get("reasoning_tokens")
                or 0
            )

            elapsed_ms = int((time.time() - start) * 1000)

            print(f"   out│ {response_content[:80].replace(chr(10), ' ')}...")

            print(
                f"{self.TAG} {time.strftime('%H:%M:%S')} ← "
                f"{input_tokens}/{output_tokens}t "
                f"(reasoning {reasoning_tokens}t) ({elapsed_ms}ms)"
            )

            if reasoning_content:
                print(
                    f"{self.TAG} 🧠 Reasoning captured "
                    f"({len(reasoning_content)} chars):"
                )
                print(reasoning_content[:150] + "...")
            elif thinking:
                choices = data.get("choices") or []
                first_choice = (choices[0] if choices else None) or {}
                message = first_choice.get("message") or {}
                print(
                    f"{self.TAG} ⚠ Thinking is ON but the provider returned no "
                    f"reasoning. message fields: "
                    f"{', '.join(sorted(message.keys())) or 'none'}"
                )

            if enable_memory:
                self._remember_turn(
                    memory_key,
                    prompt,
                    response_content,
                )

            self._save_persistent_turn(
                prompt=prompt,
                response=response_content,
            )

            return self._success(
                text=response_content,
                reasoning=reasoning_content,
                in_tok=input_tokens,
                out_tok=output_tokens,
                reasoning_tok=reasoning_tokens,
            )

        except Exception as e:
            return self._error_result(str(e))


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLoader": OpenAICompatibleLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLoader": "LLM API Adapter",
}
