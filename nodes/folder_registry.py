import os
from typing import List

# ── Paths ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(__file__)  # Убрали "..", папки внутри ноды
_PROMPTS_DIR = os.path.join(_BASE_DIR, "prompts")
NO_SYSTEM_PROMPT = "None (empty)"


def _ensure_prompts_dir() -> None:
    """Create prompts directory if it doesn't exist."""
    if not os.path.exists(_PROMPTS_DIR):
        os.makedirs(_PROMPTS_DIR, exist_ok=True)
        # Create example prompts
        examples = {
            "default.txt": "You are a helpful AI assistant.",
            "coding.txt": "You are an expert programmer. Write clean, efficient, well-documented code.",
            "creative.txt": "You are a creative writer. Use vivid language and engaging narratives.",
            "concise.txt": "Answer concisely. No fluff, no markdown unless asked.",
        }
        for fname, content in examples.items():
            fpath = os.path.join(_PROMPTS_DIR, fname)
            if not os.path.exists(fpath):
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)


def system_prompt_options() -> List[str]:
    """Return list of available system prompt presets for dropdown."""
    _ensure_prompts_dir()
    options = [NO_SYSTEM_PROMPT]
    try:
        for fname in sorted(os.listdir(_PROMPTS_DIR)):
            if fname.lower().endswith(".txt"):
                name = fname[:-4]  # strip .txt
                if name not in options:
                    options.append(name)
    except Exception:
        pass
    return options


def full_system_prompt_path(preset_name: str) -> str:
    """Resolve preset name to absolute file path."""
    if preset_name == NO_SYSTEM_PROMPT:
        return ""
    return os.path.join(_PROMPTS_DIR, f"{preset_name}.txt")


def load_system_prompt(preset_name: str) -> str:
    """Load system prompt content from preset file."""
    if preset_name == NO_SYSTEM_PROMPT:
        return ""
    path = full_system_prompt_path(preset_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""