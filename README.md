# zx21_custom_nodes

OpenAI-compatible LLM nodes for ComfyUI, built for
[nanoGPT](https://nano-gpt.com) (`https://nano-gpt.com/api/v1`, standard
chat-completions output) but working with any OpenAI-compatible API.

## Nodes (category `zx21/LLM`)

| Node | What it does |
| --- | --- |
| **LLM API Adapter** (`OpenAICompatibleLoader`) | Chat completion with two outputs: `response` + `reasoning`. Optional `image` input (ComfyUI IMAGE, up to 8 frames, auto-JPEG ≤1024px) for multimodal models. `thinking` sends `reasoning_effort`. Persistent prompt/response/reasoning history (5 entries). |
| **LLM Model Selector** (`LLMModelSelector`) | Dropdown filled from `<base_url>/models`. Output plugs into the adapter's `model` field (widgets and input sockets co-exist — no conversion needed). Right-click → «🔄 Refresh model list». |
| **Prompt History Viewer** (`PromptHistoryViewer`) | Shows the saved prompt/response/reasoning entries. |

## Setup

1. Copy into `ComfyUI/custom_nodes/`, restart ComfyUI, hard-refresh the browser.
2. Fill `base_url` / `api_key` (nanoGPT subscription endpoints
   `https://nano-gpt.com/api/subscription/v1` work too).
3. Pick a model via **LLM Model Selector** (nanoGPT lists models without a key).

Reasoning: enable *Thinking Mode* → the adapter sends `reasoning_effort` and the
model's trace arrives in `message.reasoning` (shown in the `reasoning` output,
the token readout and the history viewer).

## System prompts

The adapter reads presets from `nodes/prompts/*.txt` (created with four
examples on first run if the folder is empty — the prompt library itself is not
distributed with this repository). For image-prompt writing we recommend
[moody's prompt-writing system prompts](https://civitai.com/models/2976901/moody-krea2-prompt-writing-skill-llm-system-prompt);
drop the `.txt` files into `nodes/prompts/`.

## Notes

- Vision requires a multimodal model (e.g. `deepseek/deepseek-v4.1-flash`).
- History files (`nodes/openai_compatible_history.txt`) are local only and git-ignored.
- Server endpoints: `GET /zx21/history`, `POST /zx21/models`.
- Based on [ComfyUI-LLMs-Toolkit](https://github.com/HuangYuChuh/ComfyUI-LLMs-Toolkit) (AGPL-3.0, see LICENSE);
  most of the original code has since been rewritten.

## Tests

Standalone sanity tests (no ComfyUI needed, numpy/Pillow optional):

```bash
python tests/test_nodes.py
```
