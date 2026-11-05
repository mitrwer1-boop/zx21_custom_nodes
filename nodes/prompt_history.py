"""Viewer node for the persistent prompt history saved by the LLM adapter."""

import json
from pathlib import Path

try:
    from .openai_compatible import OpenAICompatibleLoader
except ImportError:
    from openai_compatible import OpenAICompatibleLoader

HISTORY_FILE = OpenAICompatibleLoader._PERSISTENT_HISTORY_FILE

# Dropdown labels; index 0 is always the newest entry on disk.
ENTRY_OPTIONS = [
    "1 (newest)",
    "2",
    "3",
    "4",
    "5 (oldest)",
]


def _load_history():
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (OSError, ValueError) as error:
        print(f"[zx21] Could not read prompt history: {error}")
    return []


class PromptHistoryViewer:
    """Shows the last saved prompts; outputs the selected prompt/response."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entry": (ENTRY_OPTIONS, {
                    "default": ENTRY_OPTIONS[0],
                    "tooltip": "Which saved prompt to output. 1 = newest. "
                               "Re-run the node to refresh after a new request.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "response")
    FUNCTION = "view"
    CATEGORY = "zx21/LLM"
    OUTPUT_NODE = True

    @staticmethod
    def _preview(text: str, width: int = 120) -> str:
        one_line = " ".join((text or "").split())
        return one_line[:width] + ("..." if len(one_line) > width else "")

    @staticmethod
    def _select_entry(history: list, index: int) -> dict:
        """Newest-first selection, clamped to the oldest available entry."""
        ordered_history = list(reversed(history))
        return ordered_history[min(index, len(ordered_history) - 1)]

    def view(self, entry: str) -> dict:
        history = _load_history()

        # Sidebar-style list of everything on disk, newest first.
        lines = []
        for i, item in enumerate(reversed(history), start=1):
            lines.append(
                f"#{i} [{item.get('timestamp', '?')}] "
                f"{self._preview(item.get('prompt', ''))}"
            )
        sidebar = "\n".join(lines) or "(history is empty)"

        try:
            index = ENTRY_OPTIONS.index(entry)
        except ValueError:
            index = 0

        if not history:
            return {
                "ui": {"text": [sidebar]},
                "result": ("", ""),
            }

        # Newest first, clamped to the oldest entry that actually exists.
        item = self._select_entry(history, index)

        preview_text = (
            f"{sidebar}\n\n"
            f"── Selected entry ──\n"
            f"Time:     {item.get('timestamp', '?')}\n"
            f"Prompt:   {item.get('prompt', '')}\n\n"
            f"Response: {item.get('response', '')}\n\n"
            f"Reasoning: {item.get('reasoning', '') or '(none captured)'}"
        )

        return {
            "ui": {"text": [preview_text]},
            "result": (item.get("prompt", ""), item.get("response", "")),
        }


NODE_CLASS_MAPPINGS = {
    "PromptHistoryViewer": PromptHistoryViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptHistoryViewer": "Prompt History Viewer",
}
