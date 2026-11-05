"""Model selector node: picks a model id from the provider's /models list."""

from typing import Any, Dict, List, Tuple

try:
    from .api_client import LLMClient
except ImportError:
    from api_client import LLMClient


# Known-good defaults so the node is usable even before the frontend has
# fetched anything (e.g. right after a server restart).
FALLBACK_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4.1-flash:thinking",
    "deepseek/deepseek-v4-flash-latest",
]

# base_url -> model ids, filled by the /zx21/models endpoint.
# INPUT_TYPES reads this cache, so ComfyUI's server-side "value not in list"
# validation accepts anything the frontend has fetched at least once.
_MODEL_LISTS: Dict[str, List[str]] = {}


def record_models(base_url: str, models: List[str]) -> None:
    """Store the ids returned for one base_url (used by INPUT_TYPES)."""
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url or not models:
        return
    _MODEL_LISTS[base_url] = list(models)


def all_model_options() -> List[str]:
    """Union of everything fetched so far, plus the fallback defaults."""
    merged: List[str] = list(FALLBACK_MODELS)
    for models in _MODEL_LISTS.values():
        for name in models:
            if name not in merged:
                merged.append(name)
    return merged


class LLMModelSelector:
    """Dropdown with the models the API server reports.

    The list is filled by the frontend (web/js/model_selector.js) through
    /zx21/models.  The node simply outputs the selected id as a
    STRING, so it can be plugged into the `model` input of the LLM node
    (right-click the model widget -> Convert to input).
    """

    TAG = "[zx21]"

    @classmethod
    def INPUT_TYPES(cls):
        options = all_model_options()
        default = FALLBACK_MODELS[0] if FALLBACK_MODELS[0] in options else (
            options[0] if options else ""
        )

        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "https://nano-gpt.com/api/v1",
                    "placeholder": "https://nano-gpt.com/api/v1",
                    "tooltip": "API base URL; the list is read from "
                               "<base_url>/models",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-...",
                    "tooltip": "API key. Optional for servers that list models "
                               "without auth (nanoGPT does).",
                }),
                "model": (options, {
                    "default": default,
                    "tooltip": "Filled from the API. Right-click the node and "
                               "pick 'Refresh model list' to re-fetch.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "get_model"
    CATEGORY = "zx21/LLM"

    @staticmethod
    def fetch_models(base_url: str, api_key: str) -> List[str]:
        """Read <base_url>/models and return sorted, de-duplicated ids."""
        client = LLMClient(base_url.strip(), api_key.strip(), timeout=30)
        data = client.list_models()

        entries: List[Any] = []
        if isinstance(data, dict):
            entries = data.get("data") or data.get("models") or []
        elif isinstance(data, list):
            entries = data

        ids: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict):
                value = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(value, str) and value:
                    ids.append(value)

        return sorted(dict.fromkeys(ids))

    def get_model(self, base_url: str, api_key: str, model: str) -> Tuple[str]:
        model = (model or "").strip()

        if not model or model.startswith("("):
            raise ValueError(
                "Model list is not loaded yet. Right-click the LLM Model "
                "Selector node and choose 'Refresh model list'."
            )

        return (model,)


NODE_CLASS_MAPPINGS = {
    "LLMModelSelector": LLMModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMModelSelector": "LLM Model Selector",
}
