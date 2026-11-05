"""zx21_custom_nodes — OpenAI-compatible LLM nodes for ComfyUI.

Active nodes (nanoGPT standard output):
    OpenAICompatibleLoader   "LLM API Adapter"          text + vision + reasoning
    LLMModelSelector         "LLM Model Selector"       model list from the API
    PromptHistoryViewer      "Prompt History Viewer"

An optional, non-published `legacy/` subpackage (frozen pre-nanoGPT adapter
for a special-format DeepSeek proxy) is picked up automatically when the
folder is present in a local install.
"""

import asyncio
import threading
import time

from .nodes.api_client import LLMClient
from .nodes.model_selector import (
    NODE_CLASS_MAPPINGS as _SELECTOR_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _SELECTOR_DISPLAY,
    LLMModelSelector,
    record_models,
)
from .nodes.openai_compatible import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes.prompt_history import (
    NODE_CLASS_MAPPINGS as _HISTORY_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _HISTORY_DISPLAY,
)

NODE_CLASS_MAPPINGS.update(_HISTORY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_HISTORY_DISPLAY)
NODE_CLASS_MAPPINGS.update(_SELECTOR_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_SELECTOR_DISPLAY)

try:  # optional local-only legacy nodes
    from .legacy.nodes.openai_compatible import (
        NODE_CLASS_MAPPINGS as _LEGACY_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_DISPLAY,
    )
    from .legacy.nodes.prompt_history import (
        NODE_CLASS_MAPPINGS as _LEGACY_HISTORY_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_HISTORY_DISPLAY,
    )
except ImportError:
    _LEGACY_MAPPINGS = {}
    _LEGACY_DISPLAY = {}
    _LEGACY_HISTORY_MAPPINGS = {}
    _LEGACY_HISTORY_DISPLAY = {}

NODE_CLASS_MAPPINGS.update(_LEGACY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_LEGACY_DISPLAY)
NODE_CLASS_MAPPINGS.update(_LEGACY_HISTORY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_LEGACY_HISTORY_DISPLAY)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Small read-only endpoints so the frontend can refresh the history viewers
# and the model dropdown without queueing a node.
try:
    from aiohttp import web
    from server import PromptServer

    from .nodes.prompt_history import _load_history

    @PromptServer.instance.routes.get("/zx21/history")
    async def zx21_history(_request):
        return web.json_response(_load_history())

    try:  # legacy history endpoint only exists when legacy is installed
        from .legacy.nodes.prompt_history import (
            _load_history as _load_legacy_history,
        )

        @PromptServer.instance.routes.get("/zx21/legacy/history")
        async def zx21_legacy_history(_request):
            return web.json_response(_load_legacy_history())
    except ImportError:
        pass

    # ── Model list for the LLM Model Selector node ───────────────────

    _MODEL_CACHE = {}
    _MODEL_CACHE_TTL = 300.0
    _MODEL_CACHE_LOCK = threading.Lock()

    @PromptServer.instance.routes.post("/zx21/models")
    async def zx21_models(request):
        """Return the model ids an OpenAI-compatible server reports.

        Body: {"base_url": "...", "api_key": "...", "force": bool}
        Successful answers are recorded into the selector's option cache, so
        ComfyUI's server-side combo validation accepts the chosen id.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        base_url = str(body.get("base_url") or "").strip()
        api_key = str(body.get("api_key") or "").strip()
        force = bool(body.get("force"))

        if not base_url:
            return web.json_response({"models": [], "error": "base_url is empty"})

        now = time.time()

        if not force:
            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get((base_url, api_key))
            if cached and now - cached[0] < _MODEL_CACHE_TTL:
                return web.json_response({
                    "models": cached[1],
                    "count": len(cached[1]),
                    "cached": True,
                })

        try:
            # Run the blocking urllib call in a worker thread so the ComfyUI
            # event loop (shared with the whole UI) never stalls on a slow
            # or unreachable <base_url>/models.
            models = await asyncio.to_thread(
                LLMModelSelector.fetch_models, base_url, api_key
            )
        except Exception as error:
            print(f"[zx21] model list failed for {base_url}: {error}")
            return web.json_response({"models": [], "error": str(error)[:300]})

        record_models(base_url, models)

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[(base_url, api_key)] = (now, models)

        return web.json_response({"models": models, "count": len(models)})
except Exception as _route_error:  # pragma: no cover
    print(f"[zx21] server endpoints unavailable: {_route_error}")
