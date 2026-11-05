"""Standalone sanity tests for zx21_custom_nodes (no ComfyUI required).

Run from the repository root:  python tests/test_nodes.py
Image tests are skipped when numpy/Pillow are not installed.
"""
import base64
import io
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes.api_client import _normalize_url, _parse_retry_after
from nodes.openai_compatible import OpenAICompatibleLoader as Loader
from nodes.prompt_history import PromptHistoryViewer

CHECKS = 0


def check(name, condition, detail=""):
    global CHECKS
    assert condition, f"FAIL {name} {detail}"
    CHECKS += 1
    print(f"ok  {name}")


def test_reasoning_extraction():
    got = Loader._extract_reasoning(
        "OK", {"choices": [{"message": {"content": "OK", "reasoning": "step one"}}]}
    )
    check("nanoGPT message.reasoning", got == ("OK", "step one"))

    got = Loader._extract_reasoning(
        "OK", {"choices": [{"message": {"content": "OK", "reasoning_content": "legacy"}}]}
    )
    check("legacy reasoning_content", got == ("OK", "legacy"))

    got = Loader._extract_reasoning(
        "OK",
        {"choices": [{"message": {"content": "OK", "reasoning_details": [
            {"type": "reasoning.text", "text": "a"},
            {"type": "reasoning.text", "text": "b"}]}}]},
    )
    check("reasoning_details joined", got == ("OK", "a\n\nb"))

    got = Loader._extract_reasoning(
        "<think>hidden</think>Visible",
        {"choices": [{"message": {"content": "<think>hidden</think>Visible"}}]},
    )
    check("think tags stripped", got == ("Visible", "hidden"))

    got = Loader._extract_reasoning(
        "<think>partial", {"choices": [{"message": {"content": "<think>partial"}}]}
    )
    check("unclosed think kept out of response", got == ("", "partial"))

    got = Loader._extract_reasoning(
        None, {"choices": [{"message": {"content": None, "reasoning": "only thinking"}}]}
    )
    check("null content", got == ("", "only thinking"))

    got = Loader._extract_reasoning(
        "<think>tag</think>Answer",
        {"choices": [{"message": {"content": "<think>tag</think>Answer", "reasoning": "field"}}]},
    )
    check("api field wins over tags", got == ("Answer", "field"))


def test_history_clamp():
    select = PromptHistoryViewer._select_entry
    check("newest first", select([{"i": 0}, {"i": 1}, {"i": 2}], 0)["i"] == 2)
    check("second entry", select([{"i": 0}, {"i": 1}, {"i": 2}], 1)["i"] == 1)
    check("oldest of three", select([{"i": 0}, {"i": 1}, {"i": 2}], 4)["i"] == 0)
    check("oldest of two (was IndexError)", select([{"i": 0}, {"i": 1}], 4)["i"] == 0)
    check("single entry", select([{"i": 7}], 4)["i"] == 7)


def test_memory_key():
    loader = Loader()
    loader._MEMORY_STORE.clear()
    k1 = loader._memory_key("n1", "https://a/v1", "model-a", "sys", "key-1")
    k2 = loader._memory_key("n1", "https://a/v1", "model-b", "sys", "key-1")
    k3 = loader._memory_key("n1", "https://a/v1", "model-a", "sys", "key-2")
    check("model switch isolates context", k1 != k2)
    check("account switch isolates context", k1 != k3 and k2 != k3)
    check("raw api key never stored in the key",
          "key-1" not in str(k1) and "key-2" not in str(k3))
    loader._remember_turn(k1, "hi", "hello")
    loader._remember_turn(k2, "other", "context")
    check("memory stored", len(loader._get_history(k1)) == 2)
    check("model switch sees no foreign context",
          all(m["content"] != "hi" for m in loader._get_history(k2)))
    loader._clear_memory("n1")
    check("clear drops every config of the node", loader._MEMORY_STORE == {})
    check("empty key returns no history", loader._get_history(None) == [])


def test_url_and_retry():
    check("normalize appends path",
          _normalize_url("http://h/v1") == "http://h/v1/chat/completions")
    check("normalize keeps full path",
          _normalize_url("http://h/v1/chat/completions") == "http://h/v1/chat/completions")

    check("retry numeric", _parse_retry_after({"Retry-After": "7"}) == 7.0)
    check("retry absent", _parse_retry_after({}) is None)
    check("retry garbage", _parse_retry_after({"Retry-After": "soon"}) is None)
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    delay = _parse_retry_after({"Retry-After": when})
    check("retry http-date", delay is not None and 0 <= delay <= 35, f"delay={delay}")


def test_content_building():
    check("text only", Loader._build_user_content("hi", []) == "hi")
    content = Loader._build_user_content("hi", ["data:x", "data:y"])
    check("multimodal order",
          [p["type"] for p in content] == ["image_url", "image_url", "text"]
          and content[-1]["text"] == "hi")
    check("image only", all(p["type"] == "image_url"
                            for p in Loader._build_user_content("", ["data:x"])))


def test_image_encoding():
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("skip image encoding (numpy/Pillow not installed)")
        return

    frame = np.zeros((1, 64, 64, 3), dtype=np.float32)
    frame[0, :, :, 0] = 0.86
    urls = Loader._encode_images(frame)
    check("single data url",
          len(urls) == 1 and urls[0].startswith("data:image/jpeg;base64,"))
    image = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
    check("decoded size/mode", image.size == (64, 64) and image.mode == "RGB")

    big = np.zeros((1, 1000, 2000, 3), dtype=np.float32)
    urls = Loader._encode_images(big)
    size = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1]))).size
    check("long side downscaled to 1024", max(size) == 1024, f"size={size}")


def test_http_routes():
    """Offline test of the aiohttp handlers with stubbed ComfyUI server bits."""
    import asyncio
    import importlib.util
    import types

    routes = {}

    class _Routes:
        def get(self, path):
            def deco(fn):
                routes[("GET", path)] = fn
                return fn
            return deco

        def post(self, path):
            def deco(fn):
                routes[("POST", path)] = fn
                return fn
            return deco

    server_stub = types.ModuleType("server")
    server_stub.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_Routes()))
    aiohttp_stub = types.ModuleType("aiohttp")

    class _Resp:
        def __init__(self, payload):
            self.payload = payload

    aiohttp_stub.web = types.SimpleNamespace(
        json_response=lambda payload, **kw: _Resp(payload))
    sys.modules["server"] = server_stub
    sys.modules["aiohttp"] = aiohttp_stub

    spec = importlib.util.spec_from_file_location(
        "zx21_custom_nodes", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["zx21_custom_nodes"] = pkg
    spec.loader.exec_module(pkg)

    expected = {("GET", "/zx21/history"), ("POST", "/zx21/models")}
    if (ROOT / "legacy").is_dir():  # optional local-only subpackage
        expected.add(("GET", "/zx21/legacy/history"))
    check("routes registered", set(routes) == expected, f"got {sorted(routes)}")

    async def _body(payload):
        return payload

    async def _call(path, payload):
        request = types.SimpleNamespace(json=lambda: _body(payload))
        response = await routes[("POST", path)](request)
        return response.payload

    # happy path + cache (fetch_models is stubbed: no network in tests)
    pkg.LLMModelSelector.fetch_models = staticmethod(
        lambda base_url, api_key: ["model-a", "model-b"])
    first = asyncio.run(_call("/zx21/models",
                              {"base_url": "https://fake/v1", "api_key": "k"}))
    check("models endpoint returns list", first["models"] == ["model-a", "model-b"])
    second = asyncio.run(_call("/zx21/models",
                               {"base_url": "https://fake/v1", "api_key": "k"}))
    check("models endpoint cached", second.get("cached") is True)

    # error paths
    def _boom(base_url, api_key):
        raise RuntimeError("boom")

    pkg.LLMModelSelector.fetch_models = staticmethod(_boom)
    failed = asyncio.run(_call("/zx21/models",
                               {"base_url": "https://fake2/v1", "force": True}))
    check("models endpoint error path",
          failed["models"] == [] and "boom" in failed["error"])
    empty = asyncio.run(_call("/zx21/models", {}))
    check("empty base_url rejected", empty["error"] == "base_url is empty")

    history = asyncio.run(routes[("GET", "/zx21/history")](None))
    check("history endpoint returns list", isinstance(history.payload, list))


if __name__ == "__main__":
    test_reasoning_extraction()
    test_history_clamp()
    test_memory_key()
    test_url_and_retry()
    test_content_building()
    test_image_encoding()
    test_http_routes()
    print(f"\nall {CHECKS} checks passed")
