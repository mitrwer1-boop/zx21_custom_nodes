"""
LLMClient — Robust HTTP client for OpenAI-compatible LLM APIs.

Single source of truth for all API calls in zx21_custom_nodes.
Features:
  - Smart retry with Retry-After header respect + exponential backoff + jitter
  - SSL compatibility for Chinese API providers
  - Request size logging
"""

import math
import time
import random
import json
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Any, Tuple, Optional


# ─── HTTP Client ─────────────────────────────────────────────────────────────

# Shared SSL contexts (created once, reused across all calls)
_ssl_ctx_verify = None
_ssl_ctx_noverify = None

def _get_ssl_context(skip_verify: bool = False) -> ssl.SSLContext:
    """Get or create an SSL context. Verification is ON by default."""
    global _ssl_ctx_verify, _ssl_ctx_noverify
    if skip_verify:
        if _ssl_ctx_noverify is None:
            _ssl_ctx_noverify = ssl.create_default_context()
            _ssl_ctx_noverify.check_hostname = False
            _ssl_ctx_noverify.verify_mode = ssl.CERT_NONE
            _ssl_ctx_noverify.set_ciphers("DEFAULT")
            _ssl_ctx_noverify.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0)
        return _ssl_ctx_noverify
    else:
        if _ssl_ctx_verify is None:
            _ssl_ctx_verify = ssl.create_default_context()
            _ssl_ctx_verify.set_ciphers("DEFAULT")
        return _ssl_ctx_verify


def _normalize_url(base_url: str, path: str = "/chat/completions") -> str:
    """Normalize base URL and append path, avoiding double slashes."""
    url = base_url.rstrip("/")
    if url.endswith(path):
        return url
    return url + path


def _parse_retry_after(headers) -> Optional[float]:
    """Extract Retry-After from response headers, in seconds.

    Handles both forms the RFC allows: delta-seconds ("120") and HTTP-date
    ("Wed, 21 Oct 2026 07:28:00 GMT").  Returns None when absent/unparsable.
    """
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None

    try:  # delta-seconds form
        value = float(retry_after)
    except ValueError:
        value = None

    if value is not None:
        # NaN/Infinity/negative values would break time.sleep() downstream
        if not math.isfinite(value) or value < 0:
            return None
        return value

    try:  # HTTP-date form
        target = parsedate_to_datetime(retry_after)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _is_retryable(code: int) -> bool:
    """Check if an HTTP status code warrants a retry."""
    return code == 429 or (500 <= code < 600)


class LLMClient:
    """
    Robust HTTP client for OpenAI-compatible chat completions API.

    Usage:
        client = LLMClient(base_url, api_key)
        content, data = client.chat(payload)
    """

    TAG = "[zx21]"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_retries: int = 3,
        timeout: int = 180,
        skip_ssl_verify: bool = False,
    ):
        self.url = _normalize_url(base_url)
        self.base_url = self.url.replace("/chat/completions", "")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._ctx = _get_ssl_context(skip_ssl_verify)
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/event-stream",
            "Connection": "keep-alive"
        }

    def list_models(self) -> Dict[str, Any]:
        """
        Fetch available models using the /models endpoint.
        Used primarily for connectivity checks without consuming tokens.
        """
        headers = self._get_headers()
        models_url = f"{self.base_url}/models"
        
        req = urllib.request.Request(
            models_url, headers=headers, method="GET"
        )
        with urllib.request.urlopen(
            req, timeout=self.timeout, context=self._ctx
        ) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    def chat(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Send a chat completion request.

        Returns:
            Tuple of (response_content, full_response_data)

        Raises:
            Exception with structured error message if all retries fail.
        """
        headers = self._get_headers()

        data_bytes = json.dumps(payload).encode("utf-8")
        data_size_mb = len(data_bytes) / (1024 * 1024)
        if data_size_mb > 1:
            print(f"{self.TAG} Request payload size: {data_size_mb:.2f} MB")

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=data_bytes, headers=headers, method="POST"
                )
                with urllib.request.urlopen(
                    req, timeout=self.timeout, context=self._ctx
                ) as resp:
                    body = resp.read().decode("utf-8")
                    data = json.loads(body)

                    if "choices" not in data or len(data["choices"]) == 0:
                        raise ValueError(
                            f"API response missing 'choices'. Response: {json.dumps(data)[:300]}"
                        )

                    message = data["choices"][0].get("message") or {}
                    content = message.get("content")

                    if content is None:
                        content = ""

                    return content, data

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")

                if _is_retryable(e.code) and attempt < self.max_retries:
                    # Respect Retry-After header if present ("0" = retry now)
                    wait = _parse_retry_after(e.headers)
                    if wait is None:
                        wait = self._backoff(attempt, is_rate_limit=(e.code == 429))
                    print(
                        f"{self.TAG} HTTP {e.code} Error. "
                        f"Retrying {attempt + 1}/{self.max_retries} "
                        f"(wait {wait:.1f}s)..."
                    )
                    time.sleep(wait)
                    continue

                raise Exception(f"HTTP {e.code} | {error_body}")

            except urllib.error.URLError as e:
                last_error = e
                error_msg = str(e.reason)

                if self._is_connection_error(error_msg) and attempt < self.max_retries:
                    wait = self._backoff(attempt)
                    print(f"{self.TAG} Connection Error. Retrying {attempt + 1}/{self.max_retries} (wait {wait:.1f}s)...")
                    time.sleep(wait)
                    continue

                raise Exception(f"URLError | {error_msg}")

            except (TimeoutError, OSError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self._backoff(attempt)
                    print(f"{self.TAG} Timeout/IO Error. Retrying {attempt + 1}/{self.max_retries} (wait {wait:.1f}s)...")
                    time.sleep(wait)
                    continue
                raise Exception(f"TimeoutError | Request hung for over {self.timeout} seconds")

        # All retries exhausted
        if last_error:
            raise Exception(
                f"URLError | {str(getattr(last_error, 'reason', last_error))} "
                f"(Failed after {self.max_retries} retries)"
            )
        raise Exception("Unknown error occurred")

    @staticmethod
    def _backoff(attempt: int, is_rate_limit: bool = False) -> float:
        """Calculate wait time with exponential backoff + jitter."""
        base = 2 ** attempt
        if is_rate_limit:
            base = max(5, base * 3)  # Much longer for rate limits
        jitter = random.uniform(0, base * 0.3)
        return base + jitter

    @staticmethod
    def _is_connection_error(msg: str) -> bool:
        """Check if error message indicates a retryable connection issue."""
        keywords = ("Broken pipe", "Connection", "EOF", "reset by peer", "ECONNRESET")
        return any(kw.lower() in msg.lower() for kw in keywords)
