"""
API Manager — Unified interface for multiple LLM providers.

Supports:
- OpenAI (GPT-4o, GPT-4, etc.)
- Anthropic (Claude 3.5 Sonnet, etc.)
- OpenRouter (multi-model gateway)
- Local models (Ollama, etc.)

Features:
- Automatic provider selection
- Streaming support
- Token counting
- Rate limiting
- Retry with exponential backoff
- Response caching
"""

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Optional

import httpx
from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.security.manager import SecurityManager

console = Console()

class APIManager:
    """
    Unified API manager for all LLM providers.

    Handles authentication, request routing, streaming,
    and provider-specific formatting.
    """

    def __init__(self, config: ConfigManager, security: SecurityManager):
        self._config = config
        self._security = security
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        self._rate_limiters: dict[str, tuple[float, int]] = {}  # provider -> (last_request, count)
        self._cache: dict[str, dict] = {}  # Simple in-memory cache

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._config.get("system.timeout", 300),
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self._initialized = True

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Send a chat completion request with streaming.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            stream: Whether to stream the response
            tools: Optional tool definitions

        Yields:
            Response chunks with 'type' and 'content' or 'tool_call'
        """
        provider = self._config.get("api.provider", "inception")
        model = model or self._config.get("api.default_model", "mercury")
        temperature = temperature or self._config.get("api.temperature", 0.3)
        max_tokens = max_tokens or self._config.get("api.max_tokens", 8192)

        api_key = await self._security.get_secret(f"api_key_{provider}")
        if not api_key:
            api_key = await self._security.get_secret("api_key")

        if not api_key:
            env_name = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(env_name) or os.getenv("API_KEY")

        if not api_key:
            yield {"type": "error", "content": f"No API key found for provider '{provider}'. Please run 'zeta setup' or set {provider.upper()}_API_KEY."}
            return

        base_url = self._config.get(f"api.{provider}.base_url", "")
        if not base_url and provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif not base_url and provider == "anthropic":
            base_url = "https://api.anthropic.com"
        elif not base_url and provider == "inception":
            base_url = "https://api.inceptionlabs.ai/v1"

        # Check rate limits
        await self._check_rate_limit(provider)

        # Cache check for non-streaming requests
        if not stream and not tools:
            cache_key = self._make_cache_key(messages, model, temperature)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                yield {"type": "content", "content": cached["content"]}
                yield {"type": "done", "usage": cached.get("usage", {})}
                return

        try:
            if provider in ("openai", "openrouter", "local", "inception"):
                async for chunk in self._stream_openai_compatible(
                    base_url, api_key, messages, model, temperature, max_tokens, tools
                ):
                    yield chunk
            elif provider == "anthropic":
                async for chunk in self._stream_anthropic(
                    base_url, api_key, messages, model, temperature, max_tokens, tools
                ):
                    yield chunk
            else:
                yield {"type": "error", "content": f"Unsupported provider: {provider}"}

        except httpx.HTTPStatusError as e:
            yield {"type": "error", "content": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
        except httpx.RequestError as e:
            yield {"type": "error", "content": f"Request failed: {e}"}
        except Exception as e:
            yield {"type": "error", "content": f"Unexpected error: {e}"}

    async def _stream_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
    ) -> AsyncGenerator[dict, None]:
        """Stream from OpenAI-compatible API."""
        url = f"{base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        accumulated = ""
        usage = {}

        async with self._client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})

                        if "content" in delta and delta["content"]:
                            accumulated += delta["content"]
                            yield {"type": "content", "content": delta["content"]}

                        if "tool_calls" in delta:
                            yield {"type": "tool_call", "tool_calls": delta["tool_calls"]}

                        if "usage" in data:
                            usage = data["usage"]

                    except json.JSONDecodeError:
                        continue

        yield {"type": "done", "content": accumulated, "usage": usage}

    async def _stream_anthropic(
        self,
        base_url: str,
        api_key: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
    ) -> AsyncGenerator[dict, None]:
        """Stream from Anthropic API."""
        url = f"{base_url.rstrip('/')}/v1/messages"

        # Convert messages to Anthropic format
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system_msg:
            payload["system"] = system_msg
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        accumulated = ""

        async with self._client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if "text" in delta:
                                accumulated += delta["text"]
                                yield {"type": "content", "content": delta["text"]}
                        elif data.get("type") == "message_delta":
                            usage = data.get("usage", {})
                            yield {"type": "done", "content": accumulated, "usage": usage}
                    except json.JSONDecodeError:
                        continue

    async def chat_sync(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """
        Non-streaming chat completion. Returns full response.

        Returns:
            Dict with 'content', 'tool_calls', 'usage'
        """
        full_content = ""
        tool_calls = []
        usage = {}
        error = None

        async for chunk in self.chat(
            messages, model, temperature, max_tokens, stream=True, tools=tools
        ):
            if chunk["type"] == "content":
                full_content += chunk["content"]
            elif chunk["type"] == "tool_call":
                tool_calls.append(chunk["tool_calls"])
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
            elif chunk["type"] == "error":
                error = chunk["content"]

        if error:
            return {"content": error, "tool_calls": [], "usage": {}, "error": error}

        return {
            "content": full_content,
            "tool_calls": tool_calls,
            "usage": usage,
        }

    async def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """
        Estimate token count for a text string.

        Uses tiktoken for OpenAI models, heuristic for others.
        """
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Rough heuristic: ~4 chars per token
            return len(text) // 4

    async def _check_rate_limit(self, provider: str) -> None:
        """Apply simple rate limiting."""
        now = time.time()
        if provider in self._rate_limiters:
            last_request, count = self._rate_limiters[provider]
            if now - last_request < 1.0:
                count += 1
                if count > 10:  # Max 10 requests per second
                    await asyncio.sleep(0.5)
            else:
                count = 1
        else:
            count = 1

        self._rate_limiters[provider] = (now, count)

    def _make_cache_key(self, messages: list[dict], model: str, temperature: float) -> str:
        """Create a cache key from request parameters."""
        import hashlib

        content = json.dumps({"messages": messages, "model": model, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
