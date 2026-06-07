from lib.providers.services import service
import os
import asyncio
import socket
import time
from mindroot.lib.utils.backoff import ExponentialBackoff
import base64
from io import BytesIO
from openai import AsyncOpenAI
import json

# TCP_NODELAY: disable Nagle's algorithm to avoid up to 40ms buffering
# on small streaming token chunks over localhost TCP.
_TCP_NODELAY_OPT = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# Cache of AsyncOpenAI clients keyed by server URL
_client_cache = {}

def get_client(server_url, api_key):
    """Get or create a cached AsyncOpenAI client for the given server URL/API key."""
    import hashlib
    cache_key = (server_url, hashlib.sha256((api_key or "").encode("utf-8")).hexdigest())
    if cache_key not in _client_cache:
        import httpx
        transport = httpx.AsyncHTTPTransport(socket_options=[_TCP_NODELAY_OPT])
        http_client = httpx.AsyncClient(transport=transport)
        _client_cache[cache_key] = AsyncOpenAI(
            base_url=server_url,
            api_key=api_key,
            http_client=http_client
        )
    return _client_cache[cache_key]

# Backoff managers for different error types
_429_backoff = ExponentialBackoff(initial_delay=1.0, max_delay=30.0, factor=2.0, jitter=True)
_503_backoff = ExponentialBackoff(initial_delay=0.25, max_delay=30.0, factor=2.0, jitter=True)
_MAX_RETRIES = 8

def concat_text_lists(message):
    """Normalize legacy text-list content while preserving OpenAI multimodal blocks."""
    content = message.get('content')
    if isinstance(content, str):
        return message
    if not isinstance(content, list):
        return message

    # Preserve OpenAI multimodal content such as image_url/input_audio blocks.
    for item in content:
        if isinstance(item, dict) and item.get('type') in ('image_url', 'input_audio'):
            return message

    text_parts = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            if item.get('type') == 'text' and 'text' in item:
                text_parts.append(str(item.get('text') or ''))
            elif 'text' in item and len(item) == 1:
                text_parts.append(str(item.get('text') or ''))
            else:
                # Unknown structured content: preserve original rather than corrupting it.
                return message
        else:
            return message

    new_message = dict(message)
    new_message['content'] = "\n".join(p for p in text_parts if p)
    return new_message

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000,
                     temperature=0.0, max_tokens=500, num_gpu_layers=0):
    identifier = f"stream_chat_{model or 'default'}"
    try:
        print("mr_any_llm stream_chat (OpenAI compatible mode)")

        # Read server URL and API key from env vars (done here so mindroot can specialize them)
        server_url = os.environ.get("ANY_LLM_SERVER_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("ANY_LLM_API_KEY", "")

        client = get_client(server_url, api_key)

        model_name = model or "gpt-4o"

        messages = [concat_text_lists(m) for m in messages]

        # Parse extra_params from env var
        extra_params = {}
        extra_params_str = os.environ.get("ANY_LLM_EXTRA_PARAMS", "").strip()
        if extra_params_str:
            try:
                extra_params = json.loads(extra_params_str)
            except Exception as e:
                print(f"mr_any_llm: failed to parse ANY_LLM_EXTRA_PARAMS: {e}")

        # Retry logic with exponential backoff
        for attempt in range(_MAX_RETRIES):
            try:
                wait_429 = _429_backoff.get_wait_time(identifier)
                wait_503 = _503_backoff.get_wait_time(identifier)
                wait_time = max(wait_429, wait_503)

                if wait_time > 0:
                    print(f"mr_any_llm backoff: waiting {wait_time:.2f}s before attempt {attempt + 1}")
                    await asyncio.sleep(wait_time)

                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra_params
                )

                print("Opened stream with model:", model_name)

                _429_backoff.record_success(identifier)
                _503_backoff.record_success(identifier)
                break  # Success, exit retry loop

            except Exception as e:
                error_str = str(e)
                print(f'mr_any_llm error on attempt {attempt + 1}: {e}')

                if '429' in error_str:
                    _429_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        raise Exception(f"Max retries ({_MAX_RETRIES}) exceeded for 429 error: {e}")
                elif '503' in error_str:
                    _503_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        raise Exception(f"Max retries ({_MAX_RETRIES}) exceeded for 503 error: {e}")
                else:
                    raise

        async def content_stream(original_stream):
            pending_content = ''
            last_content_flush = time.monotonic()
            flush_interval = float(os.environ.get("AH_STREAM_FLUSH_INTERVAL", "0.025"))
            flush_chars = int(os.environ.get("AH_STREAM_FLUSH_CHARS", "512"))
            slow_after_chars = int(os.environ.get("AH_STREAM_FLUSH_SLOW_AFTER_CHARS", "2000"))
            slow_interval = float(os.environ.get("AH_STREAM_FLUSH_SLOW_INTERVAL", "0.5"))
            slow_flush_chars = int(os.environ.get("AH_STREAM_FLUSH_SLOW_CHARS", "4096"))
            very_slow_after_chars = int(os.environ.get("AH_STREAM_FLUSH_VERY_SLOW_AFTER_CHARS", "8000"))
            very_slow_interval = float(os.environ.get("AH_STREAM_FLUSH_VERY_SLOW_INTERVAL", "1.0"))
            very_slow_flush_chars = int(os.environ.get("AH_STREAM_FLUSH_VERY_SLOW_CHARS", "8192"))
            streamed_content_chars = 0
            def current_flush_interval():
                return very_slow_interval if streamed_content_chars >= very_slow_after_chars else slow_interval if streamed_content_chars >= slow_after_chars else flush_interval
            def current_flush_chars():
                return very_slow_flush_chars if streamed_content_chars >= very_slow_after_chars else slow_flush_chars if streamed_content_chars >= slow_after_chars else flush_chars
            async for chunk in original_stream:
                choices = getattr(chunk, 'choices', None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], 'delta', None)
                content = getattr(delta, 'content', None) if delta is not None else None
                if os.environ.get('AH_DEBUG') == 'True':
                    print('\033[92m' + str(content) + '\033[0m', end='')
                if content:
                    if flush_interval <= 0 or flush_chars <= 0:
                        yield content or ""
                    else:
                        pending_content += content or ""
                        now = time.monotonic()
                        interval = current_flush_interval()
                        chars = current_flush_chars()
                        stripped = pending_content.rstrip()
                        if (
                            len(pending_content) >= chars
                            or (now - last_content_flush) >= interval
                            or stripped.endswith(']')
                        ):
                            yield pending_content
                            streamed_content_chars += len(pending_content)
                            pending_content = ''
                            last_content_flush = now

            if pending_content:
                streamed_content_chars += len(pending_content)
                yield pending_content

        return content_stream(stream)

    except Exception as e:
        print('mr_any_llm error:', e)
        raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions"""
    return 4096, 4096, 16777216

@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    print("get service models!!")
    identifier = "get_service_models"
    try:
        print("ANY LLM 0000")
        server_url = os.environ.get("ANY_LLM_SERVER_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("ANY_LLM_API_KEY", "")
        print("ANY LLM 00")
        print("ANY_LLM_SERVER_URL", server_url)
        print("ANY_LLM_API_KEY", "<set>" if api_key else "<empty>")
        client = get_client(server_url, api_key)
        print("ANY LLM 11")
        for attempt in range(_MAX_RETRIES):
            try:
                print('ANY LLM 22')
                wait_429 = _429_backoff.get_wait_time(identifier)
                wait_503 = _503_backoff.get_wait_time(identifier)
                wait_time = max(wait_429, wait_503)
                print('33')
                if wait_time > 0:
                    print(f"mr_any_llm backoff: waiting {wait_time:.2f}s before attempt {attempt + 1}")
                    await asyncio.sleep(wait_time)

                print("Loading LLM models..")
                all_models = await client.models.list()
                print("models:")
                print(all_models)
                ids = [model.id for model in all_models.data]
                print("ids:")
                print(ids)
                _429_backoff.record_success(identifier)
                _503_backoff.record_success(identifier)
                return {'stream_chat': ids}

            except Exception as e:
                error_str = str(e)
                print(f'mr_any_llm models error on attempt {attempt + 1}: {e}')

                if '429' in error_str:
                    _429_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        return {'stream_chat': []}
                elif '503' in error_str:
                    _503_backoff.record_failure(identifier)
                    if attempt == _MAX_RETRIES - 1:
                        return {'stream_chat': []}
                else:
                    raise
    except Exception as e:
        return {'stream_chat': []}
