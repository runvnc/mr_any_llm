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
    """Get or create a cached AsyncOpenAI client for the given server URL."""
    if server_url not in _client_cache:
        import httpx
        transport = httpx.AsyncHTTPTransport(socket_options=[_TCP_NODELAY_OPT])
        http_client = httpx.AsyncClient(transport=transport)
        _client_cache[server_url] = AsyncOpenAI(
            base_url=server_url,
            api_key=api_key,
            http_client=http_client
        )
    return _client_cache[server_url]

# Backoff managers for different error types
_429_backoff = ExponentialBackoff(initial_delay=1.0, max_delay=30.0, factor=2.0, jitter=True)
_503_backoff = ExponentialBackoff(initial_delay=0.25, max_delay=30.0, factor=2.0, jitter=True)
_MAX_RETRIES = 8

def concat_text_lists(message):
    """Concatenate text lists into a single string"""
    out_str = ""
    if isinstance(message['content'], str):
        return message
    else:
        for item in message['content']:
            if isinstance(item, str):
                out_str += item + "\n"
            else:
                out_str += item['text'] + "\n"
    message.update({'content': out_str})
    return message

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
            async for chunk in original_stream:
                if os.environ.get('AH_DEBUG') == 'True':
                    print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content or ""

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
        server_url = os.environ.get("ANY_LLM_SERVER_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("ANY_LLM_API_KEY", "")
        client = get_client(server_url, api_key)

        for attempt in range(_MAX_RETRIES):
            try:
                wait_429 = _429_backoff.get_wait_time(identifier)
                wait_503 = _503_backoff.get_wait_time(identifier)
                wait_time = max(wait_429, wait_503)

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
