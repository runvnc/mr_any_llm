# MindRoot Any LLM Plugin

A generic plugin for integrating any OpenAI-compatible LLM API into MindRoot.

## Features

- Streaming text generation
- Multimodal support (text + images)
- Configurable server URL and API key
- Extra parameters support (e.g., `extra_body` for provider-specific options)
- Per-server client caching
- Exponential backoff retry logic

## Installation

```bash
pip install -e .
```

## Configuration

The plugin uses the following environment variables:

- `ANY_LLM_SERVER_URL`: Base URL of the OpenAI-compatible API (e.g., `https://api.openai.com/v1`)
- `ANY_LLM_API_KEY`: API key for the service
- `AH_OVERRIDE_LLM_MODEL` (optional): Override the model name
- `ANY_LLM_EXTRA_PARAMS` (optional): JSON dict of extra parameters to pass to the API call
- `AH_DEBUG` (optional): Enable debug output when set to `True`

## Extra Params Example

To disable thinking on a provider that supports it:

```
ANY_LLM_EXTRA_PARAMS={"extra_body": {"enable_thinking": false}}
```

This dict is merged into the `chat.completions.create()` call alongside `model`, `messages`, etc.

## Services

### stream_chat
Streams text completions from any OpenAI-compatible API.

### format_image_message
Formats PIL images as base64 image_url content blocks.

### get_image_dimensions
Returns max supported image dimensions (4096x4096).

### get_service_models
Lists available models from the configured server.

## License

MIT License
