# OpenRouter Integration Setup

This guide explains how to set up and use the OpenRouter client with Trae Agent.

## What is OpenRouter?

OpenRouter is a unified API that provides access to multiple AI models from different providers (OpenAI, Anthropic, Meta, Google, etc.) through a single OpenAI-compatible interface. This allows you to:

- Access models from multiple providers without managing separate API keys
- Switch between models easily
- Compare model performance
- Access models that might not be directly available in your region

## Setup Instructions

### 1. Get an OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Go to [API Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Add credits to your account for usage

### 2. Configure Environment Variable

Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

Or add it to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export OPENROUTER_API_KEY="your_openrouter_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Update Configuration

Update your `trae_config.json` to use OpenRouter:

```json
{
  "default_provider": "openrouter",
  "max_steps": 20,
  "enable_lakeview": true,
  "model_providers": {
    "openrouter": {
      "api_key": "your_openrouter_api_key",
      "model": "anthropic/claude-3.5-sonnet",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 0,
      "max_retries": 10
    }
  }
}
```

## Available Models

OpenRouter provides access to many models. Here are some popular options:

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - Latest Claude model, excellent for coding
- `anthropic/claude-3-haiku` - Fast and cost-effective
- `anthropic/claude-3-opus` - Most capable Claude model

### OpenAI Models
- `openai/gpt-4o` - Latest GPT-4 model
- `openai/gpt-4o-mini` - Cost-effective GPT-4 variant
- `openai/gpt-4-turbo` - Previous generation GPT-4

### Meta Models
- `meta-llama/llama-3.1-405b-instruct` - Large Llama model
- `meta-llama/llama-3.1-70b-instruct` - Medium Llama model
- `meta-llama/llama-3.1-8b-instruct` - Small, fast Llama model

### Google Models
- `google/gemini-pro-1.5` - Google's latest model
- `google/gemini-flash-1.5` - Fast Google model

### Other Models
- `qwen/qwen-2.5-72b-instruct` - Alibaba's Qwen model
- `deepseek/deepseek-chat` - DeepSeek's chat model
- `mistralai/mistral-large` - Mistral's large model

For a complete list, visit: https://openrouter.ai/models

## Testing the Setup

Run the test script to verify your setup:

```bash
python test_openrouter.py
```

This will:
1. Test basic chat functionality
2. Check tool calling support
3. Display usage information

## Usage Examples

### Basic Usage

```python
from trae_agent.utils.llm_client import LLMClient
from trae_agent.utils.config import ModelParameters
from trae_agent.utils.llm_basics import LLMMessage

# Configure model parameters
model_params = ModelParameters(
    model="anthropic/claude-3.5-sonnet",
    api_key="your_openrouter_api_key",
    max_tokens=1000,
    temperature=0.7,
    top_p=1.0,
    top_k=0,
    parallel_tool_calls=False,
    max_retries=3
)

# Create client
client = LLMClient("openrouter", model_params)

# Send message
messages = [LLMMessage(role="user", content="Hello!")]
response = client.chat(messages, model_params)
print(response.content)
```

### Using Different Models

You can easily switch between models by changing the model parameter:

```python
# Use GPT-4
model_params.model = "openai/gpt-4o"

# Use Llama
model_params.model = "meta-llama/llama-3.1-70b-instruct"

# Use Gemini
model_params.model = "google/gemini-pro-1.5"
```

## Features

### ✅ Supported Features
- Chat completions
- Tool calling (for supported models)
- Streaming responses
- Temperature and top_p controls
- Token usage tracking
- Error handling and retries
- Trajectory recording

### 🔄 OpenAI Compatibility
The OpenRouter client uses OpenAI's API format, so it's compatible with:
- OpenAI SDK
- Tool calling schemas
- Message formats
- Response structures

## Pricing

OpenRouter uses a pay-per-use model with different pricing for each model. Check current pricing at: https://openrouter.ai/models

Generally:
- Smaller models (8B-13B parameters) are very cost-effective
- Medium models (70B parameters) offer good balance
- Large models (400B+ parameters) are most capable but more expensive

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `OPENROUTER_API_KEY` is set correctly
   - Check that your API key is valid on OpenRouter dashboard
   - Make sure you have credits in your account

2. **Model Not Found**
   - Verify the model name is correct (case-sensitive)
   - Check if the model is available on OpenRouter
   - Some models may have regional restrictions

3. **Rate Limiting**
   - OpenRouter has rate limits per model
   - The client includes automatic retry logic
   - Consider using smaller models for high-frequency requests

4. **Tool Calling Issues**
   - Not all models support tool calling
   - Check `client.supports_tool_calling(model_params)` first
   - Use models like Claude 3.5 Sonnet or GPT-4 for tool calling

### Getting Help

- OpenRouter Documentation: https://openrouter.ai/docs
- OpenRouter Discord: https://discord.gg/fVyRaUDgxW
- Model-specific documentation from original providers

## Security Notes

- Never commit API keys to version control
- Use environment variables for API keys
- Monitor your usage on the OpenRouter dashboard
- Set up billing alerts to avoid unexpected charges
- Rotate API keys regularly for security