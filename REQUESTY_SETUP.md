# Requesty Integration Setup

This guide explains how to set up and use the Requesty client with Trae Agent.

## What is Requesty?

Requesty is a unified API router that provides access to multiple AI models from different providers through a single OpenAI-compatible interface. Similar to OpenRouter, Requesty allows you to:

- Access models from multiple providers without managing separate API keys
- Switch between models easily
- Compare model performance
- Route requests intelligently based on availability and cost
- Access models that might not be directly available in your region

## Setup Instructions

### 1. Get a Requesty API Key

1. Visit [Requesty](https://requesty.ai/)
2. Sign up for an account
3. Navigate to your API keys section
4. Create a new API key
5. Add credits to your account for usage

### 2. Configure Environment Variables

Set your Requesty API key as an environment variable:

```bash
export REQUESTY_API_KEY="your_requesty_api_key_here"
```

#### Optional Headers (Recommended)

Requesty supports optional headers for better tracking and analytics:

```bash
export REQUESTY_SITE_URL="https://yoursite.com"  # Your site URL for referer tracking
export REQUESTY_SITE_NAME="Your App Name"        # Your application name
```

Add these to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export REQUESTY_API_KEY="your_requesty_api_key_here"' >> ~/.zshrc
echo 'export REQUESTY_SITE_URL="https://yoursite.com"' >> ~/.zshrc
echo 'export REQUESTY_SITE_NAME="Your App Name"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Update Configuration

Update your `trae_config.json` to use Requesty:

```json
{
  "default_provider": "requesty",
  "max_steps": 20,
  "enable_lakeview": true,
  "model_providers": {
    "requesty": {
      "api_key": "your_requesty_api_key",
      "model": "openai/gpt-4o",
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

Requesty provides access to many models using the `provider/model` format. Here are some popular options:

### OpenAI Models
- `openai/gpt-4o` - Latest GPT-4 model
- `openai/gpt-4o-mini` - Cost-effective GPT-4 variant
- `openai/gpt-4-turbo` - Previous generation GPT-4
- `openai/gpt-3.5-turbo` - Fast and economical

### Anthropic Models
- `anthropic/claude-3.5-sonnet` - Latest Claude model, excellent for coding
- `anthropic/claude-3-haiku` - Fast and cost-effective
- `anthropic/claude-3-opus` - Most capable Claude model

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

For a complete list of available models, check the Requesty documentation.

## Testing the Setup

Run the test script to verify your setup:

```bash
python test_requesty.py
```

This will:
1. Test basic chat functionality
2. Check tool calling support
3. Verify optional headers configuration
4. Display usage information

## Usage Examples

### Basic Usage

```python
from trae_agent.utils.llm_client import LLMClient
from trae_agent.utils.config import ModelParameters
from trae_agent.utils.llm_basics import LLMMessage

# Configure model parameters
model_params = ModelParameters(
    model="openai/gpt-4o",
    api_key="your_requesty_api_key",
    max_tokens=1000,
    temperature=0.7,
    top_p=1.0,
    top_k=0,
    parallel_tool_calls=False,
    max_retries=3
)

# Create client
client = LLMClient("requesty", model_params)

# Send message
messages = [LLMMessage(role="user", content="Hello!")]
response = client.chat(messages, model_params)
print(response.content)
```

### Using Different Models

You can easily switch between models by changing the model parameter:

```python
# Use Claude
model_params.model = "anthropic/claude-3.5-sonnet"

# Use Llama
model_params.model = "meta-llama/llama-3.1-70b-instruct"

# Use Gemini
model_params.model = "google/gemini-pro-1.5"
```

### With Optional Headers

The client automatically includes optional headers if environment variables are set:

```python
import os

# Set optional headers (or use environment variables)
os.environ["REQUESTY_SITE_URL"] = "https://myapp.com"
os.environ["REQUESTY_SITE_NAME"] = "My AI App"

# Client will automatically include these headers
client = LLMClient("requesty", model_params)
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
- Optional headers for tracking

### 🔄 OpenAI Compatibility
The Requesty client uses OpenAI's API format, so it's compatible with:
- OpenAI SDK
- Tool calling schemas
- Message formats
- Response structures

### 📊 Enhanced Features
- **HTTP-Referer Header**: Helps with request tracking and analytics
- **X-Title Header**: Identifies your application in Requesty's dashboard
- **Intelligent Routing**: Requesty can route requests based on availability and performance

## Configuration Options

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REQUESTY_API_KEY` | Yes | Your Requesty API key |
| `REQUESTY_SITE_URL` | No | Your site URL for referer tracking |
| `REQUESTY_SITE_NAME` | No | Your application name for identification |

### Model Parameters

```python
ModelParameters(
    model="openai/gpt-4o",           # Model in provider/model format
    api_key="your_key",              # API key (or use env var)
    max_tokens=4096,                 # Maximum response tokens
    temperature=0.7,                 # Creativity (0.0-2.0)
    top_p=1.0,                       # Nucleus sampling (0.0-1.0)
    top_k=0,                         # Top-k sampling (0 = disabled)
    parallel_tool_calls=False,       # Parallel tool execution
    max_retries=3                    # Retry attempts on failure
)
```

## Pricing

Requesty uses a pay-per-use model with different pricing for each model. Pricing is typically competitive with direct provider access and may offer cost savings through intelligent routing.

Check current pricing in your Requesty dashboard or documentation.

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: Requesty API key not provided
   ```
   - Ensure `REQUESTY_API_KEY` is set correctly
   - Check that your API key is valid in Requesty dashboard
   - Make sure you have credits in your account

2. **Model Not Found**
   ```
   Error: Model 'invalid/model' not found
   ```
   - Verify the model name uses correct `provider/model` format
   - Check if the model is available on Requesty
   - Some models may have regional restrictions

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   - Requesty has rate limits per model and account
   - The client includes automatic retry logic
   - Consider using smaller models for high-frequency requests

4. **Tool Calling Issues**
   ```
   Error: Tool calling not supported
   ```
   - Not all models support tool calling
   - Check `client.supports_tool_calling(model_params)` first
   - Use models like GPT-4 or Claude 3.5 Sonnet for tool calling

5. **Header Configuration**
   ```
   Warning: Optional headers not configured
   ```
   - This is not an error, just a notice
   - Set `REQUESTY_SITE_URL` and `REQUESTY_SITE_NAME` for better tracking
   - Headers are optional but recommended for analytics

### Debug Mode

Enable debug logging to see detailed request/response information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Connection

Test your connection with a simple script:

```python
import os
import openai

client = openai.OpenAI(
    api_key=os.getenv("REQUESTY_API_KEY"),
    base_url="https://router.requesty.ai/v1"
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("✅ Connection successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Getting Help

- Requesty Documentation: Check their official docs
- Requesty Support: Contact through their platform
- Model-specific documentation from original providers
- Community forums and Discord servers

## Security Notes

- Never commit API keys to version control
- Use environment variables for API keys
- Monitor your usage on the Requesty dashboard
- Set up billing alerts to avoid unexpected charges
- Rotate API keys regularly for security
- Be mindful of data privacy when using third-party routing services

## Comparison with Other Providers

| Feature | Requesty | OpenRouter | Direct Provider |
|---------|----------|------------|----------------|
| Multiple Models | ✅ | ✅ | ❌ |
| Single API Key | ✅ | ✅ | ❌ |
| Cost Optimization | ✅ | ✅ | ❌ |
| Request Analytics | ✅ | ✅ | Limited |
| Intelligent Routing | ✅ | Limited | ❌ |
| Setup Complexity | Low | Low | High |

Choose Requesty when you want:
- Access to multiple models with one API key
- Intelligent request routing
- Enhanced analytics and tracking
- Cost optimization across providers
- Simplified model switching and testing