#!/usr/bin/env python3
"""Test script for OpenRouter client integration."""

import os
from trae_agent.utils.llm_client import LLMClient
from trae_agent.utils.config import ModelParameters
from trae_agent.utils.llm_basics import LLMMessage

def test_openrouter_client():
    """Test the OpenRouter client with a simple chat."""
    
    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        print("You can get an API key from: https://openrouter.ai/keys")
        return
    
    # Configure model parameters for OpenRouter
    model_params = ModelParameters(
        model="anthropic/claude-3.5-sonnet",  # Example model available on OpenRouter
        api_key=api_key,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3
    )
    
    # Create OpenRouter client
    client = LLMClient("openrouter", model_params)
    
    # Test basic chat
    messages = [
        LLMMessage(role="user", content="Hello! Can you tell me a short joke?")
    ]
    
    print("Testing OpenRouter client...")
    print(f"Model: {model_params.model}")
    print("\nSending message: Hello! Can you tell me a short joke?")
    
    try:
        response = client.chat(messages, model_params)
        print(f"\nResponse: {response.content}")
        print(f"Model used: {response.model}")
        if response.usage:
            print(f"Tokens - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
        print("\n✅ OpenRouter client test successful!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_tool_calling():
    """Test tool calling capability."""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable for tool calling test")
        return
    
    model_params = ModelParameters(
        model="anthropic/claude-3.5-sonnet",
        api_key=api_key,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3
    )
    
    client = LLMClient("openrouter", model_params)
    
    # Check if model supports tool calling
    supports_tools = client.supports_tool_calling(model_params)
    print(f"\nTool calling support for {model_params.model}: {supports_tools}")

if __name__ == "__main__":
    print("OpenRouter Client Test")
    print("=" * 50)
    
    test_openrouter_client()
    test_tool_calling()
    
    print("\nTo use OpenRouter in your configuration:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Update trae_config.json with 'default_provider': 'openrouter'")
    print("3. Choose from available models at: https://openrouter.ai/models")
    print("\nExample models:")
    print("- anthropic/claude-3.5-sonnet")
    print("- openai/gpt-4o")
    print("- meta-llama/llama-3.1-405b-instruct")
    print("- google/gemini-pro-1.5")