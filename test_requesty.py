#!/usr/bin/env python3
"""Test script for Requesty client integration."""

import os
from trae_agent.utils.llm_client import LLMClient
from trae_agent.utils.config import ModelParameters
from trae_agent.utils.llm_basics import LLMMessage

def test_requesty_client():
    """Test the Requesty client with a simple chat."""
    
    # Check if API key is available
    api_key = os.getenv("REQUESTY_API_KEY")
    if not api_key:
        print("Please set REQUESTY_API_KEY environment variable")
        print("You can get an API key from: https://requesty.ai/")
        return
    
    # Configure model parameters for Requesty
    model_params = ModelParameters(
        model="openai/gpt-4o",  # Example model available on Requesty
        api_key=api_key,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3
    )
    
    # Create Requesty client
    client = LLMClient("requesty", model_params)
    
    # Test basic chat
    messages = [
        LLMMessage(role="user", content="Hello! Can you tell me a short joke?")
    ]
    
    print("Testing Requesty client...")
    print(f"Model: {model_params.model}")
    print("\nSending message: Hello! Can you tell me a short joke?")
    
    # Check for optional headers
    site_url = os.getenv("REQUESTY_SITE_URL")
    site_name = os.getenv("REQUESTY_SITE_NAME")
    if site_url or site_name:
        print("\nOptional headers configured:")
        if site_url:
            print(f"  Site URL: {site_url}")
        if site_name:
            print(f"  Site Name: {site_name}")
    
    try:
        response = client.chat(messages, model_params)
        print(f"\nResponse: {response.content}")
        print(f"Model used: {response.model}")
        if response.usage:
            print(f"Tokens - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
        print("\n✅ Requesty client test successful!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_tool_calling():
    """Test tool calling capability."""
    
    api_key = os.getenv("REQUESTY_API_KEY")
    if not api_key:
        print("Please set REQUESTY_API_KEY environment variable for tool calling test")
        return
    
    model_params = ModelParameters(
        model="openai/gpt-4o",
        api_key=api_key,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3
    )
    
    client = LLMClient("requesty", model_params)
    
    # Check if model supports tool calling
    supports_tools = client.supports_tool_calling(model_params)
    print(f"\nTool calling support for {model_params.model}: {supports_tools}")

def test_with_headers():
    """Test Requesty client with optional headers."""
    
    print("\nTesting with optional headers...")
    print("Set these environment variables for enhanced functionality:")
    print("  export REQUESTY_SITE_URL='https://yoursite.com'")
    print("  export REQUESTY_SITE_NAME='Your App Name'")
    
    site_url = os.getenv("REQUESTY_SITE_URL")
    site_name = os.getenv("REQUESTY_SITE_NAME")
    
    if site_url or site_name:
        print("\n✅ Optional headers are configured:")
        if site_url:
            print(f"  HTTP-Referer: {site_url}")
        if site_name:
            print(f"  X-Title: {site_name}")
    else:
        print("\n⚠️  Optional headers not configured (this is fine for testing)")

if __name__ == "__main__":
    print("Requesty Client Test")
    print("=" * 50)
    
    test_requesty_client()
    test_tool_calling()
    test_with_headers()
    
    print("\nTo use Requesty in your configuration:")
    print("1. Set REQUESTY_API_KEY environment variable")
    print("2. Optionally set REQUESTY_SITE_URL and REQUESTY_SITE_NAME")
    print("3. Update trae_config.json with 'default_provider': 'requesty'")
    print("4. Choose from available models (uses provider/model format)")
    print("\nExample models:")
    print("- openai/gpt-4o")
    print("- openai/gpt-4o-mini")
    print("- anthropic/claude-3.5-sonnet")
    print("- meta-llama/llama-3.1-405b-instruct")
    print("- google/gemini-pro-1.5")
    print("\nFor more information visit: https://requesty.ai/")