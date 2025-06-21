# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenRouter API client wrapper with tool integration."""

import os
import json
import random
import time
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from typing import override

from ..tools.base import Tool, ToolCall, ToolResult
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class OpenRouterClient(BaseLLMClient):
    """OpenRouter client wrapper with tool schema generation using OpenAI-compatible endpoints."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        if self.api_key == "":
            self.api_key: str = os.getenv("OPENROUTER_API_KEY", "")

        if self.api_key == "":
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY in environment variables or config file.")

        # OpenRouter uses OpenAI-compatible endpoints
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.message_history: list[dict] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True) -> LLMResponse:
        """Send chat messages to OpenRouter with optional tool support."""
        openai_messages: list[dict] = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_input_schema()
                }
            } for tool in tools]

        if reuse_history:
            self.message_history = self.message_history + openai_messages
        else:
            self.message_history = openai_messages

        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                # Prepare request parameters
                request_params = {
                    "model": model_parameters.model,
                    "messages": self.message_history,
                    "temperature": model_parameters.temperature,
                    "top_p": model_parameters.top_p,
                    "max_tokens": model_parameters.max_tokens,
                }
                
                if tool_schemas:
                    request_params["tools"] = tool_schemas
                    request_params["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**request_params)
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                # Randomly sleep for 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from OpenRouter after max retries: {error_message}")

        content = ""
        tool_calls: list[ToolCall] = []
        
        # Extract content and tool calls from response
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            if message.content:
                content = message.content
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(ToolCall(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                        id=tool_call.id
                    ))

        # Add assistant message to history
        assistant_message = {"role": "assistant"}
        if content:
            assistant_message["content"] = content
        if tool_calls:
            assistant_message["tool_calls"] = [{
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments)
                }
            } for tc in tool_calls]
        
        self.message_history.append(assistant_message)

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cache_read_input_tokens=0,  # OpenRouter doesn't provide cache info
                reasoning_tokens=0  # OpenRouter doesn't provide reasoning tokens
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason if response.choices else None,
            tool_calls=tool_calls if len(tool_calls) > 0 else None
        )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openrouter",
                model=model_parameters.model,
                tools=tools
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""
        # Most modern models on OpenRouter support tool calling
        # This is a conservative list - you can expand it based on OpenRouter's documentation
        tool_capable_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "claude-3", "claude-3.5", "claude-sonnet", "claude-haiku", "claude-opus",
            "gemini", "llama-3", "qwen", "deepseek", "mistral"
        ]
        return any(model in model_parameters.model.lower() for model in tool_capable_models)

    def parse_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Parse the messages to OpenAI format."""
        openai_messages: list[dict] = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                # Tool calls are handled in the assistant message, skip individual tool call messages
                continue
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> dict:
        """Parse the tool call result from the LLM response."""
        result: str = ""
        if tool_call_result.result:
            result = result + tool_call_result.result + "\n"
        if tool_call_result.error:
            result += tool_call_result.error
        result = result.strip()

        return {
            "role": "tool",
            "tool_call_id": tool_call_result.call_id,
            "content": result
        }