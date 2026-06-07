from __future__ import annotations

import backoff
from google import genai
from google.genai import types

from .base import LLM


GEMINI_PROVIDER = "gemini"


class OnlineLLMs(LLM):
    def __init__(self, name, api_key=None, model_version=None):
        self.name = name.lower()
        self.client = None
        self.model_version = model_version

        if self.name == GEMINI_PROVIDER:
            # Prefer explicit API key if provided; otherwise rely on GEMINI_API_KEY env var.
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = genai.Client()

    def set_model(self, model):
        # Backwards-compat shim; accept an already-created client instance
        # (for example, if you want to share a single Client across objects).
        self.client = model

    def parse_message(self, messages):
        mapping = {
            "user": "user",
            "assistant": "model"
        }
        return [
            {"role": mapping[mess["role"]], "parts": mess["content"]}
            for mess in messages
        ]
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
        if self.name != GEMINI_PROVIDER:
            raise ValueError(f"Unknown model name: {self.name}")
        if not self.client:
            raise ValueError("Gemini client is not initialized.")
        try:
            messages = self.parse_message(messages)
            response = self.client.models.generate_content(
                model=self.model_version,
                contents=[
                    {"role": "user", "parts": system_prompt},
                    {
                        "role": "model",
                        "parts": "I understand. I will strictly follow your instruction!",
                    },
                    *messages,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e   

    def generate_content(self, prompt):
        if not self.client:
            raise ValueError(
                "Client is not set. Please initialize Gemini client or call set_model()."
            )

        if self.name != GEMINI_PROVIDER:
            raise ValueError(f"Unknown model name: {self.name}")

        response = self.client.models.generate_content(
            model=self.model_version,
            contents=prompt,
        )
        # New SDK exposes a helper that returns the concatenated text.
        content = response.text
        if not isinstance(content, str):
            content = str(content)
        return content
