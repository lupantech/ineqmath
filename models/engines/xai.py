# Ref: https://docs.x.ai/docs/guides/chat
# Ref: https://docs.x.ai/docs/guides/reasoning
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use Groq models, please install the openai package by running `pip install openai`, and add 'XAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes
from .openai import ChatOpenAI

def validate_reasoning_model(model_string: str):
    # Ref: https://docs.x.ai/docs/guides/reasoning
    return any(x in model_string for x in ["grok-3-mini"])


class ChatGrok(ChatOpenAI):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="grok-3-latest",
        use_cache: bool=False,
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        **kwargs):
        """
        :param model_string: The Groq model to use
        :param use_cache: Whether to use caching
        :param system_prompt: System prompt to use
        :param is_multimodal: Whether to enable multimodal capabilities
        """
        self.use_cache = use_cache
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        self.is_reasoning_model = validate_reasoning_model(model_string)

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_groq_{model_string}.db")
            super().__init__(model_string=model_string, use_cache=use_cache, system_prompt=system_prompt, is_multimodal=is_multimodal, cache_path=cache_path)
        
        if os.getenv("XAI_API_KEY") is None:
            raise ValueError("Please set the XAI_API_KEY environment variable if you'd like to use Groq models.")
        
        self.client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        assert isinstance(self.system_prompt, str)


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    # @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is not supported for Groq models.")
            
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            if max_tokens != 10000:
                cache_key += f"_max_tokens_{max_tokens}"
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Chat with reasoning model
        if self.is_reasoning_model:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_string,
                reasoning_effort="medium",
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        # Chat with non-reasoning model
        else:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt}
            ],
            model=self.model_string,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        response_text = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        # Ref: https://docs.x.ai/docs/guides/image-understanding#image-understanding
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}",
                    },
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            if max_tokens != 10000:
                cache_key += f"_max_tokens_{max_tokens}"
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Chat with reasoning model
        if self.is_reasoning_model:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content}
                ],
                model=self.model_string,
                reasoning_effort="medium",
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Chat with non-reasoning model
        else:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content}
                ],
                model=self.model_string,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        response_text = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text