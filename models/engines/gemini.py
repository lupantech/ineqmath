# Ref: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/gemini.py
# Ref: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("If you'd like to use Gemini models, please install the google-generativeai package by running `pip install google-generativeai`, and add 'GOOGLE_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import base64
import json
from typing import List, Union
from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

class ChatGemini(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gemini-pro",
        use_cache: bool=False,
        system_prompt=SYSTEM_PROMPT,
        is_multimodal: bool=False,
    ):
        self.use_cache = use_cache
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_gemini_{model_string}.db")
            super().__init__(cache_path=cache_path)
            
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable if you'd like to use Gemini models.")
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Gemini Pro Vision.")
            
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

        client = genai.GenerativeModel(self.model_string,
                                     system_instruction=sys_prompt_arg)
        messages = [{'role': 'user', 'parts': [prompt]}]
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_count=1
        )
        
        response = client.generate_content(messages, generation_config=generation_config)
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = ""

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)
                image_media_type = f"image/{image_type}"
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "mime_type": image_media_type,
                    "data": base64_image
                })
            elif isinstance(item, str):
                formatted_content.append(item)
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

        client = genai.GenerativeModel(self.model_string,
                                     system_instruction=sys_prompt_arg)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_count=1
        )

        response = client.generate_content(formatted_content, generation_config=generation_config)
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = ""

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text