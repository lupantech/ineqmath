try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the openai package by running `pip install openai`, and add 'DEEPSEEK_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union
from .base import EngineLM, CachedEngine


class ChatDeepseek(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_string="deepseek-chat",
        use_cache: bool=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False):

        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        self.is_chat_model = any(x in model_string for x in ["deepseek-chat"])
        self.is_reasoning_model = any(x in model_string for x in ["deepseek-reasoner"])

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_deepseek_{model_string}.db")
            super().__init__(cache_path=cache_path)

        if os.getenv("DEEPSEEK_API_KEY") is None:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, list) and len(content) == 1:
            content = content[0]
        if isinstance(content, str):
            return self._generate_text(content, system_prompt=system_prompt, **kwargs)

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            if max_tokens != 10000:
                cache_key += f"_max_tokens_{max_tokens}"
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        if self.is_chat_model:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response = response.choices[0].message.content

        elif self.is_reasoning_model:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    