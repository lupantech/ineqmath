# Reference: https://github.com/zou-group/textgrad/tree/main/textgrad/engine

try:
    from together import Together
except ImportError:
    raise ImportError("If you'd like to use Together models, please install the together package by running `pip install together`, and add 'TOGETHER_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import json
import base64
from .base import EngineLM, CachedEngine

class ChatTogether(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="meta-llama/Llama-3-70b-chat-hf",
        use_cache: bool=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.model_string = model_string

        # Check if model supports multimodal inputs
        self.is_multimodal = is_multimodal or any(x in model_string.lower() for x in [
            "llama-4",
            "qwen2-vl",
            "qwen-vl",
            "vl",
            "visual"
        ])

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_together_{model_string}.db")
            super().__init__(cache_path=cache_path)

        if os.getenv("TOGETHER_API_KEY") is None:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable if you'd like to use OpenAI models.")
        
        self.client = Together(
            api_key=os.getenv("TOGETHER_API_KEY"),
        )

    def _format_content(self, content):
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def generate(
        self, content, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):  
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)
            
            elif isinstance(content, list):
                if not self.is_multimodal:
                    raise NotImplementedError("Multimodal generation is only supported for multimodal models.")
                return self._generate_multimodal(content, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            if max_tokens != 10000:
                cache_key += f"_max_tokens_{max_tokens}"
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # # Adjust max_tokens to ensure total tokens don't exceed Together's limit
        # MAX_TOTAL_TOKENS = 8000
        # max_tokens = min(max_tokens, MAX_TOTAL_TOKENS - 1000)

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def _generate_multimodal(
        self, content, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
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

        # # Adjust max_tokens to ensure total tokens don't exceed Together's limit
        # MAX_TOTAL_TOKENS = 8000
        # max_tokens = min(max_tokens, MAX_TOTAL_TOKENS - 1000)

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)