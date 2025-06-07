# Ref: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/vllm.py
# TODO: To update the codebase to adapt to OctoTools

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm`."
    )

import os
import platformdirs
from .base import EngineLM, CachedEngine

import torch._dynamo
torch._dynamo.config.suppress_errors = True

class ChatVLLM(EngineLM, CachedEngine):
    # Default system prompt for VLLM models
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        use_cache: bool=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        **llm_config,
    ):
        self.use_cache = use_cache
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_vllm_{model_string}.db")
            super().__init__(cache_path=cache_path)
        
        # Add GPU memory utilization if not provided
        if 'gpu_memory_utilization' not in llm_config:
            llm_config['gpu_memory_utilization'] = 0.1 # Reduced to 60% to avoid Out of Memory errors
        try:
            self.client = LLM(self.model_string, **llm_config)

        except RuntimeError as e:
            if "Failed to find C compiler" in str(e):
                raise RuntimeError(
                    "VLLM requires a C compiler to be installed. Please install gcc/g++ and try again. "
                    "On Ubuntu/Debian: `sudo apt-get install build-essential`\n"
                    "On CentOS/RHEL: `sudo yum groupinstall 'Development Tools'`\n"
                    "On macOS: Install Xcode Command Line Tools"
                ) from e
            raise

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        # The chat template ignores the system prompt;
        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]

        conversation += [{"role": "user", "content": prompt}]
        chat_str = str(conversation)

        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1
        )

        response = self.client.generate(chat_str, sampling_params)
        response = response[0].outputs[0].text

        self._save_cache(sys_prompt_arg + prompt, response)

        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
