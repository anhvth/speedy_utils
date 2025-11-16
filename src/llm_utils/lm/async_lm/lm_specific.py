from typing import List

from .async_lm import AsyncLM


KNOWN_CONFIG = {
    # Qwen3 family (see model card "Best Practices" section)
    "qwen3-think": {
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        },
    },
    "qwen3-no-think": {
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        },
    },
    # DeepSeek V3 (model card: temperature=0.3)
    "deepseek-v3": {
        "sampling_params": {
            "temperature": 0.3,
        },
    },
    # DeepSeek R1 (model card: temperature=0.6, top_p=0.95)
    "deepseek-r1": {
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
        },
    },
    # Mistral Small 3.2-24B Instruct (model card: temperature=0.15)
    "mistral-small-3.2-24b-instruct-2506": {
        "sampling_params": {
            "temperature": 0.15,
        },
    },
    # Magistral Small 2506 (model card: temperature=0.7, top_p=0.95)
    "magistral-small-2506": {
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.95,
        },
    },
    # Phi-4 Reasoning (model card: temperature=0.8, top_k=50, top_p=0.95)
    "phi-4-reasoning": {
        "sampling_params": {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
        },
    },
    # GLM-Z1-32B-0414 (model card: temperature=0.6, top_p=0.95, top_k=40, max_new_tokens=30000)
    "glm-z1-32b-0414": {
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "max_new_tokens": 30000,
        },
    },
    # Llama-4-Scout-17B-16E-Instruct (generation_config.json: temperature=0.6, top_p=0.9)
    "llama-4-scout-17b-16e-instruct": {
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.9,
        },
    },
    # Gemma-3-27b-it (alleged: temperature=1.0, top_k=64, top_p=0.96)
    "gemma-3-27b-it": {
        "sampling_params": {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.96,
        },
    },
    # Add more as needed...
}

KNOWN_KEYS: list[str] = list(KNOWN_CONFIG.keys())


class AsyncLMQwenThink(AsyncLM):
    def __init__(
        self,
        model: str = "Qwen32B",
        temperature: float = KNOWN_CONFIG["qwen3-think"]["sampling_params"][
            "temperature"
        ],
        top_p: float = KNOWN_CONFIG["qwen3-think"]["sampling_params"]["top_p"],
        top_k: int = KNOWN_CONFIG["qwen3-think"]["sampling_params"]["top_k"],
        presence_penalty: float = KNOWN_CONFIG["qwen3-think"]["sampling_params"][
            "presence_penalty"
        ],
        **other_kwargs,
    ):
        super().__init__(
            model="qwen3-think",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            **other_kwargs,
            think=True,
        )


class AsyncLMQwenNoThink(AsyncLM):
    def __init__(
        self,
        model: str = "Qwen32B",
        temperature: float = KNOWN_CONFIG["qwen3-no-think"]["sampling_params"][
            "temperature"
        ],
        top_p: float = KNOWN_CONFIG["qwen3-no-think"]["sampling_params"]["top_p"],
        top_k: int = KNOWN_CONFIG["qwen3-no-think"]["sampling_params"]["top_k"],
        presence_penalty: float = KNOWN_CONFIG["qwen3-no-think"]["sampling_params"][
            "presence_penalty"
        ],
        **other_kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            **other_kwargs,
            think=False,
        )
