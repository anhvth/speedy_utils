from functools import lru_cache
from typing import (
    Any,
    Dict,
    Generic,
    List,
    TypeVar,
    Union,
)

# from openai.pagination import AsyncSyncPage
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel
from typing_extensions import TypedDict

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]

# --------------------------------------------------------------------------- #
# color helpers (unchanged)
# --------------------------------------------------------------------------- #


def _color(code: int, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def _red(t):
    return _color(31, t)


def _green(t):
    return _color(32, t)


def _blue(t):
    return _color(34, t)


def _yellow(t):
    return _color(33, t)


TParsed = TypeVar("TParsed", bound=BaseModel)


class ParsedOutput(TypedDict, Generic[TParsed]):
    messages: List
    completion: Any
    parsed: TParsed


# --------------------------------------------------------------------------- #
# Module-level utility functions (async versions)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=10)
def get_tokenizer(model_name: str) -> Any:
    """Get tokenizer for the given model."""
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


async def inspect_word_probs_async(lm, tokenizer, messages):
    """Async version of inspect_word_probs."""

    import numpy as np

    async def compute_word_log_probs(
        tokenizer: Any,
        lm_client: Any,
    ) -> tuple[List[Dict[str, Any]], Any]:
        # Build a prompt that preserves literal newlines
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Don't tokenize yet, we need raw text
            add_generation_prompt=False,  # No generation prompt needed
        )

        # Request token logprobs
        response = await lm_client.client.completions.create(
            model=lm_client.model,  # type: ignore
            prompt=prompt,
            max_tokens=1,
            logprobs=1,
            extra_body={"prompt_logprobs": 0},
        )
        token_logprob_dicts = response.choices[0].prompt_logprobs  # type: ignore

        # Override first token to known start marker
        start_id = tokenizer.encode("<|im_start|>")[0]
        token_logprob_dicts[0] = {
            str(start_id): {
                "logprob": -1,
                "rank": 1,
                "decoded_token": "<|im_start|>",
            }
        }

        # Flatten tokens
        tokens: List[Dict[str, Any]] = [
            {"id": int(tid), **tdata}
            for td in token_logprob_dicts
            for tid, tdata in td.items()
        ]

        # Validate tokenization
        tokenized = tokenizer.tokenize(prompt)
        if len(tokenized) != len(tokens):
            raise ValueError(f"Token count mismatch: {len(tokenized)} vs {len(tokens)}")
        for idx, tok in enumerate(tokens):
            if tokenized[idx] != tok["decoded_token"]:
                raise AssertionError(
                    f"Token mismatch at {idx}: "
                    f"{tokenized[idx]} != {tok['decoded_token']}"
                )

        # Split on newline sentinel
        split_prompt = prompt.replace("\n", " <NL> ")
        words = split_prompt.split()

        word_log_probs: List[Dict[str, Any]] = []
        token_idx = 0

        for word in words:
            # Map sentinel back to actual newline for encoding
            target = "\n" if word == "<NL>" else word
            sub_ids = tokenizer.encode(target, add_special_tokens=False)
            count = len(sub_ids)
            if count == 0:
                continue

            subs = tokens[token_idx : token_idx + count]
            avg_logprob = sum(s["logprob"] for s in subs) / count
            prob = float(np.exp(avg_logprob))
            word_log_probs.append({"word": target, "probability": prob})
            token_idx += count

        return word_log_probs, token_logprob_dicts  # type: ignore

    def render_by_logprob(word_log_probs: List[Dict[str, Any]]) -> str:
        """
        Return an ANSI-colored string for word probabilities (red → green).
        """
        if not word_log_probs:
            return ""

        probs = [entry["probability"] for entry in word_log_probs]
        min_p, max_p = min(probs), max(probs)
        parts: List[str] = []

        for entry in word_log_probs:
            word = entry["word"]
            # Preserve actual line breaks
            if word == "\n":
                parts.append("\n")
                continue

            p = entry["probability"]
            norm = (p - min_p) / (max_p - min_p or 1.0)
            r = int(255 * (1 - norm))  # red component (high when prob is low)
            g = int(255 * norm)  # green component (high when prob is high)
            b = 0  # no blue for red-green gradient
            colored = f"\x1b[38;2;{r};{g};{b}m{word}\x1b[0m"
            parts.append(colored + " ")

        return "".join(parts).rstrip()

    word_probs, token_logprob_dicts = await compute_word_log_probs(tokenizer, lm)
    return word_probs, token_logprob_dicts, render_by_logprob(word_probs)


__all__ = [
    "TModel",
    "Messages",
    "LegacyMsgs",
    "RawMsgs",
    "TParsed",
    "ParsedOutput",
    "get_tokenizer",
    "inspect_word_probs_async",
    "_color",
    "_red",
    "_green",
    "_blue",
    "_yellow",
]
# --------------------------------------------------------------------------- #]
