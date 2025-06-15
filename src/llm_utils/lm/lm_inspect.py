from openai import OpenAI
from .lm import LM, get_tokenizer
from typing import *
from loguru import logger
import torch
import numpy as np
import re


class LMInspect(LM):
    """
    A class to inspect the LM's internal state.
    This is useful for debugging and understanding the LM's behavior.
    """

    def get_token_log_probs(self, prompt: str, topk: int = 10) -> list[dict[str, Any]]:
        """Get token log probabilities for a given prompt."""
        response = self.client.completions.create(
            model=self.model,  # type: ignore
            prompt=prompt,
            max_tokens=1,
            logprobs=topk,
            temperature=0.0,
            extra_body={"prompt_logprobs": topk},
        )
        token_logprob_dicts = response.choices[0].prompt_logprobs  # type: ignore
        start_id = self.tokenizer.encode("<|im_start|>")[0]
        token_logprob_dicts[0] = {
            str(start_id): {
                "logprob": -1,
                "rank": 1,
                "decoded_token": "<|im_start|>",
            }
        }
        return token_logprob_dicts

    def get_next_token_prob_tensor(self, prompt: str, topk: int = 10) -> torch.Tensor:
        """Get next token probabilities as tensor of shape (L-1, NUM_LABELS)."""
        token_logprob_dicts = self.get_token_log_probs(prompt, topk)
        vocab_size = len(self.tokenizer.get_vocab())

        # Shape: (L-1, NUM_LABELS) where L-1 is sequence length minus 1
        seq_len = len(token_logprob_dicts) - 1
        if seq_len <= 0:
            return torch.zeros((0, vocab_size), dtype=torch.float32)

        prob_tensor = torch.zeros((seq_len, vocab_size), dtype=torch.float32)

        # Fill in probabilities for next token predictions
        for i in range(seq_len):
            next_token_logprobs = token_logprob_dicts[i + 1]
            for token_id_str, token_data in next_token_logprobs.items():
                token_id = int(token_id_str)
                if 0 <= token_id < vocab_size:
                    prob = np.exp(token_data["logprob"])
                    prob_tensor[i, token_id] = prob

        return prob_tensor

    def inspect_word_probs(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        tokenizer: Optional[Any] = None,
        do_print: bool = True,
        add_think: bool = True,
    ) -> tuple[list[dict[str, Any]], Any, str]:
        """Inspect word probabilities in a language model response."""
        if messages is None:
            messages = self.last_messages(add_think=add_think)
            if messages is None:
                raise ValueError("No messages provided and no last messages available.")
        if tokenizer is None:
            tokenizer = get_tokenizer(self.model)

        def compute_word_log_probs(
            tokenizer: Any,
            lm_client: Any,
        ) -> tuple[list[dict[str, Any]], Any]:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            token_logprob_dicts = self.get_token_log_probs(prompt)
            tokens: list[dict[str, Any]] = [
                {"id": int(tid), **tdata}
                for td in token_logprob_dicts
                for tid, tdata in td.items()
            ]
            tokenized = tokenizer.tokenize(prompt)
            if len(tokenized) != len(tokens):
                raise ValueError(
                    f"Token count mismatch: {len(tokenized)} vs {len(tokens)}"
                )
            for idx, tok in enumerate(tokens):
                if tokenized[idx] != tok["decoded_token"]:
                    raise AssertionError(
                        f"Token mismatch at {idx}: "
                        f"{tokenized[idx]} != {tok['decoded_token']}"
                    )
            split_prompt = prompt.replace("\n", " <NL> ")
            words = split_prompt.split()
            word_log_probs: list[dict[str, Any]] = []
            token_idx = 0
            for word in words:
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
            return word_log_probs, token_logprob_dicts

        def render_by_logprob(word_log_probs: list[dict[str, Any]]) -> str:
            """Return an ANSI-colored string for word probabilities (red → green)."""
            if not word_log_probs:
                return ""
            probs = [entry["probability"] for entry in word_log_probs]
            min_p, max_p = min(probs), max(probs)
            parts: list[str] = []
            for entry in word_log_probs:
                word = entry["word"]
                if word == "\n":
                    parts.append("\n")
                    continue
                p = entry["probability"]
                norm = (p - min_p) / (max_p - min_p or 1.0)
                r = int(255 * (1 - norm))
                g = int(255 * norm)
                b = 0
                colored = f"\x1b[38;2;{r};{g};{b}m{word}\x1b[0m"
                parts.append(colored + " ")
            return "".join(parts).rstrip()

        word_probs, token_logprob_dicts = compute_word_log_probs(tokenizer, self)
        if do_print:
            print(render_by_logprob(word_probs))
        return word_probs, token_logprob_dicts, render_by_logprob(word_probs)

    def last_messages(self, add_think: bool = True) -> Optional[list[dict[str, str]]]:
        last_conv = self.last_log
        messages = last_conv[1] if len(last_conv) > 1 else None
        last_msg = last_conv[2]
        if not isinstance(last_msg, dict):
            last_conv[2] = last_conv[2].model_dump()  # type: ignore
        msg = last_conv[2]
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        message = msg["choices"][0]["message"]
        reasoning = message.get("reasoning_content")
        answer = message.get("content")
        if reasoning and add_think:
            final_answer = f"<think>\n{reasoning}\n</think>\n{answer}"
        else:
            final_answer = f"<think>\n\n</think>\n{answer}"
        assistant = {"role": "assistant", "content": final_answer}
        messages = messages + [assistant]  # type: ignore
        return messages if messages else None
