# Current Tokenizer-Related Implementation Summary

## Public Surface

There is no public `TokenizationMixin` in the current source tree, and the
current `LLM` class does not expose generic `encode()` or `decode()` methods.

The current public LLM API is centered on:

- `LLM.chat_completion()`
- `LLM.generate()`
- `LLM.pydantic_parse()`
- `Qwen3LLM.chat_completion()`
- `Qwen3LLM.complete_until()`
- `Qwen3LLM.complete_reasoning()`
- `Qwen3LLM.complete_content()`

## `LLM.generate()`

`LLM.generate()` is the raw prompt-continuation method.

Current behavior:

- input: `prompt: str`
- output: `CompletionChoice`-like object
- public API constraint: `n=1`

It is not a token-id generation API in this branch.

## Internal Tokenizer Use In `Qwen3LLM`

Tokenizer loading currently lives in `src/llm_utils/lm/llm_qwen3.py`.

The important internal flow is:

1. `_get_tokenizer()` lazily loads the Qwen tokenizer.
2. `_build_completion_prompt()` calls `tokenizer.apply_chat_template(...)` when
   the tokenizer is available.
3. if tokenizer loading fails, the code falls back to text rendering via the
   chat-format helpers.

This means tokenizer support still matters to `Qwen3LLM`, but it is an internal
prompt-rendering detail rather than a generic tokenization API for all models.

## Qwen3 Prefix Helpers

The current staged Qwen3 helpers are:

- `chat_completion(...)` -> returns a `ChatCompletionMessage`
- `complete_until(...)` -> returns a continuation-state object
- `complete_reasoning(...)` -> returns a reasoning prefix state
- `complete_content(...)` -> returns a `ChatCompletionMessage`

These helpers are the current supported way to work with Qwen3-style reasoning
prefixes.

## Documentation Implication

Docs in this repository should **not** promise any of the following unless the
runtime code is reintroduced:

- `lm.encode(...)`
- `lm.decode(...)`
- token-id input to `LLM.generate(...)`
- a public tokenization mixin
- examples built around `/tokenize` and `/detokenize` endpoints

## If Tokenization Is Reintroduced Later

Any reintroduction should come with all of the following at once:

- real source files implementing the public methods
- export updates in `llm_utils`
- tests covering public behavior
- examples that exercise the real API
- updated docs replacing this historical note
