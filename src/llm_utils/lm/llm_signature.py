"""
LLM-as-a-Judge implementation with template support and SFT export utilities.

This module provides a base class for creating LLM judges with structured
prompts, variable substitution, and export capabilities for fine-tuning.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..chat_format import get_conversation_one_turn
from .llm import LLM
from .signature import Signature


class LLMSignature(LLM):
    """Base class for LLM judges with template support and SFT export."""

    def __init__(self, signature: type[Signature], **kwargs):
        """
        Initialize LLMJudgeBase.

        Args:
            system_prompt_template: System prompt template with {variable} placeholders
            signature: Optional Signature class for structured I/O
            **kwargs: Additional arguments passed to LLMTask
        """
        self.signature = signature
        self.sft_data: list[dict[str, Any]] = []  # Store SFT training examples

        # Set instruction from signature if available
        kwargs.setdefault("instruction", signature.get_instruction())
        kwargs.setdefault("output_model", signature.get_output_model())

        super().__init__(**kwargs)
