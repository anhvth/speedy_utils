"""
LLM-as-a-Judge implementation with template support and SFT export utilities.

This module provides a base class for creating LLM judges with structured
prompts, variable substitution, and export capabilities for fine-tuning.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from ..chat_format import get_conversation_one_turn
from .llm_task import LLM
from .signature import Signature


class LLMJudgeBase(LLM):
    """Base class for LLM judges with template support and SFT export."""
    
    def __init__(
        self,
        signature: Type[Signature],
        **kwargs
    ):
        """
        Initialize LLMJudgeBase.
        
        Args:
            system_prompt_template: System prompt template with {variable} placeholders
            signature: Optional Signature class for structured I/O
            **kwargs: Additional arguments passed to LLMTask
        """
        self.signature = signature
        self.sft_data: List[Dict[str, Any]] = []  # Store SFT training examples
        
        # Set instruction from signature if available
        kwargs.setdefault('instruction', signature.get_instruction())
        kwargs.setdefault('output_model', signature.get_output_model())
        
        super().__init__(**kwargs)
    
    # def format_system_prompt(self, variables: Dict[str, Any]) -> str:
    #     """Format system prompt template with provided variables."""
    #     try:
    #         return self.system_prompt_template.format(**variables)
    #     except KeyError as e:
    #         missing_var = str(e).strip("'")
    #         raise ValueError(f"Missing required variable '{missing_var}' for system prompt template")
    
    def judge(
        self,
        input_data: Union[str, Dict[str, Any], BaseModel],
        variables: Optional[Dict[str, Any]] = None,
        **runtime_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute judgment with variable substitution in system prompt.
        
        Args:
            input_data: Input data for the judge
            variables: Variables to substitute in system prompt template
            **runtime_kwargs: Additional runtime arguments
            
        Returns:
            List of judgment results
        """
        variables = variables or {}
        
        # Format system prompt with variables
        # formatted_prompt = self.format_system_prompt(variables)
        
        # Temporarily override instruction
        original_instruction = self.instruction
        # self.instruction = formatted_prompt
        
        try:
            # Handle different input types
            if isinstance(input_data, dict):
                processed_input = json.dumps(input_data)
            else:
                processed_input = input_data
            results = self(processed_input, **runtime_kwargs)
            
            # Store for SFT if needed
            self._store_sft_example(input_data, results, variables, self.instruction) # type: ignore
            
            return results
        finally:
            # Restore original instruction
            self.instruction = original_instruction
    
    def _store_sft_example(
        self,
        input_data: Union[str, Dict[str, Any], BaseModel],
        results: List[Dict[str, Any]],
        variables: Dict[str, Any],
        formatted_prompt: str
    ) -> None:
        """Store example for SFT export."""
        for result in results:
            # Create input text
            if isinstance(input_data, str):
                input_text = input_data
            elif isinstance(input_data, BaseModel):
                input_text = input_data.model_dump_json()
            elif isinstance(input_data, dict):
                input_text = json.dumps(input_data)
            else:
                input_text = str(input_data)
            
            # Extract output
            output_text = result['parsed']
            if isinstance(output_text, BaseModel):
                output_text = output_text.model_dump_json()
            elif not isinstance(output_text, str):
                output_text = str(output_text)
            
            # Create conversation format
            messages = get_conversation_one_turn(
                formatted_prompt,
                input_text,
                output_text
            )
            
            sft_example = {
                'messages': messages,
                'variables': variables,
                'input_data': input_data,
                'output': result['parsed']
            }
            
            self.sft_data.append(sft_example)
    
    def export_sft_data(self, format: str = 'messages') -> List[Dict[str, Any]]:
        """
        Export stored examples in SFT format.
        
        Args:
            format: Export format ('messages', 'full', or 'sharegpt')
            
        Returns:
            List of SFT training examples
        """
        if format == 'messages':
            return [{'messages': example['messages']} for example in self.sft_data]
        elif format == 'full':
            return self.sft_data
        elif format == 'sharegpt':
            # Convert to ShareGPT format
            sharegpt_data = []
            for example in self.sft_data:
                conversations = []
                for msg in example['messages']:
                    conversations.append({
                        'from': 'human' if msg['role'] == 'user' else 'gpt' if msg['role'] == 'assistant' else 'system',
                        'value': msg['content']
                    })
                sharegpt_data.append({'conversations': conversations})
            return sharegpt_data
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from 'messages', 'full', or 'sharegpt'")
    
    def save_sft_data(self, filepath: str, format: str = 'messages') -> None:
        """Save SFT data to file."""
        sft_data = self.export_sft_data(format)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, indent=2, ensure_ascii=False)
    
    def clear_sft_data(self) -> None:
        """Clear stored SFT examples."""
        self.sft_data.clear()

