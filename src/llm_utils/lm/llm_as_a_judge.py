"""
LLM-as-a-Judge implementation with template support and SFT export utilities.

This module provides a base class for creating LLM judges with structured
prompts, variable substitution, and export capabilities for fine-tuning.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from ..chat_format import get_conversation_one_turn
from .llm_task import LLMTask
from .signature import Signature


class LLMJudgeBase(LLMTask):
    """Base class for LLM judges with template support and SFT export."""
    
    def __init__(
        self,
        system_prompt_template: str,
        signature: Optional[Type[Signature]] = None,
        **kwargs
    ):
        """
        Initialize LLMJudgeBase.
        
        Args:
            system_prompt_template: System prompt template with {variable} placeholders
            signature: Optional Signature class for structured I/O
            **kwargs: Additional arguments passed to LLMTask
        """
        self.system_prompt_template = system_prompt_template
        self.signature = signature
        self.sft_data: List[Dict[str, Any]] = []  # Store SFT training examples
        
        # Set instruction from signature if available
        if signature is not None:
            instruction = signature.get_instruction()
            kwargs.setdefault('instruction', instruction)
            kwargs.setdefault('output_model', signature.get_output_model())
        else:
            kwargs.setdefault('instruction', system_prompt_template)
        
        super().__init__(**kwargs)
    
    def format_system_prompt(self, variables: Dict[str, Any]) -> str:
        """Format system prompt template with provided variables."""
        try:
            return self.system_prompt_template.format(**variables)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable '{missing_var}' for system prompt template")
    
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
        formatted_prompt = self.format_system_prompt(variables)
        
        # Temporarily override instruction
        original_instruction = self.instruction
        self.instruction = formatted_prompt
        
        try:
            # Handle different input types
            if isinstance(input_data, dict):
                processed_input = json.dumps(input_data)
            else:
                processed_input = input_data
            results = self(processed_input, **runtime_kwargs)
            
            # Store for SFT if needed
            self._store_sft_example(input_data, results, variables, formatted_prompt)
            
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


class ChainOfThought:
    """DSPy-like ChainOfThought wrapper for signatures."""
    
    def __init__(self, signature: Type[Signature], **llm_kwargs):
        """
        Initialize ChainOfThought with a signature.
        
        Args:
            signature: Signature class defining input/output structure
            **llm_kwargs: Arguments passed to LLMJudgeBase
        """
        self.signature = signature
        
        # Create system prompt from signature
        system_prompt = signature.get_instruction()
        
        # Add reasoning instruction
        system_prompt += "\n\nThink step by step before providing your final answer."
        
        self.llm = LLMJudgeBase(
            system_prompt_template=system_prompt,
            signature=signature,
            **llm_kwargs
        )
    
    def __call__(self, **kwargs) -> Any:
        """Execute chain of thought reasoning."""
        # Format input using signature
        signature_instance = self.signature(**kwargs)
        input_text = signature_instance.format_input(**kwargs)
        
        results = self.llm.judge(input_text)
        
        # Return the parsed output
        if results:
            return results[0]['parsed']
        return None


# Example usage classes based on the raw code
class TranslationOutput(BaseModel):
    """Output schema for translation evaluation."""
    structure_score: int  # 0 = wrong, 1 = partially correct, 2 = correct
    translation_score: int  # 0 = not faithful, 1 = somewhat faithful, 2 = fully faithful
    term_score: int  # 0 = glossary not followed, 1 = partially followed, 2 = fully followed or no glossary provided


class TranslationEvaluatorJudge(LLMJudgeBase):
    """Translation evaluator judge based on the raw code example."""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a careful **translation evaluator**.

You are given five inputs:

* **Source Prompt** (the original text & any constraints)
* **AI Translation** (the machine translation to evaluate)
* **Human Reference** (a reference rendering; use only for guidance, not as ground truth)
* **System Message** (an automated hint about a possible structural error)
* **Glossaries** (optional terminology constraints; may be empty)

## Your tasks

1. **Check structure correctness**:
   - Use the System Message as a hint.
   - Assign a `structure_score`:
     * `0` = structure is clearly wrong or the error flagged is correct.
     * `1` = partially correct but flawed.
     * `2` = structure is correct; the system error is invalid.

2. **Check translation quality**:
   - Compare AI Translation with Source Prompt and Human Reference.
   - Assign a `translation_score`:
     * `0` = unfaithful (major omissions/additions/distortions/repetitions).
     * `1` = somewhat faithful (mostly correct but noticeable issues).
     * `2` = faithful (preserves meaning, scope, nuance; only minor style differences).

3. **Check glossary/terminology adherence**:
   - If no glossary is provided → `term_score = 2`.
   - If glossary exists but only partially followed → `term_score = 1`.
   - If glossary exists but not followed at all → `term_score = 0`.

## Output format (JSON only; no commentary)

Return exactly one JSON object with the three scores.
Do not output any explanations.

---

### Inputs

Source Prompt: {SOURCE_PROMPT}

AI Translation: {AI_TRANSLATION}

Human Reference: {HUMAN_REFERENCE}

System Message: {SYSTEM_MESSAGE}

Glossaries: {GLOSSARIES}
"""
        
        super().__init__(
            system_prompt_template=system_prompt,
            output_model=TranslationOutput,
            **kwargs
        )
    
    def evaluate_translation(
        self,
        source_prompt: str,
        ai_translation: str,
        human_reference: str,
        system_message: str,
        glossaries: str
    ) -> TranslationOutput:
        """
        Evaluate a translation with all required parameters.
        
        Returns:
            TranslationOutput with the three scores
        """
        variables = {
            'SOURCE_PROMPT': source_prompt,
            'AI_TRANSLATION': ai_translation,
            'HUMAN_REFERENCE': human_reference,
            'SYSTEM_MESSAGE': system_message,
            'GLOSSARIES': glossaries
        }
        
        input_data = {
            'source': source_prompt,
            'target': human_reference,
            'glossaries': glossaries,
            'translation': ai_translation
        }
        
        results = self.judge(json.dumps(input_data), variables=variables)
        return results[0]['parsed']


# Example usage and testing
if __name__ == "__main__":
    # Test the Signature system
    from .signature import Signature, InputField, OutputField
    
    # Example 1: Using Signature with ChainOfThought (like DSPy)
    class FactJudge(Signature):
        """Judge if the answer is factually correct based on the context."""
        
        context: str = InputField(desc="Context for the prediction")  # type: ignore
        question: str = InputField(desc="Question to be answered")  # type: ignore 
        answer: str = InputField(desc="Answer for the question")  # type: ignore
        factually_correct: bool = OutputField(desc="Is the answer factually correct based on the context?")  # type: ignore
    
    print("=== Testing Signature System ===")
    print("Instruction:")
    print(FactJudge.get_instruction())
    
    # Example 2: Using LLMJudgeBase directly
    judge_prompt = """You are a factual accuracy judge.
    
Given:
- Context: {context}
- Question: {question}  
- Answer: {answer}

Determine if the answer is factually correct based on the context.
Respond with true if correct, false if incorrect."""
    
    print("\n=== Testing LLMJudgeBase ===")
    print("System prompt template:")
    print(judge_prompt)
    
    # Example 3: Translation evaluator from raw code
    print("\n=== Translation Evaluator Example ===")
    evaluator = TranslationEvaluatorJudge()
    print("Translation evaluator initialized with structured output schema.")
    print("Output schema:", TranslationOutput.model_json_schema())
    
    # Test SFT export functionality
    print("\n=== SFT Export Test ===")
    # Create a mock judge with some example data
    mock_judge = LLMJudgeBase("Rate the quality: {text}")
    mock_judge.sft_data = [
        {
            'messages': [
                {'role': 'system', 'content': 'Rate the quality: This is good text'},
                {'role': 'user', 'content': 'Please rate this text'},
                {'role': 'assistant', 'content': '{"quality": "good"}'}
            ],
            'variables': {'text': 'This is good text'},
            'input_data': 'Please rate this text',
            'output': '{"quality": "good"}'
        }
    ]
    
    sft_formats = ['messages', 'sharegpt']
    for format_name in sft_formats:
        exported = mock_judge.export_sft_data(format_name)
        print(f"SFT export ({format_name} format): {len(exported)} examples")
        if exported:
            print(f"Sample structure: {list(exported[0].keys())}")
    
    print("\n=== All Tests Completed ===")
    print("The LLMJudgeBase system is ready for use!")
    print("\nKey features:")
    print("- System prompt templating with variables")
    print("- DSPy-like Signature system")
    print("- Automatic SFT data collection")
    print("- Multiple export formats (messages, sharegpt, full)")
    print("- Chain of Thought reasoning support")