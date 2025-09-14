#!/usr/bin/env python3
"""
Test script to demonstrate the new key renaming functionality.
"""

from typing import Tuple, List, Union
from pydantic import BaseModel
from src.llm_utils.lm.llm_task import BasePromptBuilder


class EmailBuilder(BasePromptBuilder):
    # Input fields
    topic: str
    tone: str = "professional"
    
    # Output fields
    email_content: str
    word_count: int
    estimated_read_time: int
    
    def get_instruction(self) -> str:
        return "Generate professional email content based on the given topic and tone."
    
    def get_io_keys(self) -> Tuple[List[str], List[Union[str, Tuple[str, str]]]]:
        input_keys = ["topic", "tone"]
        output_keys = [
            ("email_content", "content"),  # Rename email_content to content
            "word_count",                  # Keep as is
            ("estimated_read_time", "read_time")  # Rename estimated_read_time to read_time
        ]
        return input_keys, output_keys


def test_key_renaming():
    # Create an instance
    builder = EmailBuilder(
        topic="Meeting follow-up",
        tone="professional",
        email_content="Thank you for attending our meeting today...",
        word_count=150,
        estimated_read_time=2
    )
    
    print("Original model fields:")
    print(f"Input keys: {builder.get_io_keys()[0]}")
    print(f"Output keys: {builder.get_io_keys()[1]}")
    print()
    
    # Test input model (should use original field names)
    input_model = builder.get_input_model()
    print("Input model fields:", list(input_model.model_fields.keys()))
    print("Input JSON:")
    print(builder.model_dump_json_input(indent=2))
    print()
    
    # Test output model (should use renamed field names)
    output_model = builder.get_output_model()
    print("Output model fields:", list(output_model.model_fields.keys()))
    print("Output JSON:")
    print(builder.model_dump_json_output(indent=2))
    print()
    
    # Test full training data
    print("Training data:")
    training_data = builder.build_training_data()
    print(training_data)


if __name__ == "__main__":
    test_key_renaming()