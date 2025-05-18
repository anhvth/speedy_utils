import pytest
from llm_utils.lm import LM

# from llm_utils.lm.text_lm import TextLM
from pydantic import BaseModel

class Output(BaseModel):
    output: str

def test_pydantic_lm_forward():
    lm_pydantic = LM("gpt-4.1-nano", cache=False)
    result: Output = lm_pydantic(
        messages=[{"role": "user", "content": "Hello world"}], response_format=Output
    )
    assert isinstance(result, Output)
    assert isinstance(result.output, str)

def test_text_lm_forward():
    lm_text = LM("gpt-4.1-nano", cache=False)
    result: str = lm_text(messages=[{"role": "user", "content": "Hello world"}])
    assert isinstance(result, str)
