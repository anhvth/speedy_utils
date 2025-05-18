from llm_utils import lm
from llm_utils.lm import PydanticLM
from pydantic import BaseModel

from llm_utils.lm.text_lm import TextLM


class Output(BaseModel):
    output: str


lm_pydantic = PydanticLM("gpt-4.1-nano", cache=False)
output = lm_pydantic(prompt="Hello world", response_format=Output)

lm_text = TextLM("gpt-4.1-nano", cache=False)
output_text = lm_text(prompt="Hello world")
