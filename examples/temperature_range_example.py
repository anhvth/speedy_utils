"""Example demonstrating temperature range sampling with LLM."""

from pydantic import BaseModel

from llm_utils import LLM


class CreativeStory(BaseModel):
    """A creative story output."""

    title: str
    story: str
    moral: str


def example_temperature_range_text():
    """Example: Sample text responses with different temperatures."""
    print("=" * 60)
    print("Example 1: Temperature Range Sampling (Text Completion)")
    print("=" * 60)

    llm = LLM(
        instruction="You are a creative writer. Write a very short story.",
        output_model=str,
    )

    prompt = "Write a one-sentence story about a brave mouse."

    # Sample with 5 different temperatures from 0.1 to 1.0
    responses = llm(
        prompt,
        temperature_ranges=(0.1, 1.0),
        n=5,
    )

    print(f"\nGenerated {len(responses)} responses with varying temperatures:\n")
    for i, resp in enumerate(responses):
        temp = 0.1 + i * ((1.0 - 0.1) / (5 - 1))
        print(f"Temperature ~{temp:.2f}:")
        print(f"  {resp['parsed']}\n")


def example_temperature_range_pydantic():
    """Example: Sample structured responses with different temperatures."""
    print("=" * 60)
    print("Example 2: Temperature Range with Pydantic Models")
    print("=" * 60)

    llm = LLM(
        instruction="Create a creative short story with a moral lesson.",
        output_model=CreativeStory,
    )

    prompt = "Topic: A robot learning to feel emotions"

    # Sample with 3 different temperatures from 0.5 to 1.5
    responses = llm(
        prompt,
        temperature_ranges=(0.5, 1.5),
        n=3,
    )

    print(f"\nGenerated {len(responses)} stories with varying creativity:\n")
    for i, resp in enumerate(responses):
        temp = 0.5 + i * ((1.5 - 0.5) / (3 - 1))
        story = resp["parsed"]
        print(f"Temperature ~{temp:.2f}:")
        print(f"  Title: {story.title}")
        print(f"  Story: {story.story[:80]}...")
        print(f"  Moral: {story.moral}\n")


def example_two_step_parsing():
    """Example: Two-step Pydantic parsing for models with reasoning."""
    print("=" * 60)
    print("Example 3: Two-Step Pydantic Parsing")
    print("=" * 60)

    llm = LLM(
        instruction=(
            "Analyze the given text and extract structured information. Think through your analysis first."
        ),
        output_model=CreativeStory,
    )

    prompt = "Analyze the story: 'The tortoise won the race by persistence.'"

    # Use two-step parsing (useful for reasoning models)
    response = llm(
        prompt,
        two_step_parse_pydantic=True,
    )[0]

    story = response["parsed"]
    print("\nExtracted structure:")
    print(f"  Title: {story.title}")
    print(f"  Story: {story.story}")
    print(f"  Moral: {story.moral}")


if __name__ == "__main__":
    # Run examples
    # Note: These require a working OpenAI API key or local LLM server

    try:
        example_temperature_range_text()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")

    try:
        example_temperature_range_pydantic()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")

    try:
        example_two_step_parsing()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
