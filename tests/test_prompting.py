import asyncio

from src.agent.core.agent import (
    IntentProcessor,
    PromptEngineeringGuide,
    PromptTechnique,
    TaskIntent,
)


class RecordingLlm:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    async def generate_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_prompt_guide_formats_instructions_without_duplicates():
    guide = PromptEngineeringGuide()
    formatted = guide.format_instructions(
        [
            PromptTechnique.ZERO_SHOT,
            PromptTechnique.REACT,
            PromptTechnique.ZERO_SHOT,
        ]
    )

    assert PromptTechnique.ZERO_SHOT.value in formatted
    assert formatted.count(PromptTechnique.ZERO_SHOT.value) == 1
    assert "Apply the following prompt engineering" in formatted


def test_intent_processor_includes_prompting_instructions():
    llm = RecordingLlm('{"steps": [], "resources": {}, "duration": 0, "metrics": {}}')
    processor = IntentProcessor(llm)
    intent = TaskIntent(description="Summarize PDF", context={}, constraints={})

    asyncio.run(
        processor.process_intent(
            intent,
            techniques=[PromptTechnique.CHAIN_OF_THOUGHT, PromptTechnique.RETRIEVAL_AUGMENTED],
        )
    )

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert PromptTechnique.CHAIN_OF_THOUGHT.value in prompt
    assert PromptTechnique.RETRIEVAL_AUGMENTED.value in prompt
    assert "Generate a detailed plan" in prompt


def test_intent_processor_omits_prompting_section_when_not_requested():
    llm = RecordingLlm('{"steps": [], "resources": {}, "duration": 0, "metrics": {}}')
    processor = IntentProcessor(llm)
    intent = TaskIntent(description="Summarize PDF", context={}, constraints={})

    asyncio.run(processor.process_intent(intent))

    prompt = llm.prompts[0]
    for technique in PromptTechnique:
        assert technique.value not in prompt
