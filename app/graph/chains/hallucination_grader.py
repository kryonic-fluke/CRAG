from typing import Any, Dict
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.schemas.grader import GradeHallucination

HALLUCINATION_SYSTEM_PROMPT = """You are an expert fact-checker assessing whether an LLM generation is grounded in and supported by the retrieved context documents.
Give a binary score 'yes' or 'no'.
'yes' means the generated answer is completely backed by the context facts without hallucinated claims.
'no' means the generated answer asserts claims or details not supported by the context.
Provide concise reasoning."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved Context:\n{documents}\n\nGenerated Answer:\n{generation}",
        ),
    ]
)


class MockHallucinationGrader:
    def invoke(self, inputs: Dict[str, Any]) -> GradeHallucination:
        documents = inputs.get("documents", "").lower()
        generation = inputs.get("generation", "").lower()

        if not generation.strip():
            return GradeHallucination(
                binary_score="no", reasoning="Generated answer is empty."
            )

        if not documents.strip():
            return GradeHallucination(
                binary_score="no", reasoning="No context available to ground claims."
            )

        hallucination_markers = [
            "completely unverified claim",
            "fictional fabricated fact",
            "hallucinated claim",
        ]
        for marker in hallucination_markers:
            if marker in generation:
                return GradeHallucination(
                    binary_score="no",
                    reasoning=f"Detected unsupported claim: {marker}",
                )

        stopwords = {
            "this",
            "that",
            "based",
            "on",
            "the",
            "is",
            "are",
            "and",
            "to",
            "of",
            "a",
            "an",
            "in",
            "answers",
            "query",
            "directly",
            "retrieved",
            "context",
        }
        gen_tokens = [
            t
            for t in re.findall(r"\b\w+\b", generation)
            if len(t) > 3 and t not in stopwords
        ]

        if not gen_tokens:
            return GradeHallucination(
                binary_score="yes",
                reasoning="Answer consists of neutral conversational text.",
            )

        supported_tokens = [t for t in gen_tokens if t in documents]
        overlap_ratio = len(supported_tokens) / len(gen_tokens)

        if overlap_ratio >= 0.4:
            return GradeHallucination(
                binary_score="yes",
                reasoning=f"Claims are supported by context facts ({overlap_ratio:.1%} token overlap).",
            )
        else:
            return GradeHallucination(
                binary_score="no",
                reasoning=f"Claims lack context support (only {overlap_ratio:.1%} token overlap).",
            )


def get_hallucination_grader_chain():
    key = settings.openai_api_key
    if key and key.startswith("sk-") and "mock" not in key.lower():
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=settings.openai_model_name,
                temperature=0,
                api_key=key,
            )
            structured_llm = llm.with_structured_output(GradeHallucination)
            return hallucination_prompt | structured_llm
        except Exception:
            pass

    return RunnableLambda(lambda inputs: MockHallucinationGrader().invoke(inputs))

