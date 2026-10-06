from typing import Any, Dict
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.llm import get_chat_llm
from app.schemas.grader import GradeAnswer

ANSWER_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an answer directly addresses and resolves a user question.
Give a binary score 'yes' or 'no'.
'yes' means the answer directly resolves the question.
'no' means the answer is evasive, off-topic, or fails to address the question.
Provide concise reasoning."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        (
            "human",
            "User Question:\n{question}\n\nGenerated Answer:\n{generation}",
        ),
    ]
)


class MockAnswerGrader:
    def invoke(self, inputs: Dict[str, Any]) -> GradeAnswer:
        question = inputs.get("question", "").lower()
        generation = inputs.get("generation", "").lower()

        if not generation.strip():
            return GradeAnswer(
                binary_score="no", reasoning="Generated answer is empty."
            )

        if "do not have enough context" in generation or "cannot answer" in generation:
            return GradeAnswer(
                binary_score="no",
                reasoning="Answer states context is insufficient to resolve question.",
            )

        stopwords = {
            "what",
            "how",
            "is",
            "are",
            "the",
            "and",
            "to",
            "in",
            "of",
            "for",
            "a",
            "an",
            "does",
            "do",
        }
        q_tokens = [
            t
            for t in re.findall(r"\b\w+\b", question)
            if len(t) > 2 and t not in stopwords
        ]

        if not q_tokens:
            return GradeAnswer(
                binary_score="yes", reasoning="Neutral question resolved."
            )

        matches = [t for t in q_tokens if t in generation]
        if len(matches) > 0:
            return GradeAnswer(
                binary_score="yes",
                reasoning=f"Answer directly addresses question topics: {matches}.",
            )
        else:
            return GradeAnswer(
                binary_score="no",
                reasoning="Answer does not mention key topics from the query.",
            )


def get_answer_grader_chain():
    llm = get_chat_llm(temperature=0.0)
    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(GradeAnswer)
            return answer_prompt | structured_llm
        except Exception:
            pass

    return RunnableLambda(lambda inputs: MockAnswerGrader().invoke(inputs))
