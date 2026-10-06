from typing import Any, Dict
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.llm import get_chat_llm
from app.schemas.grader import GradeDocuments

GRADER_SYSTEM_PROMPT = """You are an expert document relevance grader assessing whether a retrieved document chunk is relevant to a user question.
Evaluate the document content against the user question.
If the document contains information, keywords, or semantic meaning that helps answer the user question, grade it as relevant ('yes').
If the document is completely off-topic, unrelated noise, or does not help answer the question, grade it as irrelevant ('no').
Provide a concise reasoning alongside your binary score."""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved document chunk:\n{document}\n\nUser question: {question}",
        ),
    ]
)


class MockDocGrader:
    def invoke(self, inputs: Dict[str, Any]) -> GradeDocuments:
        question = inputs.get("question", "").lower()
        document = inputs.get("document", "").lower()

        stopwords = {
            "what",
            "how",
            "is",
            "the",
            "are",
            "for",
            "and",
            "to",
            "in",
            "a",
            "an",
            "of",
            "does",
            "do",
            "on",
            "with",
            "using",
            "about",
        }
        q_tokens = [
            t
            for t in re.findall(r"\b\w+\b", question)
            if len(t) > 2 and t not in stopwords
        ]

        if not q_tokens:
            return GradeDocuments(
                binary_score="no", reasoning="No informative query tokens found."
            )

        matches = [t for t in q_tokens if t in document]
        match_ratio = len(matches) / len(q_tokens)

        if match_ratio >= 0.25:
            return GradeDocuments(
                binary_score="yes",
                reasoning=f"Document contains key terms matching query: {matches}.",
            )
        else:
            return GradeDocuments(
                binary_score="no",
                reasoning=f"Document lacks sufficient query overlap. Matched only: {matches}.",
            )


def get_doc_grader_chain():
    llm = get_chat_llm(temperature=0.0)
    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(GradeDocuments)
            return grader_prompt | structured_llm
        except Exception:
            pass

    return RunnableLambda(lambda inputs: MockDocGrader().invoke(inputs))
