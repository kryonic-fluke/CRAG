from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from app.core.llm import get_chat_llm

GENERATE_SYSTEM_PROMPT = """You are an authoritative assistant for question-answering tasks.
Answer the user's question clearly, precisely, and factually using ONLY the provided context documents.
If the context does not contain enough information to answer, explicitly state that the answer is not present in the provided sources.
Do not invent or extrapolate beyond the cited evidence."""

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATE_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved Context:\n{context}\n\nUser Question: {question}",
        ),
    ]
)


class MockGenerator:
    def invoke(self, inputs: Dict[str, Any]) -> str:
        question = inputs.get("question", "")
        context = inputs.get("context", "")
        if not context.strip():
            return "I do not have enough context to answer this question."

        first_chunk = context.split("\n\n---\n\n")[0]
        cleaned_snippet = first_chunk.replace("\n", " ").strip()[:180]
        return f"Based on the retrieved context: {cleaned_snippet}. This directly answers the query: '{question}'."


def get_generation_chain():
    llm = get_chat_llm(temperature=0.0)
    if llm is not None:
        try:
            return generation_prompt | llm | StrOutputParser()
        except Exception:
            pass

    return RunnableLambda(lambda inputs: MockGenerator().invoke(inputs))
