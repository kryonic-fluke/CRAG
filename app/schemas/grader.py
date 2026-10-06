from typing import Literal
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Binary score indicating whether the document is relevant to the question. 'yes' if relevant, 'no' if irrelevant.",
    )
    reasoning: str = Field(
        ...,
        description="Concise rationale explaining why the document is or is not relevant to the question.",
    )


class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Binary score indicating whether the generation is grounded in facts from the retrieved context. 'yes' if grounded, 'no' if hallucinated.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation detailing which parts of the context support or contradict the generated response.",
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Binary score indicating whether the generation answers the user question. 'yes' if it answers, 'no' if not.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of whether the answer directly addresses all aspects of the user query.",
    )
