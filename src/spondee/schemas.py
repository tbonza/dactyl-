from typing import List

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    sidx: int = Field("Simple sentence index position.")
    subject: List[str] = Field(default=[])
    subject_text: List[str] = Field(default=[])
    predicate: List[str] = Field(default=[])
    predicate_text: List[str] = Field(default=[])


class Numeric(BaseModel):
    sidx: int = Field("Simple sentence index position.")
    text: str = Field("Numeric value")
    start_char: int
    end_char: int
