from typing import List

from pydantic import BaseModel, Field


class Numeric(BaseModel):
    sidx: int = Field("Simple sentence index position.")
    text: str = Field("Numeric value")
    start_char: int
    end_char: int


class LeafLabel(BaseModel):
    """Metadata provided by Stanza"""

    id: int
    text: str
    upos: str
    xpos: str
    feats: str = Field(default="")
    start_char: int
    end_char: int
    misc: str = Field(default="")


class SentenceMetadata(BaseModel):
    sidx: int = Field("Simple sentence index position.")
    subject: List[LeafLabel] = Field(default=[])
    predicate: List[LeafLabel] = Field(default=[])
    subject_noun_phrases: List[List[LeafLabel]] = Field(default=[])
    predicate_noun_phrases: List[List[LeafLabel]] = Field(default=[])


class Sentence(BaseModel):
    sidx: int = Field("Simple sentence index position.")
    subject: List[str] = Field(default=[])
    predicate: List[str] = Field(default=[])
    subject_text: str = Field(default="")
    predicate_text: str = Field(default="")
