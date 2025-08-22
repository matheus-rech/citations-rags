from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class InContext(BaseModel):
    text: str = Field(description="Verbatim text span")
    start_offset: int = Field(ge=0, description="0-based inclusive start offset")
    end_offset: int = Field(ge=0, description="0-based exclusive end offset")


class StudyMetadata(BaseModel):
    study_title: str
    first_author: str
    year: int
    journal: str
    pmid: str


class SourceCitation(BaseModel):
    citation_id: str
    location: str
    quote: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class CohortGroup(BaseModel):
    cohort_group_id: str
    label: str
    in_context: InContext


class OutcomeValue(BaseModel):
    outcome_name: str
    outcome_type: str
    cohort_group_id: str
    timepoint: str
    statistic: str
    value: str
    unit: str
    citation_ids: List[str]
    in_context: InContext


class StudyExtraction(BaseModel):
    study_metadata: StudyMetadata
    source_citations: List[SourceCitation]
    cohort_groups: List[CohortGroup]
    outcome_values: List[OutcomeValue]
