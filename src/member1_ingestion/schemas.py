from typing import List, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str = Field(..., description="Name of the company or organization")
    role: str = Field(..., description="Job title or role held")
    duration_months: int = Field(..., description="Duration of employment in months")
    description: Optional[str] = Field(None, description="Key duties and achievements")


class Education(BaseModel):
    institution: str = Field(..., description="Name of school, university, or academy")
    degree: str = Field(..., description="Degree or certificate obtained")
    field_of_study: Optional[str] = Field(None, description="Major field of study")


class CandidateProfile(BaseModel):
    """Type-safe structure parsed and extracted from candidate resumes."""
    name: str = Field(..., description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Contact email address")
    skills: List[str] = Field(default_factory=list, description="Extracted tech stack and soft skills")
    experience: List[WorkExperience] = Field(default_factory=list, description="Chronological job history")
    education: List[Education] = Field(default_factory=list, description="Academic records")


class JobDescription(BaseModel):
    """Type-safe structure parsed and extracted from job advertisements/descriptions."""
    title: str = Field(..., description="Target job title")
    department: Optional[str] = Field(None, description="Department name")
    required_skills: List[str] = Field(..., description="Mandatory technologies and capabilities")
    preferred_skills: List[str] = Field(default_factory=list, description="Bonus skills/nice-to-haves")
    min_experience_years: int = Field(0, description="Minimum years of required professional experience")
    full_text: str = Field(..., description="Raw unstructured text of the JD")
