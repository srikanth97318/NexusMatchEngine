from typing import List, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str = Field(..., description="Name of the company or organization")
    role: str = Field(..., description="Job title or role held")
    start_date: Optional[str] = Field(
        None, description="Start date of employment (e.g. YYYY-MM or MM/YYYY)"
    )
    end_date: Optional[str] = Field(
        None, description="End date of employment (e.g. YYYY-MM, MM/YYYY, or 'Present')"
    )
    duration_months: int = Field(
        ..., description="Total duration of employment in months"
    )
    description: Optional[str] = Field(
        None,
        description="Summary of key responsibilities, achievements, and tools/technologies used",
    )


class Education(BaseModel):
    institution: str = Field(..., description="Name of school, university, or academy")
    degree: str = Field(..., description="Degree or certificate obtained")
    field_of_study: Optional[str] = Field(None, description="Major field of study")
    start_date: Optional[str] = Field(None, description="Start date of studies")
    end_date: Optional[str] = Field(None, description="End/graduation date of studies")


class CandidateProfile(BaseModel):
    """Type-safe structure parsed and extracted from candidate resumes."""

    name: str = Field(..., description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Contact email address")
    phone: Optional[str] = Field(None, description="Contact phone number")
    skills: List[str] = Field(
        default_factory=list,
        description="Extracted tech stack, programming languages, and tools",
    )
    experience: List[WorkExperience] = Field(
        default_factory=list,
        description="Chronological job history representing career timeline",
    )
    education: List[Education] = Field(
        default_factory=list, description="Academic records and timeline"
    )


class JobDescription(BaseModel):
    """Type-safe structure parsed and extracted from job advertisements/descriptions."""

    title: str = Field(..., description="Target job title")
    department: Optional[str] = Field(None, description="Department name")
    required_skills: List[str] = Field(
        ...,
        description="Hard requirements: mandatory technologies, tools, and capabilities",
    )
    preferred_skills: List[str] = Field(
        default_factory=list, description="Nice-to-have or preferred skills and tools"
    )
    min_experience_years: int = Field(
        0, description="Minimum years of required professional experience"
    )
    location: Optional[str] = Field(None, description="Job location or remote status")
    visa_requirement: Optional[str] = Field(
        None, description="Any visa/citizenship requirements"
    )
    full_text: str = Field(..., description="Raw unstructured text of the JD")
