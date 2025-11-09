"""Pydantic models for API."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class GrammarScoreResponse(BaseModel):
    """Response model for grammar scoring."""
    success: bool
    grammar_score: Optional[float] = None
    grade: Optional[str] = None
    transcription: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    metrics: Optional[Dict] = None
    detailed_analysis: Optional[Dict] = None
    error: Optional[str] = None


class TextScoreRequest(BaseModel):
    """Request model for text scoring."""
    text: str = Field(..., description="Text to analyze", min_length=1)
    return_detailed: bool = Field(default=True, description="Return detailed analysis")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
