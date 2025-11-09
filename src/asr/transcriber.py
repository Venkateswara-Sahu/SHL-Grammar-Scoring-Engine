"""Transcription service."""
import numpy as np
from typing import Dict, Optional
from .whisper_model import WhisperModel


class Transcriber:
    """Handles audio transcription."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size
            device: Device to run model on
        """
        self.whisper = WhisperModel(model_size=model_size, device=device)
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        language: str = "en",
        return_segments: bool = False
    ) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data
            language: Language code
            return_segments: Whether to return word-level segments
            
        Returns:
            Dictionary with transcription and metadata
        """
        # Perform transcription
        result = self.whisper.transcribe(
            audio,
            language=language,
            word_timestamps=return_segments
        )
        
        # Extract confidence scores
        confidence = self.whisper.get_confidence_scores(result)
        
        # Prepare output
        output = {
            "text": result["text"].strip(),
            "language": result["language"],
            "confidence": confidence["overall_confidence"],
            "confidence_details": confidence
        }
        
        # Add segments if requested
        if return_segments:
            output["segments"] = result.get("segments", [])
        
        return output
