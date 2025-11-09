"""Whisper ASR model wrapper."""
import whisper
import torch
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class WhisperModel:
    """Wrapper for OpenAI Whisper model."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to run model on (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Model loaded successfully!")
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        task: str = "transcribe",
        **kwargs
    ) -> Dict:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data (numpy array)
            language: Language code (default: "en")
            task: Task type ("transcribe" or "translate")
            **kwargs: Additional arguments for whisper.transcribe()
            
        Returns:
            Dictionary containing transcription results
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=language,
            task=task,
            verbose=False,
            **kwargs
        )
        
        return result
    
    def get_confidence_scores(self, result: Dict) -> Dict[str, float]:
        """
        Extract confidence scores from transcription result.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Dictionary with confidence metrics
        """
        segments = result.get("segments", [])
        
        if not segments:
            return {
                "overall_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "avg_confidence": 0.0
            }
        
        # Extract confidence scores (avg_logprob as proxy)
        # Note: Whisper doesn't directly provide confidence, we use log probability
        confidences = []
        for segment in segments:
            # Convert log probability to confidence-like score
            # avg_logprob typically ranges from -1 to 0
            logprob = segment.get("avg_logprob", -1.0)
            confidence = min(max((logprob + 1.0), 0.0), 1.0)  # Normalize to 0-1
            confidences.append(confidence)
        
        return {
            "overall_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "avg_confidence": np.mean(confidences),
            "confidence_variance": np.var(confidences)
        }
