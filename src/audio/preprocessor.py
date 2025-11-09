"""Audio preprocessing and enhancement."""
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Handles audio preprocessing and enhancement."""
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sample rate for audio
        """
        self.target_sr = target_sr
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return audio, sr
    
    def remove_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Remove background noise from audio.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Denoised audio
        """
        try:
            # Use noise reduction
            reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
            return reduced_noise
        except Exception as e:
            print(f"Warning: Noise reduction failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio
        """
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def remove_silence(self, audio: np.ndarray, sr: int, 
                       top_db: int = 20) -> np.ndarray:
        """
        Remove silence from audio.
        
        Args:
            audio: Audio data
            sr: Sample rate
            top_db: Threshold for silence detection
            
        Returns:
            Audio with silence removed
        """
        # Split audio at silence
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio
        
        # Concatenate non-silent intervals
        audio_trimmed = np.concatenate([audio[start:end] for start, end in intervals])
        return audio_trimmed
    
    def preprocess(self, audio_path: str, 
                   remove_noise: bool = True,
                   remove_silence: bool = True,
                   normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file
            remove_noise: Whether to apply noise reduction
            remove_silence: Whether to remove silence
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Remove noise
        if remove_noise:
            audio = self.remove_noise(audio, sr)
        
        # Remove silence
        if remove_silence:
            audio = self.remove_silence(audio, sr)
        
        # Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str):
        """
        Save processed audio to file.
        
        Args:
            audio: Audio data
            sr: Sample rate
            output_path: Output file path
        """
        sf.write(output_path, audio, sr)
