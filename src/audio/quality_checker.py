"""Audio quality assessment."""
import librosa
import numpy as np
from typing import Dict


class AudioQualityChecker:
    """Assesses audio quality metrics."""
    
    def __init__(self):
        """Initialize quality checker."""
        pass
    
    def calculate_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            SNR in dB
        """
        # Split into frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate energy per frame
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        if len(energy) == 0:
            return 0.0
        
        # Assume top 60% energy frames are signal, bottom 40% are noise
        sorted_energy = np.sort(energy)
        split_idx = int(len(sorted_energy) * 0.4)
        
        noise_energy = np.mean(sorted_energy[:split_idx]) if split_idx > 0 else 1e-10
        signal_energy = np.mean(sorted_energy[split_idx:]) if split_idx < len(sorted_energy) else 1e-10
        
        snr = 10 * np.log10(signal_energy / max(noise_energy, 1e-10))
        return max(0.0, snr)
    
    def calculate_clarity(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate audio clarity score.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Clarity score (0-1)
        """
        # Calculate zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_score = 1.0 - min(np.mean(zcr) / 0.5, 1.0)  # Lower ZCR = clearer
        
        # Calculate spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_score = min(np.mean(spectral_centroids) / 4000, 1.0)
        
        # Combine scores
        clarity = (zcr_score * 0.4 + centroid_score * 0.6)
        return clarity
    
    def calculate_duration(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate audio duration in seconds.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Duration in seconds
        """
        return len(audio) / sr
    
    def assess_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Comprehensive audio quality assessment.
        
        Args:
            audio: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary with quality metrics
        """
        snr = self.calculate_snr(audio, sr)
        clarity = self.calculate_clarity(audio, sr)
        duration = self.calculate_duration(audio, sr)
        
        # Calculate overall quality score (0-1)
        # Normalize SNR to 0-1 (assume 0-30 dB range)
        snr_normalized = min(snr / 30.0, 1.0)
        
        # Weight the scores
        overall_quality = (snr_normalized * 0.6 + clarity * 0.4)
        
        return {
            "snr": snr,
            "clarity": clarity,
            "duration": duration,
            "overall_quality": overall_quality,
            "is_acceptable": overall_quality >= 0.3  # Minimum threshold
        }
