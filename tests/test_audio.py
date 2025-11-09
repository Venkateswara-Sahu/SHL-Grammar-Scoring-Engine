"""Tests for audio processing modules."""
import pytest
import numpy as np
from src.audio.preprocessor import AudioPreprocessor
from src.audio.quality_checker import AudioQualityChecker


def test_audio_preprocessor_initialization():
    """Test audio preprocessor initialization."""
    preprocessor = AudioPreprocessor(target_sr=16000)
    assert preprocessor.target_sr == 16000


def test_normalize_audio():
    """Test audio normalization."""
    preprocessor = AudioPreprocessor()
    
    # Create test audio
    audio = np.array([0.5, -0.3, 0.8, -0.9])
    normalized = preprocessor.normalize_audio(audio)
    
    # Check if normalized to [-1, 1]
    assert np.max(np.abs(normalized)) <= 1.0
    assert np.max(np.abs(normalized)) == 1.0  # At least one value should be at max


def test_quality_checker_initialization():
    """Test quality checker initialization."""
    checker = AudioQualityChecker()
    assert checker is not None


def test_calculate_duration():
    """Test duration calculation."""
    checker = AudioQualityChecker()
    
    # Create 1 second of audio at 16000 Hz
    audio = np.random.randn(16000)
    sr = 16000
    
    duration = checker.calculate_duration(audio, sr)
    assert abs(duration - 1.0) < 0.01  # Should be close to 1 second


def test_assess_quality():
    """Test quality assessment."""
    checker = AudioQualityChecker()
    
    # Create test audio
    audio = np.random.randn(16000)
    sr = 16000
    
    quality = checker.assess_quality(audio, sr)
    
    # Check if all expected keys are present
    assert "snr" in quality
    assert "clarity" in quality
    assert "duration" in quality
    assert "overall_quality" in quality
    assert "is_acceptable" in quality
    
    # Check value ranges
    assert 0 <= quality["overall_quality"] <= 1
    assert isinstance(quality["is_acceptable"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
