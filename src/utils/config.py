# Configuration
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
LANGUAGE = "en"
DEVICE = "cpu"  # Change to "cuda" if GPU available

# Audio Processing
SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 1.0  # seconds
MAX_AUDIO_LENGTH = 300.0  # 5 minutes

# Scoring Weights
WEIGHTS = {
    "syntax": 0.30,
    "grammar_errors": 0.30,
    "fluency": 0.20,
    "readability": 0.20
}

# Quality Thresholds
MIN_AUDIO_QUALITY = 0.3
MIN_CONFIDENCE = 0.5

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8001
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
