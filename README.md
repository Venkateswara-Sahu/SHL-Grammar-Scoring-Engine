# 🎙️ Grammar Scoring Engine from Voice Samples

**SHL AI Research Intern Assessment - Option 2**

An advanced, production-ready system for evaluating grammar quality from voice recordings using state-of-the-art speech recognition and NLP techniques.

## 🌟 Features

- **High-Accuracy Transcription**: Whisper-based ASR with multiple model sizes
- **Multi-Dimensional Grammar Scoring**: 
  - Syntax correctness
  - Grammar error detection
  - Fluency metrics
  - Readability scores
- **Audio Quality Assessment**: Pre-transcription quality checks
- **Confidence Metrics**: Reliability indicators for each score
- **REST API**: Easy integration with FastAPI
- **Interactive Web Demo**: User-friendly interface
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Audio     │ -> │   Audio      │ -> │     ASR     │ -> │   Grammar    │
│   Input     │    │ Preprocessing│    │  (Whisper)  │    │   Analysis   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                    │                   │
                          ↓                    ↓                   ↓
                   Quality Check         Confidence          Error Types
                                                                  ↓
                                                          ┌──────────────┐
                                                          │    Final     │
                                                          │    Score     │
                                                          └──────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- FFmpeg (for audio processing)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd SHL-Internship

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## 🚀 Quick Start

### Using the API

```bash
# Start the API server
python src/api/main.py

# In another terminal, test it
curl -X POST "http://localhost:8000/score" \
  -F "file=@path/to/audio.wav"
```

### Using Python

```python
from src.pipeline.grammar_scorer import GrammarScoringPipeline

# Initialize pipeline
pipeline = GrammarScoringPipeline()

# Score audio file
result = pipeline.score_audio("path/to/audio.wav")

print(f"Grammar Score: {result['grammar_score']:.2f}")
print(f"Transcription: {result['transcription']}")
print(f"Errors Found: {len(result['errors'])}")
```

## 📊 Output Format

```json
{
  "grammar_score": 8.5,
  "transcription": "This is the transcribed text.",
  "confidence": 0.92,
  "audio_quality": 0.87,
  "metrics": {
    "syntax_score": 9.0,
    "error_count": 2,
    "fluency_score": 8.8,
    "readability_score": 8.2
  },
  "errors": [
    {
      "type": "grammar",
      "message": "Subject-verb agreement error",
      "context": "They was going",
      "suggestion": "They were going"
    }
  ]
}
```

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

## 📁 Project Structure

```
SHL-Internship/
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Audio preprocessing
│   │   └── quality_checker.py   # Audio quality assessment
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_model.py     # Whisper ASR wrapper
│   │   └── transcriber.py       # Transcription logic
│   ├── grammar/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Grammar analysis
│   │   ├── error_detector.py    # Error detection
│   │   └── scorer.py            # Scoring algorithms
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── grammar_scorer.py    # Main pipeline
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   └── models.py            # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration
│       └── metrics.py           # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_error_analysis.ipynb
├── tests/
│   ├── test_audio.py
│   ├── test_grammar.py
│   └── test_pipeline.py
├── data/
│   ├── raw/                     # Raw audio files
│   ├── processed/               # Processed data
│   └── results/                 # Evaluation results
├── models/
│   └── whisper/                 # Cached models
├── requirements.txt
├── README.md
└── .env.example
```

## 🎯 Scoring Methodology

### Grammar Score (0-10 scale)

The final score is a weighted combination of:

1. **Syntax Correctness** (30%): Parse tree analysis
2. **Grammar Errors** (30%): Error count and severity
3. **Fluency** (20%): Speech naturalness and flow
4. **Readability** (20%): Complexity and clarity

### Confidence Score

Based on:
- Audio quality (SNR, clarity)
- ASR confidence
- Grammar analysis certainty

## 🔬 Technical Approach

1. **Audio Preprocessing**
   - Noise reduction
   - Normalization
   - Silence removal
   - Quality assessment

2. **Speech Recognition**
   - Whisper (base/small/medium models)
   - Confidence scoring
   - Word-level timestamps

3. **Grammar Analysis**
   - LanguageTool for error detection
   - spaCy for syntax analysis
   - NLTK for additional metrics
   - Custom scoring algorithms

4. **Evaluation**
   - Precision, recall, F1 for error detection
   - Correlation with human ratings
   - Error type distribution

## 📈 Performance

- Average transcription time: ~2s per minute of audio
- Grammar analysis: ~0.5s per transcript
- API response time: <5s for typical audio samples

## 🛠️ Technologies Used

- **Whisper**: State-of-the-art ASR
- **spaCy**: NLP and syntax analysis
- **LanguageTool**: Grammar checking
- **FastAPI**: High-performance API
- **PyTorch**: Deep learning framework
- **librosa**: Audio analysis

## 📝 License

MIT

## 👤 Author

**Your Name**
- Email: your.email@example.com
- GitHub: @yourusername

## 🙏 Acknowledgments

- SHL AI Team for the opportunity
- OpenAI for Whisper
- The open-source community

---

*Built with ❤️ for SHL AI Research Intern Assessment*
