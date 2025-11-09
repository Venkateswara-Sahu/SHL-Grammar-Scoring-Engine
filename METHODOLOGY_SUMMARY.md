# Methodology Summary for SHL Submission

## 100-Word Summary (For Submission Form):

Our approach combines OpenAI Whisper ASR with multi-dimensional NLP analysis for grammar scoring. The pipeline processes audio through noise reduction and quality checks, then transcribes using Whisper's base model. Grammar analysis incorporates four weighted components: syntax correctness via spaCy dependency parsing (30%), error detection with LanguageTool (30%), fluency metrics (20%), and readability scores (20%). Raw scores (0-10 scale) are calibrated to the target MOS Likert scale (1-5) using linear regression trained on labeled data. The model achieves strong correlation with human ratings while maintaining robust error handling for audio quality variations.

---

## Extended Methodology

### 1. Audio Preprocessing
- Noise reduction using spectral gating
- Normalization and silence removal
- Quality assessment (SNR, clarity metrics)

### 2. Speech Recognition
- OpenAI Whisper (base model) for transcription
- Confidence scoring for reliability metrics
- Robust error handling for low-quality audio

### 3. Grammar Analysis (Multi-Dimensional)
- **Syntax (30%)**: Dependency parsing with spaCy
- **Errors (30%)**: LanguageTool for grammar mistakes
- **Fluency (20%)**: Speech pattern analysis
- **Readability (20%)**: Flesch-Kincaid and complexity metrics

### 4. Calibration
- Linear regression mapping from 0-10 to 1-5 scale
- Trained on 409 labeled training samples
- Clipping to ensure valid range [1, 5]

### 5. Evaluation
- RMSE and Pearson Correlation on training data
- Comprehensive visualizations and error analysis
- Component-level correlation analysis

---

## Key Features
- State-of-the-art ASR technology
- Multi-dimensional grammar assessment
- Data-driven calibration
- Robust error handling
- Production-ready architecture
