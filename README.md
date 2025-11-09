# **🎙️ Grammar Scoring Engine for Spoken Audio**

**SHL Intern Hiring Assessment 2025**

A comprehensive machine learning solution for evaluating the grammatical quality of spoken audio samples on a scale of 1-5 (MOS Likert Scale). This project combines state-of-the-art speech recognition (Whisper) with advanced NLP techniques for multi-dimensional grammar analysis.

## **📚 Table of Contents**

* [Competition Overview](https://www.google.com/search?q=%23-competition-overview)  
* [Project Results](https://www.google.com/search?q=%23-project-results)  
* [Key Features](https://www.google.com/search?q=%23-key-features)  
* [Pipeline Architecture](https://www.google.com/search?q=%23%EF%B8%8F-pipeline-architecture)  
* [Technologies & Libraries](https://www.google.com/search?q=%23-technologies--libraries)  
* [Installation & Setup](https://www.google.com/search?q=%23-installation--setup)  
* [Quick Start](https://www.google.com/search?q=%23-quick-start)  
* [Project Deliverables](https://www.google.com/search?q=%23-project-deliverables)  
* [Project Structure](https://www.google.com/search?q=%23-project-structure)  
* [Methodology](https://www.google.com/search?q=%23-methodology)  
* [Performance Metrics](https://www.google.com/search?q=%23-performance-metrics)  
* [Key Insights](https://www.google.com/search?q=%23-key-insights)  
* [Future Improvements](https://www.google.com/search?q=%23-future-improvements)  
* [Author & Submission](https://www.google.com/search?q=%23-author--submission)  
* [Acknowledgments](https://www.google.com/search?q=%23-acknowledgments)  
* [License](https://www.google.com/search?q=%23-license)

## **🎯 Competition Overview**

This is a submission for the Kaggle competition **SHL Intern Hiring Assessment 2025 \- Grammar Scoring Engine**.

* **Task**: Predict MOS Likert grammar scores (1-5) for spoken audio.  
* **Training Dataset**: 409 audio samples (45-60 seconds each) with MOS Likert grammar scores (1-5).  
* **Test Dataset**: 197 audio samples for prediction.  
* **Evaluation Metrics**: RMSE and Pearson Correlation.  
* **Format**: WAV audio files

## **📊 Project Results**

* **Training Samples**: 402/409 processed (**98.3%** success rate)  
* **Test Samples**: 197 predictions generated  
* **RMSE (Training)**: **0.76**  
* **Pearson Correlation**: **0.084**  
* **Average Processing Time**: \~6 seconds per audio file

## **🌟 Key Features**

* **State-of-the-Art ASR**: OpenAI Whisper (base model) for accurate transcription.  
* **Multi-Dimensional Grammar Analysis**:  
  * Syntax correctness (30% weight) \- spaCy dependency parsing  
  * Grammar error detection (30% weight) \- LanguageTool  
  * Fluency metrics (20% weight) \- Speech pattern analysis  
  * Readability scores (20% weight) \- Flesch-Kincaid metrics  
* **Intelligent Calibration**: Linear regression to map 0-10 scores to 1-5 MOS scale.  
* **Audio Quality Assessment**: Pre-transcription quality checks.  
* **Robust Error Handling**: Fallback mechanisms for failed transcriptions.  
* **Comprehensive Jupyter Notebook**: Complete analysis with visualizations.

## **🏗️ Pipeline Architecture**

Audio Input (WAV)  
↓  
Audio Preprocessing (Noise Reduction, Normalization)  
↓  
Quality Assessment (SNR, Clarity)  
↓  
ASR \- Whisper Base Model (Transcription)  
↓  
Grammar Analysis (Multi-Dimensional)  
├── Syntax Analysis (spaCy) → 30%  
├── Error Detection (LanguageTool) → 30%  
├── Fluency Metrics (NLTK) → 20%  
└── Readability (Flesch-Kincaid) → 20%  
↓  
Raw Score (0-10 scale)  
↓  
Calibration (Linear Regression)  
↓  
Final Score (1-5 MOS Likert Scale)

## **🛠️ Technologies & Libraries**

| Category | Technology | Purpose |
| :---- | :---- | :---- |
| ASR | OpenAI Whisper | State-of-the-art speech recognition |
| NLP | spaCy 3.8 | Syntax analysis, dependency parsing |
| Grammar | LanguageTool | Grammar error detection |
| Audio | librosa 0.10 | Audio processing, feature extraction |
| Audio | noisereduce | Noise reduction |
| ML | scikit-learn | Calibration, metrics |
| API | FastAPI | REST API framework |
| DL | PyTorch 2.0+ | Whisper model backend |
| Data | pandas, numpy | Data manipulation |
| Viz | matplotlib, seaborn | Visualizations |

## **📦 Installation & Setup**

### **Prerequisites**

* Python 3.12+  
* FFmpeg (for audio processing)  
* Git

### **Quick Setup**

\# Clone the repository  
git clone \[https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine.git\](https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine.git)  
cd SHL-Grammar-Scoring-Engine

\# Create virtual environment  
python \-m venv venv

\# Activate virtual environment  
\# Windows  
venv\\Scripts\\activate  
\# Linux/Mac  
\# source venv/bin/activate

\# Install all dependencies  
pip install \-r requirements.txt

\# Install additional requirements for notebooks  
pip install jupyter notebook ipykernel noisereduce

\# Download spaCy English model  
python \-m spacy download en\_core\_web\_sm

\# Download NLTK data  
python \-c "import nltk; nltk.download('punkt'); nltk.download('averaged\_perceptron\_tagger')"

## **🚀 Quick Start**

### **1\. Running the Jupyter Notebook (Recommended)**

The main analysis, training, and prediction generation are in Grammar\_Scoring\_Engine.ipynb.

\# Activate virtual environment  
venv\\Scripts\\activate

\# Start Jupyter Notebook  
jupyter notebook Grammar\_Scoring\_Engine.ipynb

The notebook includes:

* ✅ Complete data exploration with visualizations  
* ✅ Audio file analysis (waveforms, spectrograms)  
* ✅ Full pipeline demonstration  
* ✅ Training on 409 samples with RMSE & Pearson metrics  
* ✅ Test predictions for 197 samples  
* ✅ Calibration visualization  
* ✅ Error analysis by score ranges  
* ✅ submission.csv generation

### **2\. Using the Pipeline Programmatically**

from src.pipeline.grammar\_scorer import GrammarScoringPipeline

\# Initialize pipeline  
pipeline \= GrammarScoringPipeline()

\# Score audio file (returns 0-10 scale)  
result \= pipeline.score\_audio("path/to/audio.wav", preprocess=True)

print(f"Raw Grammar Score (0-10): {result\['grammar\_score'\]:.2f}")  
print(f"Transcription: {result\['transcription'\]}")  
print(f"Confidence: {result\['confidence'\]:.2f}")

### **3\. Using the REST API**

\# Start the API server (runs on port 8001\)  
cd src/api  
python main.py

Test the API from another terminal:

\# Test the API  
curl \-X POST "http://localhost:8001/score/audio" \\  
  \-F "file=@path/to/audio.wav"

## **📊 Project Deliverables**

### **Main Files**

* Grammar\_Scoring\_Engine.ipynb: ⭐ **Main Jupyter Notebook** with complete analysis, training, evaluation, and test prediction generation.  
* submission.csv: ⭐ **Kaggle Competition Predictions** (197 test samples).  
* data/results/training\_metrics.json: Performance metrics (RMSE, Pearson).  
* data/results/training\_predictions.csv: Detailed results from the training set.

### **Pipeline Output Format**

The programmatic pipeline and API return a detailed JSON object:

{  
  "success": true,  
  "grammar\_score": 8.37,  
  "grade": "Very Good",  
  "transcription": "This is the transcribed text.",  
  "confidence": 0.92,  
  "processing\_time": 6.24,  
  "metrics": {  
    "syntax\_score": 9.33,  
    "error\_count": 1,  
    "fluency\_score": 6.56,  
    "readability\_score": 6.57  
  }  
}

## **📁 Project Structure**

SHL-Grammar-Scoring-Engine/  
├── Grammar\_Scoring\_Engine.ipynb     \# ⭐ Main Jupyter Notebook (Primary Deliverable)  
├── submission.csv                 \# ⭐ Kaggle Competition Predictions  
│  
├── src/                             \# Source code modules  
│   ├── audio/  
│   │   ├── preprocessor.py        \# Noise reduction, normalization  
│   │   └── quality\_checker.py     \# SNR and quality metrics  
│   ├── asr/  
│   │   ├── whisper\_model.py       \# Whisper ASR wrapper  
│   │   └── transcriber.py         \# Transcription service  
│   ├── grammar/  
│   │   ├── analyzer.py            \# Syntax, fluency, readability  
│   │   ├── error\_detector.py      \# Grammar error detection  
│   │   └── scorer.py              \# Weighted scoring algorithm  
│   ├── pipeline/  
│   │   └── grammar\_scorer.py      \# Main end-to-end pipeline  
│   ├── api/  
│   │   ├── main.py                \# FastAPI REST API  
│   │   └── models.py              \# Pydantic schemas  
│   └── utils/  
│       ├── config.py              \# Configuration  
│       └── metrics.py             \# Evaluation utilities  
│  
├── data/  
│   ├── raw/  
│   │   ├── train/                 \# 409 training audio files  
│   │   ├── test/                  \# 197 test audio files  
│   │   ├── train.csv              \# Training labels  
│   │   └── test.csv               \# Test filenames  
│   └── results/  
│       ├── training\_metrics.json    \# RMSE, Pearson correlation  
│       └── training\_predictions.csv \# Detailed results  
│  
├── tests/                         \# Unit tests  
├── requirements.txt               \# Python dependencies  
├── README.md                      \# This file  
└── .gitignore

## **🎯 Methodology**

### **1\. Audio Preprocessing**

* **Noise Reduction**: Spectral gating using noisereduce.  
* **Normalization**: RMS-based audio level normalization.  
* **Quality Assessment**: SNR calculation and clarity metrics.  
* **Duration Check**: Validate 1-300 second audio clips.

### **2\. Speech-to-Text (ASR)**

* **Model**: OpenAI Whisper Base (74M parameters).  
* **Configuration**: English language, CPU inference.  
* **Output**: Transcribed text with confidence scores.  
* **Average Time**: \~4-5 seconds per audio file.

### **3\. Multi-Dimensional Grammar Analysis**

A raw score (0-10 scale) is calculated as a weighted combination of:

* **Syntax Correctness (30%)**:  
  * spaCy dependency parsing  
  * Parse tree depth and complexity  
  * Sentence structure validation  
* **Grammar Errors (30%)**:  
  * LanguageTool error detection (with spaCy fallback)  
  * Error type classification and severity weighting  
* **Fluency Metrics (20%)**:  
  * Average sentence length  
  * Word complexity  
  * Speech pattern naturalness  
* **Readability Scores (20%)**:  
  * Flesch-Kincaid Grade Level  
  * Gunning Fog Index

### **4\. Calibration (0-10 → 1-5 Scale)**

* **Model**: A Linear Regression model is trained on the 402 labeled training samples.  
* **Task**: Maps the raw 0-10 scores to the final 1-5 MOS Likert scale.  
* **Formula**: predicted\_label \= slope × raw\_score \+ intercept  
* **Constraint**: Predictions are clipped to the valid range \[1, 5\].

### **5\. Evaluation Metrics**

* **RMSE (Root Mean Square Error)**: 0.76  
* **Pearson Correlation**: 0.084  
* **MAE (Mean Absolute Error)**: 0.55  
* **Processing Success Rate**: 98.3% (402/409 samples)

## **📈 Performance Metrics**

| Metric | Value |
| :---- | :---- |
| Average Processing Time | \~6 seconds per audio file |
| Training Success Rate | 98.3% (402/409) |
| Test Predictions | 197 samples |
| **RMSE (Training)** | **0.7604** |
| **Pearson Correlation** | **0.0836** |
| Mean Absolute Error | 0.5513 |
| Mean Predicted Score | \~2.5-3.0 (1-5 scale) |

## **🎓 Key Insights**

* **Multi-dimensional analysis** works better than single-metric scoring.  
* **Calibration is crucial** for mapping different score scales.  
* **Audio quality** significantly impacts transcription accuracy.  
* **Whisper base model** provides a good balance of speed and accuracy.  
* **Fallback mechanisms** improve robustness (e.g., spaCy when LanguageTool fails).

## **🚧 Future Improvements**

* Fine-tune Whisper on domain-specific data.  
* Incorporate acoustic features (pitch, energy, speaking rate).  
* Ensemble methods combining multiple scoring approaches.  
* Speaker normalization for accent/dialect variations.  
* GPU acceleration for faster processing.

## **👤 Author & Submission**

* **Venkateswara Sahu**  
* **GitHub**: [@Venkateswara-Sahu](https://www.google.com/search?q=https://github.com/Venkateswara-Sahu)  
* **Repository**: [https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine](https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine)  
* **Submission Date**: November 9, 2025

## **🙏 Acknowledgments**

* SHL Team for the challenging assessment opportunity  
* OpenAI for the Whisper ASR model  
* spaCy team for excellent NLP tools  
* Kaggle for hosting the competition

## **📝 License**

This project is submitted as part of the SHL Intern Hiring Assessment 2025\.