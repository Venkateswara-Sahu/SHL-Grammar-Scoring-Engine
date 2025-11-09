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

### **Final Kaggle Score: 0.895 RMSE** 🏆

- **Kaggle Test RMSE**: **0.895** (21% improvement over baseline)
- **Model**: Random Forest with Audio Features + Grammar Analysis
- **Training Samples**: 402/409 processed (98.3% success rate)
- **Test Samples**: 197 predictions generated
- **Training RMSE**: 0.76
- **Cross-Validation RMSE**: ~0.85
- **Pearson Correlation**: Improved with feature engineering
- **Average Processing Time**: ~6-8 seconds per audio file

## **🌟 Key Features**

* **State-of-the-Art ASR**: OpenAI Whisper (base model) for accurate transcription.  
* **Advanced Audio Feature Engineering**: 
  * Acoustic features (pitch, energy, MFCCs)
  * Spectral features (centroid, rolloff, bandwidth)
  * Voice quality metrics (zero-crossing rate, chroma)
  * Speaking rate and duration analysis
* **Multi-Dimensional Grammar Analysis**:  
  * Syntax correctness (30% weight) \- spaCy dependency parsing  
  * Grammar error detection (30% weight) \- LanguageTool  
  * Fluency metrics (20% weight) \- Speech pattern analysis  
  * Readability scores (20% weight) \- Flesch-Kincaid metrics  
* **Machine Learning Model**: Random Forest Regressor with 100+ trees for robust predictions
* **Intelligent Calibration**: Linear regression to map raw scores to 1-5 MOS scale.  
* **Audio Quality Assessment**: Pre-transcription quality checks.  
* **Robust Error Handling**: Fallback mechanisms for failed transcriptions.  
* **Comprehensive Jupyter Notebook**: Complete analysis with visualizations.

## **🏗️ Improved Pipeline Architecture**

### **Enhanced Model (0.895 RMSE)**

Audio Input (WAV)  
↓  
**[Audio Feature Extraction]**  
├── Acoustic Features (pitch, energy, MFCCs)  
├── Spectral Features (centroid, rolloff, bandwidth)  
├── Voice Quality (zero-crossing rate, chroma)  
└── Duration & Speaking Rate  
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
**[Feature Combination]**  
Audio Features + Grammar Scores + Confidence  
↓  
**Random Forest Regressor (100 estimators)**  
↓  
Final Score (1-5 MOS Likert Scale)

**Key Improvement**: Combining audio features with grammar analysis improved RMSE from 1.138 to 0.895

## **🛠️ Technologies & Libraries**

| Category | Technology | Purpose |
| :---- | :---- | :---- |
| ASR | OpenAI Whisper | State-of-the-art speech recognition |
| NLP | spaCy 3.8 | Syntax analysis, dependency parsing |
| Grammar | LanguageTool | Grammar error detection |
| Audio | librosa 0.10 | Audio processing, feature extraction |
| Audio | noisereduce | Noise reduction |
| ML | scikit-learn | Random Forest, StandardScaler, metrics |
| ML | Random Forest | Main prediction model (100 estimators) |
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

* **improved_model.py**: 🏆 **Best Model** - Random Forest with audio features + grammar (0.895 RMSE)
* **submission_improved.csv**: ⭐ **Best Kaggle Submission** (197 test samples, 0.895 RMSE)
* Grammar\_Scoring\_Engine.ipynb: ⭐ **Main Jupyter Notebook** with complete analysis, training, evaluation
* submission.csv: Baseline submission (197 test samples, 1.138 RMSE)
* data/results/training\_metrics.json: Performance metrics (RMSE, Pearson)
* data/results/training\_predictions.csv: Detailed results from the training set

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

### **Improved Model Architecture (0.895 RMSE)**

Our best-performing model combines audio features with grammar analysis using a Random Forest ensemble:

### **1\. Audio Feature Extraction**

Extract comprehensive acoustic features from audio:

* **Acoustic Features**:
  * Pitch statistics (mean, std via librosa.piptrack)
  * Energy/RMS (root mean square energy)
  * Zero-crossing rate (voice clarity indicator)
* **Spectral Features**:
  * Spectral centroid (brightness of sound)
  * Spectral rolloff (frequency below which 85% of energy is contained)
  * Spectral bandwidth
* **MFCCs (Mel-Frequency Cepstral Coefficients)**:
  * 13 MFCC coefficients capturing voice characteristics
  * Mean and standard deviation for each coefficient
* **Duration & Tempo**: Audio length and speaking rate

### **2\. Audio Preprocessing**

* **Noise Reduction**: Spectral gating using noisereduce.  
* **Normalization**: RMS-based audio level normalization.  
* **Quality Assessment**: SNR calculation and clarity metrics.  

### **3\. Speech-to-Text (ASR)**

* **Model**: OpenAI Whisper Base (74M parameters).  
* **Configuration**: English language, CPU inference.  
* **Output**: Transcribed text with confidence scores.  
* **Average Time**: \~4-5 seconds per audio file.

### **4\. Multi-Dimensional Grammar Analysis**

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

### **5\. Machine Learning Model**

* **Algorithm**: Random Forest Regressor (100 estimators, max_depth=10)
* **Features**: 
  - Audio features (21 dimensions)
  - Grammar scores (6 dimensions)
  - Transcription metadata (2 dimensions)
* **Training**: Cross-validation with 5 folds
* **Output**: Direct prediction on 1-5 scale

### **6\. Evaluation Metrics**

**Improved Model (Random Forest + Audio Features)**:
* **Kaggle Test RMSE**: **0.895** ✅
* **Cross-Validation RMSE**: ~0.85
* **Training RMSE**: 0.76
* **Processing Success Rate**: 98.3% (402/409 samples)

**Baseline Model (Grammar Only)**:
* **Kaggle Test RMSE**: 1.138
* **Pearson Correlation**: 0.084
* **MAE**: 0.55

**Key Insight**: Adding audio features improved RMSE by 21% (1.138 → 0.895)

## **📈 Performance Metrics**

### **Model Comparison**

| Model | Kaggle Test RMSE | Improvement |
| :---- | :---- | :---- |
| **Improved Model (RF + Audio)** | **0.895** | **Best** ✅ |
| Baseline (Grammar Only) | 1.138 | -21% worse |
| Median Baseline | 1.179 | -24% worse |
| Mean Baseline | 1.213 | -26% worse |

### **Detailed Metrics**

| Metric | Value |
| :---- | :---- |
| **Kaggle Test RMSE** | **0.895** 🏆 |
| Training RMSE | 0.76 |
| Cross-Validation RMSE | ~0.85 |
| Training Success Rate | 98.3% (402/409) |
| Test Predictions | 197 samples |
| Average Processing Time | ~6-8 seconds per audio file |
| Feature Dimensions | 29 (21 audio + 6 grammar + 2 metadata) |
| Mean Predicted Score | \~2.5-3.0 (1-5 scale) |

## **🎓 Key Insights**

* **Audio features are crucial**: Adding acoustic features improved RMSE by 21% (1.138 → 0.895)
* **Multi-dimensional analysis** works better than single-metric scoring.  
* **Random Forest ensemble** captures complex relationships between features better than linear models
* **Feature engineering matters**: Combining 29 features (audio + grammar + metadata) yields better predictions
* **Audio quality** significantly impacts transcription accuracy.  
* **Whisper base model** provides a good balance of speed and accuracy.  
* **Fallback mechanisms** improve robustness (e.g., spaCy when LanguageTool fails).

## **🚧 Future Improvements**

* Fine-tune Whisper on domain-specific pronunciation data
* Add prosodic features (intonation, stress patterns, pauses)
* Ensemble methods combining Random Forest with XGBoost/LightGBM
* Deep learning models (CNN/LSTM) on raw audio spectrograms
* Speaker normalization for accent/dialect variations
* GPU acceleration for faster processing
* Hyperparameter tuning with Bayesian optimization

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

This project is submitted as part of the SHL Intern Hiring Assessment 2025.

---

## **🏆 Final Results Summary**

* **Best Kaggle Score**: 0.895 RMSE
* **Model**: Random Forest + Audio Features + Grammar Analysis
* **Improvement**: 21% better than grammar-only baseline (1.138 → 0.895)
* **GitHub**: https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine
* **Submission File**: `submission_improved.csv`

*Built with ❤️ for SHL Intern Hiring Assessment 2025*
