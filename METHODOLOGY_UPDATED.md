# Methodology Summary for SHL Submission Form

## Updated Methodology (100 words) - Improved Model (0.895 RMSE)

Our best approach combines comprehensive audio feature extraction with multi-dimensional grammar analysis using a Random Forest ensemble. We extract 21 acoustic features (pitch, energy, MFCCs, spectral characteristics) alongside transcription-based grammar metrics from Whisper ASR and spaCy NLP. The pipeline processes audio through noise reduction and quality checks, then generates features from both acoustic properties and linguistic content. A Random Forest Regressor (100 estimators) trained on these 29 combined features directly predicts MOS Likert scores (1-5). This hybrid approach achieved 0.895 RMSE on the Kaggle test set, representing a 21% improvement over grammar-only baseline (1.138 RMSE).

---

## Key Points for Submission:

- **Best Score**: 0.895 RMSE (Kaggle test set)
- **Model**: Random Forest Regressor with audio + grammar features
- **GitHub**: https://github.com/Venkateswara-Sahu/SHL-Grammar-Scoring-Engine
- **Key Innovation**: Combining acoustic features with NLP analysis
- **Improvement**: 21% better than grammar-only approach
