"""
Improved Grammar Scoring Model with Audio Features
Combines acoustic features with transcription-based analysis
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from src.pipeline.grammar_scorer import GrammarScoringPipeline

class ImprovedGrammarScorer:
    """
    Enhanced scoring model that combines:
    - Audio features (pitch, energy, speaking rate)
    - Transcription quality
    - Grammar analysis
    """
    
    def __init__(self):
        self.pipeline = GrammarScoringPipeline()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
    def extract_audio_features(self, audio_path):
        """Extract acoustic features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 1. Pitch features (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # 2. Energy features
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # 3. Zero crossing rate (voice quality)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # 4. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroid)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_mean = np.mean(spectral_rolloff)
            
            # 5. MFCCs (voice characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # 6. Duration and speaking rate
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Combine features
            features = {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'zcr_mean': zcr_mean,
                'spectral_mean': spectral_mean,
                'rolloff_mean': rolloff_mean,
                'duration': duration,
            }
            
            # Add MFCC features
            for i, mfcc_val in enumerate(mfcc_means):
                features[f'mfcc_{i}'] = mfcc_val
                
            return features
            
        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            # Return default features
            return {f: 0.0 for f in ['pitch_mean', 'pitch_std', 'energy_mean', 
                                     'energy_std', 'zcr_mean', 'spectral_mean',
                                     'rolloff_mean', 'duration'] + 
                   [f'mfcc_{i}' for i in range(13)]}
    
    def score_sample(self, audio_path):
        """Score a single audio sample using pipeline"""
        try:
            result = self.pipeline.score_audio(audio_path, preprocess=True)
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio_path)
            
            # Combine pipeline scores with audio features
            features = {
                'grammar_score': result.get('grammar_score', 5.0),
                'confidence': result.get('confidence', 0.5),
                'syntax_score': result.get('metrics', {}).get('syntax_score', 5.0),
                'error_count': result.get('metrics', {}).get('error_count', 5),
                'fluency_score': result.get('metrics', {}).get('fluency_score', 5.0),
                'readability_score': result.get('metrics', {}).get('readability_score', 5.0),
                'transcription_length': len(result.get('transcription', '')),
                **audio_features  # Add all audio features
            }
            
            return features, result.get('transcription', '')
            
        except Exception as e:
            print(f"Error scoring {audio_path}: {e}")
            # Return default features
            default_audio = self.extract_audio_features(audio_path)
            return {
                'grammar_score': 3.0,
                'confidence': 0.3,
                'syntax_score': 3.0,
                'error_count': 10,
                'fluency_score': 3.0,
                'readability_score': 3.0,
                'transcription_length': 0,
                **default_audio
            }, ""
    
    def train(self, train_dir, labels_df):
        """Train the improved model"""
        print("Training improved model...")
        
        X_train = []
        y_train = []
        successful = 0
        failed = 0
        
        for idx, row in labels_df.iterrows():
            filename = row['filename']
            # Add .wav extension if not present
            if not filename.endswith('.wav'):
                filename = filename + '.wav'
            true_label = row['label']
            audio_path = os.path.join(train_dir, filename)
            
            if not os.path.exists(audio_path):
                print(f"File not found: {filename}")
                failed += 1
                continue
            
            print(f"Processing {idx+1}/{len(labels_df)}: {filename}", end='\r')
            
            features, _ = self.score_sample(audio_path)
            
            if features:
                # Convert to feature vector
                feature_vector = [features[k] for k in sorted(features.keys())]
                X_train.append(feature_vector)
                y_train.append(true_label)
                successful += 1
            else:
                failed += 1
        
        print(f"\nProcessed: {successful} successful, {failed} failed")
        
        # Train model
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                     cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Training predictions
        y_pred = self.model.predict(X_train)
        y_pred = np.clip(y_pred, 1, 5)  # Clip to valid range
        
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        pearson = pearsonr(y_train, y_pred)[0]
        
        print(f"\n=== Training Results ===")
        print(f"Training RMSE: {rmse:.4f}")
        print(f"Cross-Validation RMSE: {cv_rmse:.4f}")
        print(f"Pearson Correlation: {pearson:.4f}")
        
        # Feature importance
        feature_names = sorted(features.keys())
        importances = self.model.feature_importances_
        top_features = sorted(zip(feature_names, importances), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        print("\n=== Top 10 Most Important Features ===")
        for feat, imp in top_features:
            print(f"{feat}: {imp:.4f}")
        
        return {
            'training_rmse': rmse,
            'cv_rmse': cv_rmse,
            'pearson': pearson,
            'samples_processed': successful,
            'feature_names': feature_names
        }
    
    def predict(self, test_dir, test_filenames):
        """Make predictions on test set"""
        print("\nGenerating predictions...")
        
        predictions = []
        transcriptions = []
        
        for idx, filename in enumerate(test_filenames):
            # Add .wav extension if not present
            if not filename.endswith('.wav'):
                filename = filename + '.wav'
            audio_path = os.path.join(test_dir, filename)
            
            if not os.path.exists(audio_path):
                print(f"Test file not found: {filename}")
                predictions.append(3.0)  # Default prediction
                transcriptions.append("")
                continue
            
            print(f"Predicting {idx+1}/{len(test_filenames)}: {filename}", end='\r')
            
            features, transcription = self.score_sample(audio_path)
            feature_vector = [features[k] for k in sorted(features.keys())]
            
            pred = self.model.predict([feature_vector])[0]
            pred = np.clip(pred, 1, 5)  # Clip to valid range
            
            predictions.append(pred)
            transcriptions.append(transcription)
        
        print(f"\nCompleted predictions for {len(predictions)} samples")
        
        return predictions, transcriptions


def main():
    """Main training and prediction pipeline"""
    
    print("="*60)
    print("IMPROVED GRAMMAR SCORING MODEL")
    print("Audio Features + Grammar Analysis")
    print("="*60)
    
    # Paths
    train_dir = 'data/raw/train'
    test_dir = 'data/raw/test'
    train_labels_path = 'data/raw/train.csv'
    test_labels_path = 'data/raw/test.csv'
    
    # Load data
    train_df = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(test_labels_path)
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize model
    scorer = ImprovedGrammarScorer()
    
    # Train
    metrics = scorer.train(train_dir, train_df)
    
    # Predict on test set
    predictions, transcriptions = scorer.predict(test_dir, test_df['filename'].tolist())
    
    # Create submission
    submission_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': predictions
    })
    
    output_path = 'submission_improved.csv'
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission saved to: {output_path}")
    print(f"Mean prediction: {np.mean(predictions):.4f}")
    print(f"Std prediction: {np.std(predictions):.4f}")
    print(f"Min prediction: {np.min(predictions):.4f}")
    print(f"Max prediction: {np.max(predictions):.4f}")
    
    # Also save detailed results
    detailed_df = submission_df.copy()
    detailed_df['transcription'] = transcriptions
    detailed_df.to_csv('submission_improved_detailed.csv', index=False)
    
    print("\n✅ Detailed results saved to: submission_improved_detailed.csv")
    print("\nNext steps:")
    print("1. Upload 'submission_improved.csv' to Kaggle")
    print("2. Compare with your previous score (1.138)")
    print("3. If better, update GitHub and resubmit form!")


if __name__ == "__main__":
    main()
