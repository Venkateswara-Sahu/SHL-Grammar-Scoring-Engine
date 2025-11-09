"""Batch evaluation script for Kaggle dataset."""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.grammar_scorer import GrammarScoringPipeline


def evaluate_dataset(audio_dir: str, output_dir: str, model_size: str = "base"):
    """
    Evaluate all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save results
        model_size: Whisper model size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"**/*{ext}"))
    
    print(f"\nFound {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found. Please add audio files to the directory.")
        return
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = GrammarScoringPipeline(whisper_model=model_size)
    
    # Process each file
    results = []
    errors = []
    
    print("\nProcessing audio files...\n")
    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            result = pipeline.score_audio(
                str(audio_path),
                preprocess=True,
                return_detailed=True
            )
            
            # Add filename
            result["filename"] = audio_path.name
            result["filepath"] = str(audio_path)
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing {audio_path.name}: {e}")
            errors.append({
                "filename": audio_path.name,
                "error": str(e)
            })
    
    # Save results
    print("\n\nSaving results...")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    # Convert results to JSON-serializable format
    results_serializable = convert_to_json_serializable(results)
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, "detailed_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved detailed results to: {json_path}")
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        if result.get("success", False):
            summary_data.append({
                "filename": result["filename"],
                "grammar_score": result["grammar_score"],
                "grade": result["grade"],
                "transcription": result["transcription"],
                "confidence": result["confidence"],
                "syntax_score": result["metrics"]["syntax_score"],
                "error_count": result["metrics"]["error_count"],
                "fluency_score": result["metrics"]["fluency_score"],
                "readability_score": result["metrics"]["readability_score"],
                "audio_quality": result["metrics"]["audio_quality"],
                "processing_time": result["processing_time"]
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = os.path.join(output_dir, "summary_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved summary CSV to: {csv_path}")
        
        # Generate statistics
        print("\n" + "="*60)
        print("EVALUATION STATISTICS")
        print("="*60)
        print(f"\nTotal files processed: {len(results)}")
        print(f"Successful: {len(summary_data)}")
        print(f"Errors: {len(errors)}")
        
        print(f"\n--- Grammar Scores ---")
        print(f"Mean: {df['grammar_score'].mean():.2f}")
        print(f"Median: {df['grammar_score'].median():.2f}")
        print(f"Std Dev: {df['grammar_score'].std():.2f}")
        print(f"Min: {df['grammar_score'].min():.2f}")
        print(f"Max: {df['grammar_score'].max():.2f}")
        
        print(f"\n--- Grade Distribution ---")
        grade_counts = df['grade'].value_counts()
        for grade, count in grade_counts.items():
            print(f"{grade}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n--- Component Scores (Average) ---")
        print(f"Syntax: {df['syntax_score'].mean():.2f}")
        print(f"Fluency: {df['fluency_score'].mean():.2f}")
        print(f"Readability: {df['readability_score'].mean():.2f}")
        
        print(f"\n--- Audio Quality ---")
        print(f"Mean Quality: {df['audio_quality'].mean():.2f}")
        print(f"Mean Confidence: {df['confidence'].mean():.2f}")
        
        print(f"\n--- Performance ---")
        print(f"Avg Processing Time: {df['processing_time'].mean():.2f}s")
        print(f"Total Processing Time: {df['processing_time'].sum():.2f}s")
        
        # Save statistics
        stats = {
            "total_files": len(results),
            "successful": len(summary_data),
            "errors": len(errors),
            "grammar_scores": {
                "mean": float(df['grammar_score'].mean()),
                "median": float(df['grammar_score'].median()),
                "std": float(df['grammar_score'].std()),
                "min": float(df['grammar_score'].min()),
                "max": float(df['grammar_score'].max())
            },
            "grade_distribution": grade_counts.to_dict(),
            "component_scores": {
                "syntax": float(df['syntax_score'].mean()),
                "fluency": float(df['fluency_score'].mean()),
                "readability": float(df['readability_score'].mean())
            },
            "audio_metrics": {
                "mean_quality": float(df['audio_quality'].mean()),
                "mean_confidence": float(df['confidence'].mean())
            },
            "performance": {
                "avg_processing_time": float(df['processing_time'].mean()),
                "total_processing_time": float(df['processing_time'].sum())
            }
        }
        
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Saved statistics to: {stats_path}")
    
    # Save errors if any
    if errors:
        errors_path = os.path.join(output_dir, "errors.json")
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Saved errors to: {errors_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate grammar from audio files in batch"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/raw",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Grammar Scoring Engine - Batch Evaluation           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    evaluate_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        model_size=args.model_size
    )
