"""Main grammar scoring pipeline."""
import numpy as np
from typing import Dict, Optional, Union
import time
import os

from ..audio.preprocessor import AudioPreprocessor
from ..audio.quality_checker import AudioQualityChecker
from ..asr.transcriber import Transcriber
from ..grammar.analyzer import GrammarAnalyzer
from ..grammar.error_detector import GrammarErrorDetector
from ..grammar.scorer import GrammarScorer
from ..utils.config import *


class GrammarScoringPipeline:
    """End-to-end pipeline for scoring grammar from voice samples."""
    
    def __init__(
        self,
        whisper_model: str = MODEL_SIZE,
        device: Optional[str] = DEVICE,
        language: str = LANGUAGE
    ):
        """
        Initialize the grammar scoring pipeline.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            device: Device to run models on (cuda/cpu)
            language: Language code for analysis
        """
        print("=" * 60)
        print("Initializing Grammar Scoring Pipeline")
        print("=" * 60)
        
        self.language = language
        
        # Initialize components
        print("\n[1/5] Initializing audio preprocessor...")
        self.audio_preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        
        print("[2/5] Initializing audio quality checker...")
        self.quality_checker = AudioQualityChecker()
        
        print("[3/5] Initializing transcriber (this may take a moment)...")
        self.transcriber = Transcriber(model_size=whisper_model, device=device)
        
        print("[4/5] Initializing grammar analyzer...")
        self.grammar_analyzer = GrammarAnalyzer()
        
        print("[5/5] Initializing error detector...")
        self.error_detector = GrammarErrorDetector(language=f"{language}-US")
        
        self.scorer = GrammarScorer(weights=WEIGHTS)
        
        print("\n" + "=" * 60)
        print("✓ Pipeline initialized successfully!")
        print("=" * 60 + "\n")
    
    def score_audio(
        self,
        audio_path: str,
        preprocess: bool = True,
        return_detailed: bool = True
    ) -> Dict:
        """
        Score grammar from audio file.
        
        Args:
            audio_path: Path to audio file
            preprocess: Whether to preprocess audio
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary with scores and analysis
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(audio_path)}")
        print(f"{'='*60}\n")
        
        # Step 1: Load and preprocess audio
        print("[1/5] Loading and preprocessing audio...")
        if preprocess:
            audio, sr = self.audio_preprocessor.preprocess(audio_path)
        else:
            audio, sr = self.audio_preprocessor.load_audio(audio_path)
        
        # Step 2: Check audio quality
        print("[2/5] Assessing audio quality...")
        quality_metrics = self.quality_checker.assess_quality(audio, sr)
        print(f"  ✓ Audio quality: {quality_metrics['overall_quality']:.2f}")
        print(f"  ✓ Duration: {quality_metrics['duration']:.2f}s")
        
        if not quality_metrics["is_acceptable"]:
            return {
                "success": False,
                "error": "Audio quality too low for reliable transcription",
                "quality_metrics": quality_metrics
            }
        
        # Step 3: Transcribe audio
        print("[3/5] Transcribing audio (this may take a moment)...")
        transcription = self.transcriber.transcribe_audio(audio, language=self.language)
        text = transcription["text"]
        print(f"  ✓ Transcription complete")
        print(f"  ✓ Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        
        if not text.strip():
            return {
                "success": False,
                "error": "No speech detected in audio",
                "quality_metrics": quality_metrics,
                "transcription": transcription
            }
        
        # Step 4: Analyze grammar
        print("[4/5] Analyzing grammar...")
        
        # Syntax analysis
        syntax_metrics = self.grammar_analyzer.analyze_syntax(text)
        print(f"  ✓ Syntax score: {syntax_metrics['syntax_score']:.2f}")
        
        # Error detection
        errors = self.error_detector.detect_errors(text)
        error_score = self.scorer.calculate_error_score(errors, len(text))
        print(f"  ✓ Errors found: {len(errors)}")
        
        # Fluency analysis
        fluency_metrics = self.grammar_analyzer.analyze_fluency(text)
        print(f"  ✓ Fluency score: {fluency_metrics['fluency_score']:.2f}")
        
        # Readability analysis
        readability_metrics = self.grammar_analyzer.analyze_readability(text)
        print(f"  ✓ Readability score: {readability_metrics['readability_score']:.2f}")
        
        # Step 5: Calculate final score
        print("[5/5] Calculating final score...")
        final_score = self.scorer.calculate_overall_score(
            syntax_score=syntax_metrics["syntax_score"],
            error_score=error_score,
            fluency_score=fluency_metrics["fluency_score"],
            readability_score=readability_metrics["readability_score"]
        )
        
        grade = self.scorer.get_grade_label(final_score)
        
        processing_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"  FINAL GRAMMAR SCORE: {final_score:.2f}/10.0 ({grade})")
        print(f"{'='*60}")
        print(f"Processing time: {processing_time:.2f}s\n")
        
        # Prepare result
        result = {
            "success": True,
            "grammar_score": round(final_score, 2),
            "grade": grade,
            "transcription": text,
            "confidence": round(transcription["confidence"], 3),
            "processing_time": round(processing_time, 2),
            "metrics": {
                "syntax_score": round(syntax_metrics["syntax_score"], 2),
                "error_score": round(error_score, 2),
                "fluency_score": round(fluency_metrics["fluency_score"], 2),
                "readability_score": round(readability_metrics["readability_score"], 2),
                "error_count": len(errors),
                "audio_quality": round(quality_metrics["overall_quality"], 3)
            }
        }
        
        # Add detailed information if requested
        if return_detailed:
            result["detailed_analysis"] = {
                "syntax_metrics": syntax_metrics,
                "fluency_metrics": fluency_metrics,
                "readability_metrics": readability_metrics,
                "quality_metrics": quality_metrics,
                "errors": errors,
                "error_breakdown": self.error_detector.count_errors_by_type(errors),
                "error_severity": self.error_detector.count_errors_by_severity(errors),
                "feedback": self.scorer.get_detailed_feedback(
                    syntax_metrics["syntax_score"],
                    error_score,
                    fluency_metrics["fluency_score"],
                    readability_metrics["readability_score"],
                    errors
                )
            }
        
        return result
    
    def score_text(self, text: str, return_detailed: bool = True) -> Dict:
        """
        Score grammar from text directly (skip audio processing).
        
        Args:
            text: Input text to analyze
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary with scores and analysis
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Analyzing text...")
        print(f"{'='*60}\n")
        
        if not text.strip():
            return {
                "success": False,
                "error": "Empty text provided"
            }
        
        # Analyze grammar
        print("[1/4] Analyzing syntax...")
        syntax_metrics = self.grammar_analyzer.analyze_syntax(text)
        
        print("[2/4] Detecting errors...")
        errors = self.error_detector.detect_errors(text)
        error_score = self.scorer.calculate_error_score(errors, len(text))
        
        print("[3/4] Analyzing fluency...")
        fluency_metrics = self.grammar_analyzer.analyze_fluency(text)
        
        print("[4/4] Analyzing readability...")
        readability_metrics = self.grammar_analyzer.analyze_readability(text)
        
        # Calculate final score
        final_score = self.scorer.calculate_overall_score(
            syntax_score=syntax_metrics["syntax_score"],
            error_score=error_score,
            fluency_score=fluency_metrics["fluency_score"],
            readability_score=readability_metrics["readability_score"]
        )
        
        grade = self.scorer.get_grade_label(final_score)
        processing_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"  FINAL GRAMMAR SCORE: {final_score:.2f}/10.0 ({grade})")
        print(f"{'='*60}\n")
        
        result = {
            "success": True,
            "grammar_score": round(final_score, 2),
            "grade": grade,
            "text": text,
            "processing_time": round(processing_time, 2),
            "metrics": {
                "syntax_score": round(syntax_metrics["syntax_score"], 2),
                "error_score": round(error_score, 2),
                "fluency_score": round(fluency_metrics["fluency_score"], 2),
                "readability_score": round(readability_metrics["readability_score"], 2),
                "error_count": len(errors)
            }
        }
        
        if return_detailed:
            result["detailed_analysis"] = {
                "syntax_metrics": syntax_metrics,
                "fluency_metrics": fluency_metrics,
                "readability_metrics": readability_metrics,
                "errors": errors,
                "error_breakdown": self.error_detector.count_errors_by_type(errors),
                "error_severity": self.error_detector.count_errors_by_severity(errors),
                "feedback": self.scorer.get_detailed_feedback(
                    syntax_metrics["syntax_score"],
                    error_score,
                    fluency_metrics["fluency_score"],
                    readability_metrics["readability_score"],
                    errors
                )
            }
        
        return result
