"""Example usage of the Grammar Scoring Pipeline."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.grammar_scorer import GrammarScoringPipeline


def example_text_scoring():
    """Example: Score grammar from text."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Scoring Text")
    print("="*60)
    
    # Initialize pipeline
    pipeline = GrammarScoringPipeline()
    
    # Example texts with varying grammar quality
    texts = [
        "Hello, my name is John and I am very excited to be here today. I have been working in the field of artificial intelligence for over five years.",
        "me go store yesterday buy apple they was good but expensive",
        "The quick brown fox jumps over the lazy dog. This sentence demonstrates perfect grammar and proper structure."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Input: \"{text}\"")
        
        result = pipeline.score_text(text, return_detailed=False)
        
        if result["success"]:
            print(f"\nScore: {result['grammar_score']}/10.0 ({result['grade']})")
            print(f"Errors: {result['metrics']['error_count']}")


def example_audio_scoring():
    """Example: Score grammar from audio file."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Scoring Audio")
    print("="*60)
    
    # Check if sample audio exists
    audio_path = "data/sample_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"\n⚠️  No sample audio found at: {audio_path}")
        print("Please add a sample audio file to test audio scoring.")
        return
    
    # Initialize pipeline
    pipeline = GrammarScoringPipeline()
    
    # Score audio
    result = pipeline.score_audio(audio_path, return_detailed=True)
    
    if result["success"]:
        print(f"\nResults:")
        print(f"  Transcription: \"{result['transcription']}\"")
        print(f"  Grammar Score: {result['grammar_score']}/10.0 ({result['grade']})")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Processing Time: {result['processing_time']}s")
        
        if "detailed_analysis" in result:
            print(f"\n  Detailed Metrics:")
            print(f"    - Syntax: {result['metrics']['syntax_score']}")
            print(f"    - Errors: {result['metrics']['error_count']}")
            print(f"    - Fluency: {result['metrics']['fluency_score']}")
            print(f"    - Readability: {result['metrics']['readability_score']}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Grammar Scoring Engine - Example Usage              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    example_text_scoring()
    
    # Uncomment to test audio scoring (requires sample audio file)
    # example_audio_scoring()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
