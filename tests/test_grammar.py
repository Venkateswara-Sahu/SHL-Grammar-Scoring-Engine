"""Tests for grammar analysis modules."""
import pytest
from src.grammar.analyzer import GrammarAnalyzer
from src.grammar.error_detector import GrammarErrorDetector
from src.grammar.scorer import GrammarScorer


def test_grammar_analyzer_initialization():
    """Test grammar analyzer initialization."""
    analyzer = GrammarAnalyzer()
    assert analyzer.nlp is not None


def test_analyze_syntax():
    """Test syntax analysis."""
    analyzer = GrammarAnalyzer()
    text = "This is a simple sentence."
    
    metrics = analyzer.analyze_syntax(text)
    
    assert "num_sentences" in metrics
    assert "num_tokens" in metrics
    assert "syntax_score" in metrics
    assert metrics["num_sentences"] >= 1


def test_analyze_fluency():
    """Test fluency analysis."""
    analyzer = GrammarAnalyzer()
    text = "The quick brown fox jumps over the lazy dog. This is another sentence."
    
    metrics = analyzer.analyze_fluency(text)
    
    assert "lexical_diversity" in metrics
    assert "fluency_score" in metrics
    assert 0 <= metrics["fluency_score"] <= 10


def test_analyze_readability():
    """Test readability analysis."""
    analyzer = GrammarAnalyzer()
    text = "This is a simple sentence. It is easy to read."
    
    metrics = analyzer.analyze_readability(text)
    
    assert "flesch_reading_ease" in metrics
    assert "readability_score" in metrics


def test_error_detector_initialization():
    """Test error detector initialization."""
    detector = GrammarErrorDetector()
    assert detector.tool is not None


def test_detect_errors():
    """Test error detection."""
    detector = GrammarErrorDetector()
    
    # Text with intentional errors
    text = "He go to the store yesterday."
    
    errors = detector.detect_errors(text)
    
    assert isinstance(errors, list)
    assert len(errors) > 0  # Should detect at least one error
    
    # Check error structure
    if len(errors) > 0:
        error = errors[0]
        assert "type" in error
        assert "message" in error


def test_count_errors_by_type():
    """Test error counting by type."""
    detector = GrammarErrorDetector()
    
    errors = [
        {"type": "grammar", "severity": "high"},
        {"type": "grammar", "severity": "medium"},
        {"type": "spelling", "severity": "low"}
    ]
    
    counts = detector.count_errors_by_type(errors)
    
    assert counts["grammar"] == 2
    assert counts["spelling"] == 1


def test_grammar_scorer_initialization():
    """Test scorer initialization."""
    scorer = GrammarScorer()
    assert scorer.weights is not None


def test_calculate_error_score():
    """Test error score calculation."""
    scorer = GrammarScorer()
    
    errors = [
        {"severity": "high"},
        {"severity": "medium"}
    ]
    
    score = scorer.calculate_error_score(errors, 100)
    
    assert 0 <= score <= 10


def test_calculate_overall_score():
    """Test overall score calculation."""
    scorer = GrammarScorer()
    
    score = scorer.calculate_overall_score(
        syntax_score=8.0,
        error_score=7.5,
        fluency_score=8.5,
        readability_score=7.0
    )
    
    assert 0 <= score <= 10


def test_get_grade_label():
    """Test grade label generation."""
    scorer = GrammarScorer()
    
    assert scorer.get_grade_label(9.5) == "Excellent"
    assert scorer.get_grade_label(8.5) == "Very Good"
    assert scorer.get_grade_label(7.5) == "Good"
    assert scorer.get_grade_label(4.0) == "Needs Improvement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
