"""Grammar scoring algorithms."""
from typing import Dict, List
import numpy as np


class GrammarScorer:
    """Calculates overall grammar scores."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize grammar scorer.
        
        Args:
            weights: Weights for different scoring components
        """
        self.weights = weights or {
            "syntax": 0.30,
            "grammar_errors": 0.30,
            "fluency": 0.20,
            "readability": 0.20
        }
    
    def calculate_error_score(self, errors: List[Dict], text_length: int) -> float:
        """
        Calculate score based on grammar errors.
        
        Args:
            errors: List of detected errors
            text_length: Length of text (in characters)
            
        Returns:
            Error score (0-10, higher is better)
        """
        if text_length == 0:
            return 0.0
        
        # Count errors by severity
        severity_weights = {"high": 3.0, "medium": 1.5, "low": 0.5}
        weighted_errors = sum(
            severity_weights.get(error["severity"], 1.0) 
            for error in errors
        )
        
        # Calculate error density (errors per 100 characters)
        error_density = (weighted_errors / text_length) * 100
        
        # Convert to 0-10 score (lower density = higher score)
        # Assume 0 errors = 10, 5+ errors per 100 chars = 0
        score = max(0, 10 - (error_density * 2))
        
        return score
    
    def calculate_overall_score(
        self,
        syntax_score: float,
        error_score: float,
        fluency_score: float,
        readability_score: float
    ) -> float:
        """
        Calculate weighted overall grammar score.
        
        Args:
            syntax_score: Syntax correctness score (0-10)
            error_score: Error-based score (0-10)
            fluency_score: Fluency score (0-10)
            readability_score: Readability score (0-10)
            
        Returns:
            Overall grammar score (0-10)
        """
        overall = (
            syntax_score * self.weights["syntax"] +
            error_score * self.weights["grammar_errors"] +
            fluency_score * self.weights["fluency"] +
            readability_score * self.weights["readability"]
        )
        
        return min(max(overall, 0.0), 10.0)
    
    def get_grade_label(self, score: float) -> str:
        """
        Convert numeric score to grade label.
        
        Args:
            score: Numeric score (0-10)
            
        Returns:
            Grade label
        """
        if score >= 9.0:
            return "Excellent"
        elif score >= 8.0:
            return "Very Good"
        elif score >= 7.0:
            return "Good"
        elif score >= 6.0:
            return "Satisfactory"
        elif score >= 5.0:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def get_detailed_feedback(
        self,
        syntax_score: float,
        error_score: float,
        fluency_score: float,
        readability_score: float,
        errors: List[Dict]
    ) -> Dict[str, str]:
        """
        Generate detailed feedback based on scores.
        
        Args:
            syntax_score: Syntax score
            error_score: Error score
            fluency_score: Fluency score
            readability_score: Readability score
            errors: List of detected errors
            
        Returns:
            Dictionary with feedback for each dimension
        """
        feedback = {}
        
        # Syntax feedback
        if syntax_score >= 8.0:
            feedback["syntax"] = "Excellent sentence structure with proper grammatical construction."
        elif syntax_score >= 6.0:
            feedback["syntax"] = "Good sentence structure with minor issues."
        else:
            feedback["syntax"] = "Sentence structure needs improvement. Consider using complete sentences."
        
        # Error feedback
        if error_score >= 8.0:
            feedback["errors"] = "Very few grammatical errors detected."
        elif error_score >= 6.0:
            feedback["errors"] = "Some grammatical errors present. Review the specific errors for improvement."
        else:
            feedback["errors"] = f"Multiple grammatical errors detected ({len(errors)} errors). Focus on basic grammar rules."
        
        # Fluency feedback
        if fluency_score >= 8.0:
            feedback["fluency"] = "Excellent fluency and natural language flow."
        elif fluency_score >= 6.0:
            feedback["fluency"] = "Good fluency with some room for improvement."
        else:
            feedback["fluency"] = "Fluency needs work. Consider reducing filler words and varying sentence structure."
        
        # Readability feedback
        if readability_score >= 8.0:
            feedback["readability"] = "Highly readable and clear communication."
        elif readability_score >= 6.0:
            feedback["readability"] = "Reasonably readable with moderate complexity."
        else:
            feedback["readability"] = "Text complexity is high. Consider simplifying sentence structure and vocabulary."
        
        return feedback
