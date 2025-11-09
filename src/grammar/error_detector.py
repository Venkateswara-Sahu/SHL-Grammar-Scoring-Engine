"""Grammar error detection."""
from typing import List, Dict
import re

# Try to import LanguageTool, but don't fail if it's not available
try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False


class GrammarErrorDetector:
    """Detects grammar errors in text."""
    
    def __init__(self, language: str = "en-US"):
        """
        Initialize grammar error detector.
        
        Args:
            language: Language code for LanguageTool
        """
        self.tool = None
        
        if LANGUAGETOOL_AVAILABLE:
            print("Initializing LanguageTool...")
            try:
                # Try to initialize LanguageTool with timeout settings
                self.tool = language_tool_python.LanguageTool(language, config={'maxSpellingSuggestions': 3})
                print("LanguageTool ready!")
            except Exception as e:
                print(f"Note: LanguageTool initialization skipped (network issue)")
                print("Using spaCy-based error detection...")
                self.tool = None
        else:
            print("Note: LanguageTool not available")
            print("Using spaCy-based error detection...")
    
    def detect_errors(self, text: str) -> List[Dict]:
        """
        Detect grammar errors in text.
        
        Args:
            text: Input text to check
            
        Returns:
            List of detected errors with details
        """
        if self.tool is None:
            # Fallback: basic error detection without LanguageTool
            return self._basic_error_detection(text)
        
        try:
            matches = self.tool.check(text)
        except Exception as e:
            print(f"Warning: LanguageTool check failed: {e}")
            return self._basic_error_detection(text)
        
        errors = []
        for match in matches:
            error = {
                "type": self._categorize_error(match.ruleId),
                "rule_id": match.ruleId,
                "message": match.message,
                "context": match.context,
                "offset": match.offset,
                "length": match.errorLength,
                "suggestions": match.replacements[:3],  # Top 3 suggestions
                "severity": self._get_severity(match)
            }
            errors.append(error)
        
        return errors
    
    def _categorize_error(self, rule_id: str) -> str:
        """
        Categorize error by rule ID.
        
        Args:
            rule_id: LanguageTool rule ID
            
        Returns:
            Error category
        """
        rule_id_lower = rule_id.lower()
        
        if any(x in rule_id_lower for x in ["spelling", "speller"]):
            return "spelling"
        elif any(x in rule_id_lower for x in ["grammar", "agreement"]):
            return "grammar"
        elif any(x in rule_id_lower for x in ["punctuation", "comma"]):
            return "punctuation"
        elif any(x in rule_id_lower for x in ["style", "redundancy"]):
            return "style"
        elif any(x in rule_id_lower for x in ["typography", "whitespace"]):
            return "typography"
        else:
            return "other"
    
    def _get_severity(self, match) -> str:
        """
        Get error severity.
        
        Args:
            match: LanguageTool match object
            
        Returns:
            Severity level
        """
        # Map LanguageTool's issue types to severity
        issue_type = str(match.ruleIssueType).lower() if hasattr(match, 'ruleIssueType') else ""
        
        if "misspelling" in issue_type or "grammar" in issue_type:
            return "high"
        elif "style" in issue_type or "typographical" in issue_type:
            return "medium"
        else:
            return "low"
    
    def count_errors_by_type(self, errors: List[Dict]) -> Dict[str, int]:
        """
        Count errors by type.
        
        Args:
            errors: List of detected errors
            
        Returns:
            Dictionary with error counts by type
        """
        counts = {}
        for error in errors:
            error_type = error["type"]
            counts[error_type] = counts.get(error_type, 0) + 1
        
        return counts
    
    def count_errors_by_severity(self, errors: List[Dict]) -> Dict[str, int]:
        """
        Count errors by severity.
        
        Args:
            errors: List of detected errors
            
        Returns:
            Dictionary with error counts by severity
        """
        counts = {"high": 0, "medium": 0, "low": 0}
        for error in errors:
            severity = error["severity"]
            counts[severity] = counts.get(severity, 0) + 1
        
        return counts
    
    def _basic_error_detection(self, text: str) -> List[Dict]:
        """
        Basic error detection without LanguageTool (fallback using spaCy).
        
        Args:
            text: Input text
            
        Returns:
            List of basic errors detected
        """
        errors = []
        
        # Note: Since we have comprehensive spaCy-based syntax analysis
        # in GrammarAnalyzer, this fallback focuses on simple pattern matching
        
        # Check for common errors
        patterns = [
            (r'\bi\b', "Pronoun 'I' should be capitalized", "grammar"),
            (r'\s{2,}', "Multiple spaces detected", "typography"),
            (r'[.!?]{2,}', "Multiple punctuation marks", "punctuation"),
        ]
        
        for pattern, message, error_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                errors.append({
                    "type": error_type,
                    "rule_id": f"BASIC_{error_type.upper()}",
                    "message": message,
                    "context": text[max(0, match.start()-20):min(len(text), match.end()+20)],
                    "offset": match.start(),
                    "length": match.end() - match.start(),
                    "suggestions": [],
                    "severity": "medium"
                })
        
        return errors
    
    def __del__(self):
        """Cleanup LanguageTool instance."""
        if hasattr(self, 'tool') and self.tool is not None:
            try:
                self.tool.close()
            except:
                pass
