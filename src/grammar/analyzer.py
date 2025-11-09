"""Grammar and text analysis."""
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, List
import textstat
import warnings
warnings.filterwarnings('ignore')


class GrammarAnalyzer:
    """Analyzes grammatical structure and text quality."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize grammar analyzer.
        
        Args:
            spacy_model: spaCy model to load
        """
        print(f"Loading spaCy model: {spacy_model}...")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Model {spacy_model} not found. Downloading...")
            import os
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        print("spaCy model loaded!")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def analyze_syntax(self, text: str) -> Dict:
        """
        Analyze syntactic structure of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with syntax metrics
        """
        doc = self.nlp(text)
        
        # Count different syntactic elements
        sentences = list(doc.sents)
        
        metrics = {
            "num_sentences": len(sentences),
            "num_tokens": len(doc),
            "num_words": len([token for token in doc if not token.is_punct]),
            "avg_sentence_length": len(doc) / max(len(sentences), 1),
            "num_clauses": self._count_clauses(doc),
            "num_complex_sentences": self._count_complex_sentences(sentences),
            "syntax_score": self._calculate_syntax_score(doc, sentences)
        }
        
        return metrics
    
    def _count_clauses(self, doc) -> int:
        """Count number of clauses (approximation based on verbs)."""
        return len([token for token in doc if token.pos_ == "VERB"])
    
    def _count_complex_sentences(self, sentences) -> int:
        """Count sentences with subordinate clauses."""
        complex_count = 0
        for sent in sentences:
            # Look for subordinating conjunctions
            if any(token.dep_ in ["mark", "advcl", "ccomp"] for token in sent):
                complex_count += 1
        return complex_count
    
    def _calculate_syntax_score(self, doc, sentences) -> float:
        """
        Calculate overall syntax correctness score.
        
        Args:
            doc: spaCy Doc object
            sentences: List of sentences
            
        Returns:
            Syntax score (0-10)
        """
        if len(doc) == 0:
            return 0.0
        
        # Check for basic syntactic requirements
        has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        
        # Calculate sentence structure score
        valid_sentences = sum(1 for sent in sentences if self._is_valid_sentence(sent))
        sentence_score = valid_sentences / max(len(sentences), 1)
        
        # Base score
        base_score = 5.0
        
        # Add points for good structure
        if has_subject:
            base_score += 1.5
        if has_verb:
            base_score += 1.5
        
        # Add sentence structure score
        base_score += sentence_score * 2.0
        
        return min(base_score, 10.0)
    
    def _is_valid_sentence(self, sent) -> bool:
        """Check if sentence has basic valid structure."""
        has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in sent)
        has_verb = any(token.pos_ == "VERB" for token in sent)
        return has_subject and has_verb
    
    def analyze_fluency(self, text: str) -> Dict:
        """
        Analyze text fluency.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with fluency metrics
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Calculate various fluency indicators
        words = [token.text for token in doc if not token.is_punct]
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(len(words), 1)
        
        # Check for filler words
        filler_words = ["um", "uh", "like", "you know", "i mean", "actually", "basically"]
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_ratio = filler_count / max(len(words), 1)
        
        # Sentence variation (coefficient of variation in sentence length)
        if len(sentences) > 1:
            sent_lengths = [len(list(sent)) for sent in sentences]
            import numpy as np
            sent_variation = np.std(sent_lengths) / max(np.mean(sent_lengths), 1)
        else:
            sent_variation = 0.0
        
        # Calculate fluency score
        fluency_score = self._calculate_fluency_score(ttr, filler_ratio, sent_variation)
        
        return {
            "lexical_diversity": ttr,
            "filler_word_count": filler_count,
            "filler_ratio": filler_ratio,
            "sentence_variation": sent_variation,
            "fluency_score": fluency_score
        }
    
    def _calculate_fluency_score(self, ttr: float, filler_ratio: float, 
                                  sent_variation: float) -> float:
        """Calculate overall fluency score."""
        # Higher TTR is better (but very high might indicate lack of repetition for clarity)
        ttr_score = min(ttr * 10, 10.0)
        
        # Lower filler ratio is better
        filler_score = max(0, 10 - (filler_ratio * 100))
        
        # Some variation is good, too much is bad
        variation_score = min(sent_variation * 5, 10.0) if sent_variation < 1.0 else 10.0 - sent_variation
        
        # Weighted combination
        fluency = (ttr_score * 0.4 + filler_score * 0.4 + variation_score * 0.2)
        return min(fluency, 10.0)
    
    def analyze_readability(self, text: str) -> Dict:
        """
        Analyze text readability.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability metrics
        """
        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "gunning_fog": 0.0,
                "readability_score": 0.0
            }
        
        # Calculate various readability metrics
        flesch = textstat.flesch_reading_ease(text)
        fk_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        
        # Normalize to 0-10 scale
        # Flesch: 0-100, higher is easier. Target: 60-70
        flesch_normalized = min(max(flesch / 10, 0), 10)
        
        # FK Grade: 0-18+, lower is easier. Target: 6-8
        fk_normalized = max(0, 10 - (fk_grade * 0.5))
        
        # Gunning Fog: 6-17+, lower is easier. Target: 7-9
        fog_normalized = max(0, 10 - ((gunning_fog - 6) * 0.5))
        
        # Average for overall readability score
        readability_score = (flesch_normalized + fk_normalized + fog_normalized) / 3
        
        return {
            "flesch_reading_ease": flesch,
            "flesch_kincaid_grade": fk_grade,
            "gunning_fog": gunning_fog,
            "readability_score": min(readability_score, 10.0)
        }
