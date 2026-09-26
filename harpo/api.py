"""
HARPO Core API for Evaluation, Comparison, and Explanation

Provides clean, modular interfaces for scoring, comparing, and explaining outputs.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from .model import HARPOMTv2
from .evaluation import (
    recall_at_k, mrr_at_k, ndcg_at_k, hit_rate_at_k
)


@dataclass
class ScoreResult:
    """Result from scoring a response"""
    relevance: float
    diversity: float
    satisfaction: float
    engagement: float
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ComparisonResult:
    """Result from comparing two responses"""
    preference_prob_a: float  # P(response_a > response_b)
    preference_prob_b: float  # P(response_b > response_a)
    margin: float  # Difference in scores
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preference_prob_a": self.preference_prob_a,
            "preference_prob_b": self.preference_prob_b,
            "margin": self.margin,
            "reasoning": self.reasoning
        }


@dataclass
class ExplanationResult:
    """Result from explaining a response"""
    reasoning_trace: List[str]
    vto_usage: List[Dict[str, Any]]
    confidence_signals: Dict[str, float]
    weak_signals: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evaluator:
    """
    HARPO Evaluator: Score responses across multiple dimensions.
    
    Usage:
        evaluator = Evaluator(model_path="path/to/model")
        scores = evaluator.score(context, response)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize evaluator with HARPO model"""
        self.device = device
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pretrained HARPO model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})
        
        self.model = HARPOMTv2(**config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
    
    def score(self, context: str, response: str) -> ScoreResult:
        """
        Score a single response in context.
        
        Args:
            context: Conversation context
            response: Model response to evaluate
            
        Returns:
            ScoreResult with relevance, diversity, satisfaction, engagement
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            # Tokenize input
            text = context + "\n" + response
            encoding = self.model.tokenizer(
                text, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            ).to(self.device)
            
            # Get model output
            outputs = self.model.base_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                output_hidden_states=True
            )
            
            # Extract reward scores
            hidden_states = outputs.hidden_states[-1]
            reward_logits = self.model.reward_head(hidden_states[:, -1, :])
            reward_scores = torch.sigmoid(reward_logits).cpu().numpy()[0]
        
        return ScoreResult(
            relevance=float(reward_scores[0]),
            diversity=float(reward_scores[1]),
            satisfaction=float(reward_scores[2]),
            engagement=float(reward_scores[3]),
            confidence=float(reward_scores[4]) if len(reward_scores) > 4 else None
        )
    
    def batch_score(self, contexts: List[str], responses: List[str]) -> List[ScoreResult]:
        """Score multiple response-context pairs"""
        return [self.score(ctx, resp) for ctx, resp in zip(contexts, responses)]


class Comparator:
    """
    HARPO Comparator: Compare two responses for preference.
    
    Usage:
        comparator = Comparator(model_path="path/to/model")
        result = comparator.compare(context, response_a, response_b)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize comparator with HARPO model"""
        self.device = device
        self.model = None
        self.evaluator = Evaluator(model_path, device)
        self.model = self.evaluator.model
    
    def load_model(self, model_path: str) -> None:
        """Load pretrained HARPO model"""
        self.evaluator.load_model(model_path)
        self.model = self.evaluator.model
    
    def compare(self, context: str, response_a: str, response_b: str) -> ComparisonResult:
        """
        Compare two responses in context.
        
        Args:
            context: Conversation context
            response_a: First response to compare
            response_b: Second response to compare
            
        Returns:
            ComparisonResult with preference probabilities and margin
        """
        score_a = self.evaluator.score(context, response_a)
        score_b = self.evaluator.score(context, response_b)
        
        # Compute preference scores
        score_a_avg = (score_a.relevance + score_a.diversity + 
                       score_a.satisfaction + score_a.engagement) / 4
        score_b_avg = (score_b.relevance + score_b.diversity + 
                       score_b.satisfaction + score_b.engagement) / 4
        
        margin = abs(score_a_avg - score_b_avg)
        
        # Softmax to get probabilities
        diff = score_a_avg - score_b_avg
        prob_a = 1.0 / (1.0 + torch.exp(torch.tensor(-diff)).item())
        prob_b = 1.0 - prob_a
        
        return ComparisonResult(
            preference_prob_a=prob_a,
            preference_prob_b=prob_b,
            margin=margin
        )
    
    def batch_compare(self, contexts: List[str], pairs: List[Tuple[str, str]]) -> List[ComparisonResult]:
        """Compare multiple response pairs"""
        return [self.compare(ctx, a, b) for ctx, (a, b) in zip(contexts, pairs)]


class Explainer:
    """
    HARPO Explainer: Explain why a response is good or bad.
    
    Usage:
        explainer = Explainer(model_path="path/to/model")
        explanation = explainer.explain(context, response)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize explainer with HARPO model"""
        self.device = device
        self.model = None
        self.evaluator = Evaluator(model_path, device)
        self.model = self.evaluator.model
    
    def load_model(self, model_path: str) -> None:
        """Load pretrained HARPO model"""
        self.evaluator.load_model(model_path)
        self.model = self.evaluator.model
    
    def explain(self, context: str, response: str) -> ExplanationResult:
        """
        Explain model's evaluation of a response.
        
        Args:
            context: Conversation context
            response: Response to explain
            
        Returns:
            ExplanationResult with reasoning traces and signal analysis
        """
        scores = self.evaluator.score(context, response)
        
        # Build reasoning trace
        reasoning_trace = []
        if scores.relevance > 0.7:
            reasoning_trace.append("Response is highly relevant to context")
        elif scores.relevance < 0.3:
            reasoning_trace.append("Response shows low relevance to context")
        
        if scores.diversity > 0.7:
            reasoning_trace.append("Response introduces diverse information")
        
        if scores.satisfaction > 0.7:
            reasoning_trace.append("Response likely satisfies user preferences")
        
        if scores.engagement > 0.7:
            reasoning_trace.append("Response is engaging and interactive")
        
        # Signal analysis
        confidence_signals = {
            "relevance": scores.relevance,
            "diversity": scores.diversity,
            "satisfaction": scores.satisfaction,
            "engagement": scores.engagement
        }
        
        weak_signals = []
        for signal, value in confidence_signals.items():
            if 0.3 < value < 0.7:
                weak_signals.append(f"Average {signal}: {value:.2f}")
        
        return ExplanationResult(
            reasoning_trace=reasoning_trace,
            vto_usage=[],  # Placeholder for VTO tracking
            confidence_signals=confidence_signals,
            weak_signals=weak_signals
        )
