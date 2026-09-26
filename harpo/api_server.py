"""
HARPO REST API

FastAPI wrapper for easy HTTP access to evaluation, comparison, and explanation.

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Examples:
    POST /evaluate
    POST /compare
    POST /explain
    GET /health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from .api import Evaluator, Comparator, Explainer


app = FastAPI(
    title="HARPO API",
    description="Hierarchical Agentic Reasoning with Preference Optimization",
    version="1.0.0"
)

# Global model instances (loaded once)
_evaluator: Optional[Evaluator] = None
_comparator: Optional[Comparator] = None
_explainer: Optional[Explainer] = None


def get_evaluator(model_path: Optional[str] = None) -> Evaluator:
    """Get or create evaluator instance"""
    global _evaluator
    if _evaluator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _evaluator = Evaluator(model_path=model_path, device=device)
    return _evaluator


def get_comparator(model_path: Optional[str] = None) -> Comparator:
    """Get or create comparator instance"""
    global _comparator
    if _comparator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _comparator = Comparator(model_path=model_path, device=device)
    return _comparator


def get_explainer(model_path: Optional[str] = None) -> Explainer:
    """Get or create explainer instance"""
    global _explainer
    if _explainer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _explainer = Explainer(model_path=model_path, device=device)
    return _explainer


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ScoreRequest(BaseModel):
    context: str = Field(..., description="Conversation context")
    response: str = Field(..., description="Model response to evaluate")
    model_path: Optional[str] = Field(None, description="Path to pretrained model")


class ScoreResponse(BaseModel):
    relevance: float
    diversity: float
    satisfaction: float
    engagement: float
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class CompareRequest(BaseModel):
    context: str = Field(..., description="Conversation context")
    response_a: str = Field(..., description="First response")
    response_b: str = Field(..., description="Second response")
    model_path: Optional[str] = Field(None, description="Path to pretrained model")


class CompareResponse(BaseModel):
    preference_prob_a: float
    preference_prob_b: float
    margin: float
    reasoning: Optional[str] = None
    winner: str = Field(..., description="'a', 'b', or 'tie'")


class ExplainRequest(BaseModel):
    context: str = Field(..., description="Conversation context")
    response: str = Field(..., description="Response to explain")
    model_path: Optional[str] = Field(None, description="Path to pretrained model")


class ExplainResponse(BaseModel):
    reasoning_trace: List[str]
    vto_usage: List[Dict[str, Any]]
    confidence_signals: Dict[str, float]
    weak_signals: List[str]


class BatchScoreRequest(BaseModel):
    items: List[Dict[str, str]] = Field(
        ...,
        description="List of {context, response} pairs"
    )
    model_path: Optional[str] = Field(None, description="Path to pretrained model")


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    summary: Dict[str, float]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/evaluate", response_model=ScoreResponse)
async def evaluate(request: ScoreRequest):
    """
    Score a single response.
    
    Returns metrics: relevance, diversity, satisfaction, engagement
    """
    try:
        evaluator = get_evaluator(request.model_path)
        result = evaluator.score(request.context, request.response)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-evaluate", response_model=BatchScoreResponse)
async def batch_evaluate(request: BatchScoreRequest):
    """
    Score multiple responses in batch.
    
    Efficient for evaluating many outputs at once.
    """
    try:
        evaluator = get_evaluator(request.model_path)
        
        results = []
        for item in request.items:
            context = item.get("context", "")
            response = item.get("response", "")
            score = evaluator.score(context, response)
            results.append(score.to_dict())
        
        # Compute summary statistics
        summary = {
            "avg_relevance": sum(r["relevance"] for r in results) / len(results),
            "avg_diversity": sum(r["diversity"] for r in results) / len(results),
            "avg_satisfaction": sum(r["satisfaction"] for r in results) / len(results),
            "avg_engagement": sum(r["engagement"] for r in results) / len(results),
        }
        
        return {"results": results, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    Compare two responses.
    
    Returns preference probabilities and margin.
    """
    try:
        comparator = get_comparator(request.model_path)
        result = comparator.compare(request.context, request.response_a, request.response_b)
        
        # Determine winner
        if result.preference_prob_a > result.preference_prob_b + 0.1:
            winner = "a"
        elif result.preference_prob_b > result.preference_prob_a + 0.1:
            winner = "b"
        else:
            winner = "tie"
        
        response_dict = result.to_dict()
        response_dict["winner"] = winner
        return response_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Explain evaluation of a response.
    
    Returns reasoning traces, VTO usage, and signal analysis.
    """
    try:
        explainer = get_explainer(request.model_path)
        result = explainer.explain(request.context, request.response)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Get API status and available models"""
    return {
        "api_version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "evaluator_loaded": _evaluator is not None,
        "comparator_loaded": _comparator is not None,
        "explainer_loaded": _explainer is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
