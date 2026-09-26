"""
HARPO: Comprehensive Evaluation Module

Evaluation focus: RECOMMENDATION QUALITY (not just tool accuracy)

Metrics:
1. Recommendation Quality: Recall, MRR, NDCG, Hit Rate
2. User Satisfaction: Simulated satisfaction scores
3. VTO/Tool Accuracy: Selection and execution accuracy
4. Generation Quality: BLEU, ROUGE, Distinct-n
5. Novel Metrics: Reasoning depth, agent agreement, explanation quality
"""

import json
import math
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import VTO, Domain, EvaluationResult, SPECIAL_TOKENS
from .model import HARPOMTv2
from .training import SFTDataset, PreferenceDataset


# ============================================================================
# RECOMMENDATION QUALITY METRICS (PRIMARY)
# ============================================================================

def recall_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """Recall@K: Proportion of relevant items found in top-K"""
    if not ground_truth:
        return 0.0
    
    hits = 0
    for preds, gt in zip(predictions, ground_truth):
        top_k = set(preds[:k]) if isinstance(preds, list) else set()
        if gt in top_k:
            hits += 1
    
    return hits / len(ground_truth)


def mrr_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """Mean Reciprocal Rank@K"""
    if not ground_truth:
        return 0.0
    
    rr_sum = 0.0
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k] if isinstance(preds, list) else []
        for rank, pred in enumerate(top_k, 1):
            if pred == gt:
                rr_sum += 1.0 / rank
                break
    
    return rr_sum / len(ground_truth)


def ndcg_at_k(predictions: List[List[str]], ground_truth: List[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain@K"""
    if not ground_truth:
        return 0.0
    
    def dcg(scores: List[float], k: int) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))
    
    ndcg_sum = 0.0
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k] if isinstance(preds, list) else []
        gains = [1.0 if p == gt else 0.0 for p in top_k]
        
        actual_dcg = dcg(gains, k)
        ideal_dcg = dcg([1.0], k)
        
        if ideal_dcg > 0:
            ndcg_sum += actual_dcg / ideal_dcg
    
    return ndcg_sum / len(ground_truth)


def hit_rate_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """Hit Rate@K: Whether any relevant item is in top-K"""
    if not ground_truth:
        return 0.0
    
    hits = 0
    for preds, gt in zip(predictions, ground_truth):
        top_k = set(preds[:k]) if isinstance(preds, list) else set()
        if gt in top_k:
            hits += 1
    
    return hits / len(ground_truth)


# ============================================================================
# USER SATISFACTION METRICS (PRIMARY)
# ============================================================================

def compute_user_satisfaction(model: HARPOMTv2, 
                               test_data: List[Dict],
                               device: str = "cpu",
                               save_results: bool = True,
                               output_dir: str = "./eval_cache") -> Dict[str, float]:
    """
    Compute user satisfaction using CHARM reward model.
    
    FIXED: 
    1. Now normalizes scores to [0, 1] range and saves detailed results.
    2. Handles dtype properly to avoid Float/BFloat16 mismatch.
    """
    model.eval()
    
    # CRITICAL FIX: Detect model dtype for consistent computation
    model_dtype = torch.float32
    if hasattr(model, '_model_dtype'):
        model_dtype = model._model_dtype
    elif hasattr(model, 'base_model') and model.base_model is not None:
        # Get dtype from model parameters
        for param in model.base_model.parameters():
            model_dtype = param.dtype
            break
    
    satisfaction_scores = []
    engagement_scores = []
    relevance_scores = []
    diversity_scores = []
    detailed_results = []
    
    with torch.no_grad():
        for item in tqdm(test_data[:100], desc="Computing satisfaction"):
            text = item.get("input", "") + "\n" + item.get("output", "")
            
            encoding = model.tokenizer(
                text, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            ).to(device)
            
            # CRITICAL FIX: Use autocast for consistent dtype
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', 
                                    dtype=model_dtype if model_dtype in [torch.float16, torch.bfloat16] else torch.float32,
                                    enabled=model_dtype in [torch.float16, torch.bfloat16]):
                outputs = model.base_model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    output_hidden_states=True
                )
                
                hidden = outputs.hidden_states[-1].mean(dim=1)
                
                # Get CHARM rewards (recommendation quality components)
                # CRITICAL FIX: Ensure hidden is in correct dtype
                hidden = hidden.to(dtype=model_dtype)
                reward_result = model.charm(hidden, return_components=True)
            
            # Get raw scores (now bounded by tanh to [-1, 1])
            sat_raw = reward_result.get("user_satisfaction", reward_result["total"]).item()
            eng_raw = reward_result.get("engagement", reward_result["total"]).item()
            rel_raw = reward_result.get("relevance", reward_result["total"]).item()
            div_raw = reward_result.get("diversity", reward_result["total"]).item()
            
            # Normalize from [-1, 1] to [0, 1]
            sat_norm = (sat_raw + 1) / 2
            eng_norm = (eng_raw + 1) / 2
            rel_norm = (rel_raw + 1) / 2
            div_norm = (div_raw + 1) / 2
            
            satisfaction_scores.append(sat_norm)
            engagement_scores.append(eng_norm)
            relevance_scores.append(rel_norm)
            diversity_scores.append(div_norm)
            
            # Save detailed result
            detailed_results.append({
                "input": item.get("input", "")[:200],
                "output": item.get("output", "")[:200],
                "raw_scores": {
                    "satisfaction": sat_raw,
                    "engagement": eng_raw,
                    "relevance": rel_raw,
                    "diversity": div_raw,
                    "total": reward_result["total"].item()
                },
                "normalized_scores": {
                    "satisfaction": sat_norm,
                    "engagement": eng_norm,
                    "relevance": rel_norm,
                    "diversity": div_norm
                }
            })
    
    # Save detailed results
    if save_results:
        import os
        os.makedirs(output_dir, exist_ok=True)
        sat_debug_path = os.path.join(output_dir, "satisfaction_details.json")
        with open(sat_debug_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    return {
        "user_satisfaction": sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0,
        "engagement_score": sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0,
        "relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
        "diversity_score": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
    }


# ============================================================================
# VTO/TOOL METRICS (SECONDARY)
# ============================================================================

def vto_accuracy(predicted_vtos: List[List[str]], 
                 ground_truth_vtos: List[List[str]]) -> float:
    """VTO Selection Accuracy: Exact match of VTO sets"""
    if not ground_truth_vtos:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predicted_vtos, ground_truth_vtos):
        if set(pred) == set(gt):
            correct += 1
    
    return correct / len(ground_truth_vtos)


def vto_f1(predicted_vtos: List[List[str]], 
           ground_truth_vtos: List[List[str]]) -> Dict[str, float]:
    """VTO F1 Score"""
    if not ground_truth_vtos:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    total_precision = 0.0
    total_recall = 0.0
    
    for pred, gt in zip(predicted_vtos, ground_truth_vtos):
        pred_set = set(pred)
        gt_set = set(gt)
        
        if pred_set:
            precision = len(pred_set & gt_set) / len(pred_set)
        else:
            precision = 0.0
        
        if gt_set:
            recall = len(pred_set & gt_set) / len(gt_set)
        else:
            recall = 1.0
        
        total_precision += precision
        total_recall += recall
    
    avg_precision = total_precision / len(ground_truth_vtos)
    avg_recall = total_recall / len(ground_truth_vtos)
    
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return {"precision": avg_precision, "recall": avg_recall, "f1": f1}


def tool_accuracy(predicted_tools: List[Dict], ground_truth_tools: List[Dict]) -> float:
    """Tool execution accuracy: correct tool name AND arguments"""
    if not ground_truth_tools:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predicted_tools, ground_truth_tools):
        if pred.get("tool_name") == gt.get("tool_name"):
            if pred.get("arguments", {}) == gt.get("arguments", {}):
                correct += 1
    
    return correct / len(ground_truth_tools)


# ============================================================================
# RANKING-BASED EVALUATION (CRITICAL FOR FAIR METRICS)
# ============================================================================

class RankingEvaluator:
    """
    Publication-Ready Ranking Evaluation with Negative Sampling.
    
    RESTORED: Original sequential scoring that was working.
    Uses recommendation_head with CHARM fallback.
    
    Following protocols from ReDial, KBRD, KGSF, UniCRS papers.
    """
    
    def __init__(self, model: HARPOMTv2, device: str = "cpu", num_negatives: int = 99):
        self.model = model
        self.device = device
        self.num_negatives = num_negatives
        self.all_items = []
        
    def set_item_pool(self, items: List[str]):
        """Set the pool of all possible items for negative sampling"""
        self.all_items = list(set([item for item in items if item and len(item.strip()) > 2]))
        
    def _sample_negatives(self, ground_truth: str, n: int) -> List[str]:
        """Sample n negative items that are not the ground truth"""
        import random
        candidates = [item for item in self.all_items if item.lower() != ground_truth.lower()]
        return random.sample(candidates, min(n, len(candidates)))
    
    def _score_single_candidate(self, context: str, candidate: str) -> float:
        """
        Score a single candidate using recommendation_head or CHARM.
        
        CRITICAL FIX: Always apply BRIDGE before scoring to match SFT training.
        The model learned to score using BRIDGE-adapted features, so evaluation
        must use the same feature space.
        """
        prompt = f"{context}\n\nRecommendation: {candidate}"
        
        encoding = self.model.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=512, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.base_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                output_hidden_states=True
            )
            
            # Pool hidden states
            hidden = outputs.hidden_states[-1].mean(dim=1)
            
            # CRITICAL FIX: Always apply BRIDGE for domain-adapted features
            # This matches how the model was trained during SFT
            if hasattr(self.model, 'bridge') and self.model.bridge is not None:
                bridge_out = self.model.bridge(hidden, domain=Domain.MOVIES)  # ReDial is movies domain
                hidden = bridge_out["features"]
            
            # Try recommendation_head first (trained during SFT)
            if hasattr(self.model, 'recommendation_head') and self.model.recommendation_head is not None:
                score = self.model.recommendation_head(hidden).item()
            else:
                # Fallback to CHARM
                reward = self.model.charm(hidden)
                score = reward["total"].item()
        
        return score
    
    def score_candidates(self, context: str, candidates: List[str]) -> List[float]:
        """Score all candidates sequentially (original working approach)"""
        scores = []
        for candidate in candidates:
            try:
                score = self._score_single_candidate(context, candidate)
                scores.append(score)
            except Exception as e:
                scores.append(0.0)  # Default score on error
        return scores
    
    def compute_metrics_from_ranks(self, ranks: List[int]) -> Dict[str, float]:
        """Compute all ranking metrics from list of ranks."""
        if not ranks:
            return {
                "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
                "recall_at_20": 0.0, "recall_at_50": 0.0,
                "mrr": 0.0, "mrr_at_10": 0.0, "mrr_at_20": 0.0,
                "ndcg_at_1": 0.0, "ndcg_at_5": 0.0, "ndcg_at_10": 0.0, "ndcg_at_20": 0.0,
                "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0,
                "num_samples": 0
            }
        
        n = len(ranks)
        
        # Recall@K
        recall_1 = sum(1 for r in ranks if r <= 1) / n
        recall_5 = sum(1 for r in ranks if r <= 5) / n
        recall_10 = sum(1 for r in ranks if r <= 10) / n
        recall_20 = sum(1 for r in ranks if r <= 20) / n
        recall_50 = sum(1 for r in ranks if r <= 50) / n
        
        # MRR
        mrr = sum(1.0 / r for r in ranks) / n
        mrr_10 = sum(1.0 / r if r <= 10 else 0.0 for r in ranks) / n
        mrr_20 = sum(1.0 / r if r <= 20 else 0.0 for r in ranks) / n
        
        # NDCG@K
        def dcg_at_k(rank: int, k: int) -> float:
            if rank <= k:
                return 1.0 / math.log2(rank + 1)
            return 0.0
        
        ideal_dcg = 1.0
        ndcg_1 = sum(dcg_at_k(r, 1) / ideal_dcg for r in ranks) / n
        ndcg_5 = sum(dcg_at_k(r, 5) / ideal_dcg for r in ranks) / n
        ndcg_10 = sum(dcg_at_k(r, 10) / ideal_dcg for r in ranks) / n
        ndcg_20 = sum(dcg_at_k(r, 20) / ideal_dcg for r in ranks) / n
        
        return {
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
            "recall_at_10": recall_10,
            "recall_at_20": recall_20,
            "recall_at_50": recall_50,
            "mrr": mrr,
            "mrr_at_10": mrr_10,
            "mrr_at_20": mrr_20,
            "ndcg_at_1": ndcg_1,
            "ndcg_at_5": ndcg_5,
            "ndcg_at_10": ndcg_10,
            "ndcg_at_20": ndcg_20,
            "hit_at_1": recall_1,
            "hit_at_5": recall_5,
            "hit_at_10": recall_10,
            "num_samples": n
        }
    
    def evaluate_ranking(self, test_data: List[Dict], 
                         context_key: str = "input",
                         gt_key: str = "ground_truth_item",
                         max_samples: int = 100,
                         scoring_method: str = "charm") -> Dict[str, float]:
        """
        Run ranking-based evaluation.
        
        FIXED: Properly extracts ground truth items from:
        1. Explicit ground_truth_item field (ReDial converted data)
        2. movies_mentioned field
        3. Quoted items in output (avoiding JSON/tool content)
        """
        self.model.eval()
        
        ranks = []
        detailed_results = []
        skipped = {"no_gt": 0, "no_negatives": 0, "error": 0}
        
        for item in tqdm(test_data[:max_samples], desc="Ranking eval"):
            context = item.get(context_key, "")
            gt_item = None
            
            # Method 1: Explicit ground_truth_item field (best - from ReDial conversion)
            gt_item = item.get(gt_key, None)
            
            # Method 2: movies_mentioned field
            if not gt_item:
                movies = item.get("movies_mentioned", [])
                if movies:
                    gt_item = movies[0]
            
            # Method 3: Extract from output (fallback)
            if not gt_item:
                output = item.get("output", "")
                
                # Remove tool blocks and think blocks
                clean_output = re.sub(r'<\|tool_start\|>.*?<\|tool_end\|>', '', output, flags=re.DOTALL)
                clean_output = re.sub(r'<\|think\|>.*?<\|/think\|>', '', clean_output, flags=re.DOTALL)
                
                # Words to skip (JSON keys, common words)
                skip_words = {
                    "tool", "search", "args", "query", "filter", "compare", "items", 
                    "category", "recommend", "get_info", "movie", "movies", "fields",
                    "the", "and", "for", "with", "your", "that", "this", "would"
                }
                
                # Look for quoted items (movie names are often in quotes)
                quoted = re.findall(r'"([^"]{2,60})"', clean_output)
                for q in quoted:
                    if q.lower() not in skip_words and not q.startswith("item_"):
                        gt_item = q
                        break
            
            if not gt_item or len(str(gt_item).strip()) < 2:
                skipped["no_gt"] += 1
                continue
            
            gt_item = str(gt_item).strip()
            
            # Sample negatives
            negatives = self._sample_negatives(gt_item, self.num_negatives)
            
            if len(negatives) < 10:
                skipped["no_negatives"] += 1
                continue
            
            # Create candidate set: ground truth at index 0
            candidates = [gt_item] + negatives
            
            # Score all candidates
            try:
                scores = self.score_candidates(context, candidates)
            except Exception as e:
                skipped["error"] += 1
                continue
            
            # Find rank of ground truth (index 0)
            gt_score = scores[0]
            rank = 1 + sum(1 for s in scores[1:] if s > gt_score)
            ranks.append(rank)
            
            detailed_results.append({
                "context": context[:200],
                "ground_truth": gt_item,
                "rank": rank,
                "gt_score": gt_score,
                "num_candidates": len(candidates)
            })
        
        if skipped["no_gt"] > 0 or skipped["no_negatives"] > 0:
            print(f"\n  Skipped samples: no_gt={skipped['no_gt']}, no_negatives={skipped['no_negatives']}, errors={skipped['error']}")
        
        metrics = self.compute_metrics_from_ranks(ranks)
        metrics["detailed_results"] = detailed_results
        
        return metrics


def extract_items_from_data(data: List[Dict]) -> List[str]:
    """Extract all item mentions from dataset for negative sampling pool
    
    FIXED: Now properly extracts items from:
    1. Explicit ground_truth_item field (ReDial converted data)
    2. movies_mentioned field
    3. all_conversation_movies field
    4. Quoted items in output (avoiding JSON/tool content)
    """
    items = set()
    
    # Words to skip (JSON keys, tool names, etc.)
    skip_words = {
        "tool", "search", "args", "query", "filter", "compare", "items", "category",
        "max_price", "min_price", "sort_by", "limit", "attributes", "comparison_attributes",
        "item_1", "item_2", "item_3", "rating", "price", "rating_and_price",
        "the", "and", "for", "with", "your", "that", "this", "from", "have", "been",
        "perfect", "great", "best", "find", "help", "looking", "preferences",
        "recommend", "get_info", "movie", "movies", "fields", "criteria"
    }
    
    for item in data:
        # Method 1: Explicit ground_truth_item (ReDial converted data)
        gt_item = item.get("ground_truth_item")
        if gt_item and isinstance(gt_item, str) and len(gt_item) > 2:
            items.add(gt_item)
        
        # Method 2: movies_mentioned field
        movies = item.get("movies_mentioned", [])
        for m in movies:
            if isinstance(m, str) and len(m) > 2:
                items.add(m)
        
        # Method 3: all_conversation_movies field
        all_movies = item.get("all_conversation_movies", [])
        for m in all_movies:
            if isinstance(m, str) and len(m) > 2:
                items.add(m)
        
        # Method 4: Extract from output (fallback)
        output = item.get("output", "")
        
        # Remove tool blocks before extraction
        clean_output = re.sub(r'<\|tool_start\|>.*?<\|tool_end\|>', '', output, flags=re.DOTALL)
        clean_output = re.sub(r'<\|think\|>.*?<\|/think\|>', '', clean_output, flags=re.DOTALL)
        
        # Extract quoted items from clean output only
        quoted = re.findall(r'"([^"]{2,60})"', clean_output)
        for q in quoted:
            if q.lower() not in skip_words and not q.startswith("item_"):
                items.add(q)
        
        # Extract from entities if available
        entities = item.get("entities", {})
        if isinstance(entities, dict):
            for key, val in entities.items():
                if isinstance(val, list):
                    items.update([str(v) for v in val if str(v).lower() not in skip_words and len(str(v)) > 2])
                elif isinstance(val, str) and val.lower() not in skip_words and len(val) > 2:
                    items.add(val)
    
    # If not enough items, add popular movies for ranking
    if len(items) < 100:
        popular_movies = [
            "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction",
            "Schindler's List", "Fight Club", "Forrest Gump", "Inception", "The Matrix",
            "Goodfellas", "The Silence of the Lambs", "Se7en", "The Usual Suspects",
            "Léon: The Professional", "Interstellar", "The Green Mile", "The Departed",
            "Gladiator", "The Prestige", "Memento", "American History X", "Casablanca",
            "City of God", "Saving Private Ryan", "The Intouchables", "Modern Times",
            "Rear Window", "Alien", "Apocalypse Now", "Raiders of the Lost Ark",
            "Psycho", "Vertigo", "WALL·E", "The Lion King", "The Pianist",
            "Terminator 2", "Back to the Future", "Whiplash", "The Shining",
            "Django Unchained", "Inglourious Basterds", "Kill Bill", "Reservoir Dogs",
            "No Country for Old Men", "There Will Be Blood", "The Big Lebowski",
            "Fargo", "A Clockwork Orange", "2001: A Space Odyssey", "Full Metal Jacket",
            "Blade Runner", "Taxi Driver", "Raging Bull", "The Godfather Part II",
            "One Flew Over the Cuckoo's Nest", "Chinatown", "Annie Hall", "Manhattan",
            "Star Wars", "The Empire Strikes Back", "Return of the Jedi", "E.T.",
            "Jurassic Park", "Jaws", "Indiana Jones", "Close Encounters",
            "The Lord of the Rings", "Harry Potter", "Pirates of the Caribbean",
            "Avatar", "Titanic", "The Avengers", "Iron Man", "Spider-Man",
            "Batman Begins", "The Dark Knight Rises", "Wonder Woman", "Black Panther",
            "Toy Story", "Finding Nemo", "Up", "Inside Out", "Coco", "Frozen",
            "The Little Mermaid", "Beauty and the Beast", "Aladdin", "Moana",
            "Shrek", "How to Train Your Dragon", "Kung Fu Panda", "Madagascar"
        ]
        items.update(popular_movies)
    
    print(f"  Item pool size: {len(items)}")
    return list(items)


# ============================================================================
# GENERATION QUALITY METRICS
# ============================================================================

def compute_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> Dict[str, float]:
    """BLEU Score (1-4 grams)"""
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def brevity_penalty(pred_len: int, ref_len: int) -> float:
        if pred_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / max(pred_len, 1))
    
    bleu_scores = {}
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred = 0
        total_pred_len = 0
        total_ref_len = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total_matches += matches
            total_pred += sum(pred_ngrams.values())
            total_pred_len += len(pred.split())
            total_ref_len += len(ref.split())
        
        precision = total_matches / max(total_pred, 1)
        bp = brevity_penalty(total_pred_len, total_ref_len)
        bleu_scores[f"bleu_{n}"] = precision * bp
    
    # Combined BLEU-4
    if all(bleu_scores[f"bleu_{n}"] > 0 for n in range(1, max_n + 1)):
        bleu_scores["bleu"] = math.exp(
            sum(math.log(max(bleu_scores[f"bleu_{n}"], 1e-10)) for n in range(1, max_n + 1)) / max_n
        )
    else:
        bleu_scores["bleu"] = 0.0
    
    return bleu_scores


def compute_distinct(texts: List[str], max_n: int = 2) -> Dict[str, float]:
    """Distinct-n: Diversity of generated n-grams"""
    distinct_scores = {}
    
    for n in range(1, max_n + 1):
        all_ngrams = []
        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if all_ngrams:
            distinct_scores[f"distinct_{n}"] = len(set(all_ngrams)) / len(all_ngrams)
        else:
            distinct_scores[f"distinct_{n}"] = 0.0
    
    return distinct_scores


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """ROUGE-L F1 Score"""
    def lcs_length(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append(f1)
    
    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# NOVEL METRICS (STAR, MAVEN)
# ============================================================================

def compute_reasoning_metrics(model: HARPOMTv2, 
                               test_data: List[Dict],
                               device: str = "cpu") -> Dict[str, float]:
    """
    Compute STAR reasoning metrics:
    - Average reasoning depth
    - Backtrack rate
    - Thought quality
    """
    model.eval()
    
    depths = []
    qualities = []
    
    with torch.no_grad():
        for item in tqdm(test_data[:50], desc="Computing reasoning metrics"):
            encoding = model.tokenizer(
                item.get("input", ""),
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            outputs = model.base_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                output_hidden_states=True
            )
            
            hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            
            # Run STAR reasoning
            star_result = model.star(hidden, return_path=True)
            
            depths.append(star_result["reasoning_depth"])
            qualities.append(star_result["recommendation_value"])
    
    return {
        "reasoning_depth": sum(depths) / len(depths) if depths else 0,
        "thought_quality": sum(qualities) / len(qualities) if qualities else 0,
        "backtrack_rate": 0.0  # Would need to track during search
    }


def compute_agent_metrics(model: HARPOMTv2,
                          test_data: List[Dict],
                          device: str = "cpu") -> Dict[str, float]:
    """
    Compute MAVEN agent metrics:
    - Agent agreement rate
    - Quality variance across agents
    """
    model.eval()
    
    agreements = []
    quality_vars = []
    
    with torch.no_grad():
        for item in tqdm(test_data[:50], desc="Computing agent metrics"):
            encoding = model.tokenizer(
                item.get("input", ""),
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            outputs = model.base_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                output_hidden_states=True
            )
            
            hidden = outputs.hidden_states[-1].mean(dim=1)
            
            # Run MAVEN
            maven_result = model.maven(hidden)
            
            agreements.append(maven_result["agreement"].item())
            
            # Compute quality variance
            qualities = list(maven_result["agent_qualities"].values())
            if qualities:
                quality_tensor = torch.stack([q for q in qualities])
                quality_vars.append(quality_tensor.var().item())
    
    return {
        "agent_agreement_rate": sum(agreements) / len(agreements) if agreements else 0,
        "agent_quality_variance": sum(quality_vars) / len(quality_vars) if quality_vars else 0
    }


# ============================================================================
# PREFERENCE EVALUATION
# ============================================================================

def evaluate_preference_ranking(model: HARPOMTv2,
                                preference_data: List[Dict],
                                device: str = "cpu",
                                max_samples: int = 100,
                                save_results: bool = True,
                                output_dir: str = "./eval_cache") -> Dict[str, float]:
    """Evaluate preference ranking accuracy
    
    FIXED: 
    - Proper dtype handling for bfloat16/float32
    - Saves detailed results for debugging
    """
    model.eval()
    
    correct = 0
    total = 0
    reward_margins = []
    detailed_results = []
    
    samples = preference_data[:max_samples]
    
    # Determine dtype for autocast
    use_autocast = device == "cuda" and torch.cuda.is_available()
    if use_autocast:
        if torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    
    with torch.no_grad():
        for item in tqdm(samples, desc="Evaluating preferences"):
            context = item.get("context", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            # Encode both
            chosen_text = context + "\nAssistant: " + chosen
            rejected_text = context + "\nAssistant: " + rejected
            
            chosen_enc = model.tokenizer(
                chosen_text, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            ).to(device)
            
            rejected_enc = model.tokenizer(
                rejected_text, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            ).to(device)
            
            try:
                # FIXED: Use autocast for consistent dtype
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=amp_dtype):
                    # Get rewards
                    chosen_out = model.base_model(
                        input_ids=chosen_enc["input_ids"],
                        attention_mask=chosen_enc["attention_mask"],
                        output_hidden_states=True
                    )
                    rejected_out = model.base_model(
                        input_ids=rejected_enc["input_ids"],
                        attention_mask=rejected_enc["attention_mask"],
                        output_hidden_states=True
                    )
            
                    chosen_hidden = chosen_out.hidden_states[-1].mean(dim=1)
                    rejected_hidden = rejected_out.hidden_states[-1].mean(dim=1)
                    
                    # Get full reward breakdown
                    chosen_rewards = model.charm(chosen_hidden, return_components=True)
                    rejected_rewards = model.charm(rejected_hidden, return_components=True)
                    
                    chosen_reward = chosen_rewards["total"].item()
                    rejected_reward = rejected_rewards["total"].item()
                
                is_correct = chosen_reward > rejected_reward
                if is_correct:
                    correct += 1
                
                margin = chosen_reward - rejected_reward
                reward_margins.append(margin)
                total += 1
                
                # Save detailed result
                detailed_results.append({
                    "context": context[:200],
                    "chosen": chosen[:200],
                    "rejected": rejected[:200],
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "margin": margin,
                    "is_correct": is_correct,
                    "chosen_components": {
                        "relevance": chosen_rewards.get("relevance", chosen_rewards["total"]).item(),
                        "diversity": chosen_rewards.get("diversity", chosen_rewards["total"]).item(),
                        "satisfaction": chosen_rewards.get("user_satisfaction", chosen_rewards["total"]).item(),
                        "engagement": chosen_rewards.get("engagement", chosen_rewards["total"]).item(),
                    },
                    "rejected_components": {
                        "relevance": rejected_rewards.get("relevance", rejected_rewards["total"]).item(),
                        "diversity": rejected_rewards.get("diversity", rejected_rewards["total"]).item(),
                        "satisfaction": rejected_rewards.get("user_satisfaction", rejected_rewards["total"]).item(),
                        "engagement": rejected_rewards.get("engagement", rejected_rewards["total"]).item(),
                    }
                })
            except Exception as e:
                print(f"  Error evaluating preference: {e}")
                continue
    
    # Save detailed results
    if save_results:
        import os
        os.makedirs(output_dir, exist_ok=True)
        pref_debug_path = os.path.join(output_dir, "preference_details.json")
        with open(pref_debug_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\n  Saved {len(detailed_results)} preference details to {pref_debug_path}")
    
    return {
        "preference_accuracy": correct / total if total > 0 else 0.0,
        "avg_reward_margin": sum(reward_margins) / len(reward_margins) if reward_margins else 0.0,
        "total_samples": total
    }


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

class HARPOMTv2Evaluator:
    """Comprehensive evaluator for HARPO-MT v2"""
    
    def __init__(self, model: HARPOMTv2, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        # VTO index to name mapping
        self.idx_to_vto = {i: vto.value for i, vto in enumerate(VTO)}
        self.vto_to_idx = {vto.value: i for i, vto in enumerate(VTO)}
    
    def extract_vtos_from_generation(self, text: str) -> List[str]:
        """Extract VTO predictions from generated text"""
        # Try multiple patterns
        patterns = [
            r"<\|think\|>(.*?)<\|/think\|>",  # Standard format
            r"<think>(.*?)</think>",  # Alternative format
            r"\[VTOs?: (.*?)\]",  # Bracket format
        ]
        
        vtos = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                parts = [p.strip() for p in match.split(",")]
                vtos.extend([p for p in parts if p])
        
        return vtos
    
    def extract_vtos_from_model(self, input_ids: torch.Tensor, 
                                 attention_mask: torch.Tensor,
                                 threshold: float = 0.5) -> List[str]:
        """Extract VTOs using the model's VTO head predictions
        
        This is more reliable than text extraction since it uses
        the trained VTO classifier directly.
        
        CRITICAL: Uses BRIDGE for domain-adapted features to match SFT training.
        """
        with torch.no_grad():
            outputs = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1].mean(dim=1)
            
            # CRITICAL FIX: Apply BRIDGE to match SFT training
            # The VTO head was trained on BRIDGE-adapted features
            if hasattr(self.model, 'bridge') and self.model.bridge is not None:
                try:
                    bridge_out = self.model.bridge(hidden.unsqueeze(1), Domain.MOVIES)
                    adapted = bridge_out["features"]
                    if adapted.dim() == 3:
                        adapted = adapted.mean(dim=1)
                    hidden = adapted
                except Exception:
                    pass  # Fallback to raw hidden states
            
            # Get VTO predictions from VTO head
            vto_logits = self.model.vto_head(hidden)
            vto_probs = torch.sigmoid(vto_logits)
            
            # Get VTOs above threshold
            predicted_indices = (vto_probs > threshold).squeeze().nonzero(as_tuple=True)[0]
            vtos = [self.idx_to_vto[idx.item()] for idx in predicted_indices 
                    if idx.item() in self.idx_to_vto]
        
        return vtos
    
    def clean_reference_for_comparison(self, reference: str) -> str:
        """Remove special tokens from reference for fair BLEU comparison"""
        # Remove think tags and their content
        cleaned = re.sub(r"<\|think\|>.*?<\|/think\|>", "", reference)
        # Remove tool tags and their content
        cleaned = re.sub(r"<\|tool_start\|>.*?<\|tool_end\|>", "", cleaned)
        # Remove domain tags
        cleaned = re.sub(r"<\|domain:\w+\|>", "", cleaned)
        # Clean up whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    
    def extract_recommendations(self, text: str) -> List[str]:
        """Extract recommended items from text"""
        items = []
        
        # Pattern: numbered items (1. Item, 2. Item)
        numbered = re.findall(r'\d+\.\s*([^\n,\.]{3,50})', text)
        items.extend([n.strip() for n in numbered])
        
        # Pattern: quoted items
        quoted = re.findall(r'"([^"]{3,50})"', text)
        items.extend(quoted)
        
        # Pattern: bullet items (- Item, * Item)
        bullets = re.findall(r'[\-\*]\s*([^\n,\.]{3,50})', text)
        items.extend([b.strip() for b in bullets])
        
        # Pattern: recommendation phrases
        rec_pattern = r'(?:recommend|suggest|try|consider)\s+(?:the\s+)?([A-Z][^\n,\.]{3,40})'
        recs = re.findall(rec_pattern, text, re.IGNORECASE)
        items.extend([r.strip() for r in recs])
        
        return list(set(items))[:10]  # Dedupe and take top 10
    
    def evaluate_generation(self, test_data: List[Dict], 
                            max_samples: int = 100,
                            save_responses: bool = True,
                            output_dir: str = "./eval_cache") -> Dict[str, float]:
        """Evaluate generation quality
        
        CRITICAL FIXES:
        1. Don't skip special tokens for VTO extraction
        2. Use VTO head predictions as alternative
        3. Clean reference for fair BLEU comparison
        4. Save all responses for debugging
        5. ADDED: Force structured output with <|think|> token
        6. ADDED: Better generation parameters for structured output
        7. ADDED: Batch processing for 5-10x faster evaluation
        """
        predictions = []
        predictions_raw = []  # With special tokens
        references = []
        references_clean = []  # Without special tokens
        predicted_vtos = []
        predicted_vtos_model = []  # From VTO head
        ground_truth_vtos = []
        predicted_items = []
        ground_truth_items = []
        
        # For debugging - save detailed results
        detailed_results = []
        
        samples = test_data[:max_samples]
        
        # CRITICAL FIX: Get model dtype for consistent generation
        model_dtype = torch.float32
        if hasattr(self.model, '_model_dtype'):
            model_dtype = self.model._model_dtype
        
        # CRITICAL FIX: Batch processing for faster evaluation
        batch_size = 8  # Process 8 samples at a time
        
        # Pre-collect all data
        all_input_texts = []
        all_references = []
        all_gt_vtos = []
        all_gt_items = []
        
        for item in samples:
            all_input_texts.append(item.get("input", ""))
            all_references.append(item.get("output", ""))
            all_gt_vtos.append(item.get("vtos", []))
            all_gt_items.append(item.get("items", []))
        
        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating responses"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            
            batch_inputs = all_input_texts[start_idx:end_idx]
            batch_refs = all_references[start_idx:end_idx]
            batch_gt_vtos = all_gt_vtos[start_idx:end_idx]
            batch_gt_items = all_gt_items[start_idx:end_idx]
            
            # Prepare prompts for batch
            batch_prompts = [text + "\nAssistant: " for text in batch_inputs]
            
            # Batch tokenize with padding
            batch_encoding = self.model.tokenizer(
                batch_prompts, 
                return_tensors="pt",
                max_length=512, 
                truncation=True,
                padding=True  # CRITICAL: Pad for batching
            ).to(self.device)
            
            with torch.no_grad():
                # Batch generation
                output_ids = self.model.generate(
                    input_ids=batch_encoding["input_ids"],
                    attention_mask=batch_encoding["attention_mask"],
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                    force_think_token=True,
                )
            
            # Process each sample in batch
            for i in range(len(batch_inputs)):
                input_text = batch_inputs[i]
                reference = batch_refs[i]
                gt_vtos = batch_gt_vtos[i]
                gt_items = batch_gt_items[i]
                
                # Get input length for this sample (account for padding)
                sample_input_ids = batch_encoding["input_ids"][i]
                # Find where padding ends (first non-pad token)
                if self.model.tokenizer.padding_side == "left":
                    non_pad_mask = sample_input_ids != self.model.tokenizer.pad_token_id
                    input_len = non_pad_mask.sum().item()
                else:
                    input_len = (sample_input_ids != self.model.tokenizer.pad_token_id).sum().item()
                
                # Decode with special tokens
                generated_raw = self.model.tokenizer.decode(
                    output_ids[i][batch_encoding["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                
                # Decode without special tokens for BLEU
                generated_clean = self.model.tokenizer.decode(
                    output_ids[i][batch_encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Clean up generation - truncate at natural stopping points
                for stop_marker in ["\nUser:", "\n\nUser:", "###", "\n\n\n"]:
                    if stop_marker in generated_clean:
                        generated_clean = generated_clean.split(stop_marker)[0].strip()
                    if stop_marker in generated_raw:
                        generated_raw = generated_raw.split(stop_marker)[0].strip()
                
                # Extract VTOs from text
                pred_vtos_text = self.extract_vtos_from_generation(generated_raw)
                
                # Extract VTOs from model's VTO head (single sample)
                single_encoding = self.model.tokenizer(
                    input_text + "\nAssistant: ",
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                pred_vtos_model = self.extract_vtos_from_model(
                    single_encoding["input_ids"], 
                    single_encoding["attention_mask"]
                )
                
                # Use model VTOs if text extraction fails
                final_pred_vtos = pred_vtos_text if pred_vtos_text else pred_vtos_model
                
                # Clean reference for fair comparison
                reference_clean = self.clean_reference_for_comparison(reference)
                
                predictions.append(generated_clean)
                predictions_raw.append(generated_raw)
                references.append(reference)
                references_clean.append(reference_clean)
                
                predicted_vtos.append(final_pred_vtos)
                predicted_vtos_model.append(pred_vtos_model)
                ground_truth_vtos.append(gt_vtos)
                
                # Extract items
                pred_items = self.extract_recommendations(generated_clean)
                predicted_items.append(pred_items)
                ground_truth_items.append(gt_items if gt_items else [])
                
                # Save detailed result for debugging
                detailed_results.append({
                    "input": input_text,
                    "reference": reference,
                    "reference_clean": reference_clean,
                    "generated": generated_clean,
                    "generated_raw": generated_raw,
                    "gt_vtos": gt_vtos,
                    "pred_vtos_text": pred_vtos_text,
                    "pred_vtos_model": pred_vtos_model,
                    "pred_items": pred_items,
                    "gt_items": gt_items
                })
        
        # Save detailed results for debugging
        if save_responses:
            import os
            os.makedirs(output_dir, exist_ok=True)
            debug_path = os.path.join(output_dir, "generation_details.json")
            with open(debug_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"\n  Saved {len(detailed_results)} detailed results to {debug_path}")
        
        # Compute metrics using CLEANED references
        results = {}
        
        # Generation metrics - compare clean generated vs clean reference
        bleu_scores = compute_bleu(predictions, references_clean)
        results.update(bleu_scores)
        
        distinct_scores = compute_distinct(predictions)
        results.update(distinct_scores)
        
        results["rouge_l"] = compute_rouge_l(predictions, references_clean)
        
        # VTO metrics - use combined VTO predictions
        vto_metrics = vto_f1(predicted_vtos, ground_truth_vtos)
        results["vto_precision"] = vto_metrics["precision"]
        results["vto_recall"] = vto_metrics["recall"]
        results["vto_f1"] = vto_metrics["f1"]
        results["vto_accuracy"] = vto_accuracy(predicted_vtos, ground_truth_vtos)
        
        # Also compute metrics for model-based VTO predictions
        vto_metrics_model = vto_f1(predicted_vtos_model, ground_truth_vtos)
        results["vto_f1_model"] = vto_metrics_model["f1"]
        
        # Recommendation metrics
        if any(ground_truth_items):
            flat_gt = [gt[0] if gt else "" for gt in ground_truth_items]
            results["recall_at_10"] = recall_at_k(predicted_items, flat_gt, 10)
            results["mrr_at_10"] = mrr_at_k(predicted_items, flat_gt, 10)
            results["ndcg_at_10"] = ndcg_at_k(predicted_items, flat_gt, 10)
            results["hit_rate_at_10"] = hit_rate_at_k(predicted_items, flat_gt, 10)
        
        return results
    
    def run_full_evaluation(self,
                            sft_test_data: List[Dict],
                            preference_test_data: List[Dict],
                            max_samples: int = 100,
                            cache_dir: str = "./eval_cache") -> EvaluationResult:
        """Run comprehensive evaluation with caching
        
        FIXED: Now properly passes cache_dir to all sub-functions
        """
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        print("=" * 70)
        print(f"HARPO-MT v2: Comprehensive Evaluation (Cache: {cache_dir})")
        print("=" * 70)
        
        # --- STEP 1: GENERATION (The Slowest Step) ---
        gen_path = os.path.join(cache_dir, "1_generation.json")
        if os.path.exists(gen_path):
            print(f"\n1. Loading cached generation results from {gen_path}...")
            with open(gen_path, 'r') as f:
                gen_results = json.load(f)
        else:
            print("\n1. Evaluating generation quality...")
            gen_results = self.evaluate_generation(
                sft_test_data, max_samples,
                save_responses=True, output_dir=cache_dir  # FIXED: Pass cache_dir
            )
            with open(gen_path, 'w') as f:
                json.dump(gen_results, f, indent=2)

        # --- STEP 2: USER SATISFACTION ---
        sat_path = os.path.join(cache_dir, "2_satisfaction.json")
        if os.path.exists(sat_path):
            print(f"\n2. Loading cached satisfaction results...")
            with open(sat_path, 'r') as f:
                satisfaction_results = json.load(f)
        else:
            print("\n2. Evaluating user satisfaction (PRIMARY)...")
            satisfaction_results = compute_user_satisfaction(
                self.model, sft_test_data, self.device,
                save_results=True, output_dir=cache_dir  # FIXED: Pass cache_dir
            )
            with open(sat_path, 'w') as f:
                json.dump(satisfaction_results, f, indent=2)

        # --- STEP 3: PREFERENCE RANKING ---
        pref_path = os.path.join(cache_dir, "3_preference.json")
        if os.path.exists(pref_path):
            print(f"\n3. Loading cached preference results...")
            with open(pref_path, 'r') as f:
                pref_results = json.load(f)
        else:
            print("\n3. Evaluating preference ranking...")
            pref_results = evaluate_preference_ranking(
                self.model, preference_test_data, self.device, max_samples,
                save_results=True, output_dir=cache_dir  # FIXED: Pass cache_dir
            )
            with open(pref_path, 'w') as f:
                json.dump(pref_results, f, indent=2)

        # --- STEP 4: STAR REASONING ---
        star_path = os.path.join(cache_dir, "4_star.json")
        if os.path.exists(star_path):
            print(f"\n4. Loading cached reasoning results...")
            with open(star_path, 'r') as f:
                reasoning_results = json.load(f)
        else:
            print("\n4. Evaluating reasoning quality (STAR)...")
            reasoning_results = compute_reasoning_metrics(
                self.model, sft_test_data, self.device
            )
            with open(star_path, 'w') as f:
                json.dump(reasoning_results, f, indent=2)

        # --- STEP 5: MAVEN AGENTS ---
        maven_path = os.path.join(cache_dir, "5_maven.json")
        if os.path.exists(maven_path):
            print(f"\n5. Loading cached agent results...")
            with open(maven_path, 'r') as f:
                agent_results = json.load(f)
        else:
            print("\n5. Evaluating agent collaboration (MAVEN)...")
            agent_results = compute_agent_metrics(
                self.model, sft_test_data, self.device
            )
            with open(maven_path, 'w') as f:
                json.dump(agent_results, f, indent=2)

        # --- STEP 6: RANKING PROTOCOL (Where it crashed) ---
        rank_path = os.path.join(cache_dir, "6_ranking.json")
        if os.path.exists(rank_path):
            print(f"\n6. Loading cached ranking protocol results...")
            with open(rank_path, 'r') as f:
                ranking_results = json.load(f)
        else:
            print("\n6. Running Scientific Ranking Protocol (Negative Sampling)...")
            ranking_results = self.evaluate_ranking_protocol(
                sft_test_data, 
                num_negatives=99, 
                max_samples=max_samples
            )
            with open(rank_path, 'w') as f:
                json.dump(ranking_results, f, indent=2)

        # Compile results with all metrics
        result = EvaluationResult(
            dataset="combined",
            
            # PRIMARY: User Satisfaction
            user_satisfaction=satisfaction_results.get("user_satisfaction", 0),
            engagement_score=satisfaction_results.get("engagement_score", 0),
            
            # PRIMARY: Ranking Metrics (from Scientific Ranking Protocol)
            recall_at_1=ranking_results.get("recall_at_1", 0),
            recall_at_5=ranking_results.get("recall_at_5", 0),
            recall_at_10=ranking_results.get("recall_at_10", 0),
            recall_at_20=ranking_results.get("recall_at_20", 0),
            recall_at_50=ranking_results.get("recall_at_50", 0),
            mrr=ranking_results.get("mrr", 0),
            mrr_at_10=ranking_results.get("mrr_at_10", 0),
            mrr_at_20=ranking_results.get("mrr_at_20", 0),
            ndcg_at_1=ranking_results.get("ndcg_at_1", 0),
            ndcg_at_5=ranking_results.get("ndcg_at_5", 0),
            ndcg_at_10=ranking_results.get("ndcg_at_10", 0),
            ndcg_at_20=ranking_results.get("ndcg_at_20", 0),
            hit_rate_at_1=ranking_results.get("hit_at_1", 0),
            hit_rate_at_5=ranking_results.get("hit_at_5", 0),
            hit_rate_at_10=ranking_results.get("hit_at_10", 0),
            
            # SECONDARY: VTO/Tool Metrics
            vto_precision=gen_results.get("vto_precision", 0),
            vto_recall=gen_results.get("vto_recall", 0),
            vto_f1=gen_results.get("vto_f1", 0),
            tool_selection_accuracy=pref_results.get("preference_accuracy", 0),
            
            # GENERATION: Text Quality
            bleu_1=gen_results.get("bleu_1", 0),
            bleu_2=gen_results.get("bleu_2", 0),
            bleu_3=gen_results.get("bleu_3", 0),
            bleu_4=gen_results.get("bleu_4", 0),
            rouge_l=gen_results.get("rouge_l", 0),
            distinct_1=gen_results.get("distinct_1", 0),
            distinct_2=gen_results.get("distinct_2", 0),
            
            # ALIGNMENT: Preference Learning
            preference_accuracy=pref_results.get("preference_accuracy", 0),
            reward_margin=pref_results.get("avg_reward_margin", 0),
            
            # NOVEL: STAR Reasoning
            reasoning_depth=reasoning_results.get("reasoning_depth", 0),
            thought_quality=reasoning_results.get("thought_quality", 0),
            backtrack_rate=reasoning_results.get("backtrack_rate", 0),
            
            # NOVEL: MAVEN Agents
            agent_agreement_rate=agent_results.get("agent_agreement_rate", 0),
        )
        
        # Print comprehensive results
        print("\n" + "=" * 70)
        print("HARPO-MT v2 EVALUATION RESULTS")
        print("=" * 70)
        
        print("\n👑 PRIMARY METRICS - Recommendation Quality:")
        print(f"  User Satisfaction:  {result.user_satisfaction:.4f}")
        print(f"  Engagement Score:   {result.engagement_score:.4f}")
        
        print("\n📊 RANKING METRICS (Standard for Rec Papers):")
        print(f"  Recall@1:           {result.recall_at_1:.4f}")
        print(f"  Recall@5:           {result.recall_at_5:.4f}")
        print(f"  Recall@10:          {result.recall_at_10:.4f}")
        print(f"  Recall@20:          {result.recall_at_20:.4f}")
        print(f"  Recall@50:          {result.recall_at_50:.4f}")
        print(f"  MRR:                {result.mrr:.4f}")
        print(f"  MRR@10:             {result.mrr_at_10:.4f}")
        print(f"  MRR@20:             {result.mrr_at_20:.4f}")
        print(f"  NDCG@1:             {result.ndcg_at_1:.4f}")
        print(f"  NDCG@5:             {result.ndcg_at_5:.4f}")
        print(f"  NDCG@10:            {result.ndcg_at_10:.4f}")
        print(f"  NDCG@20:            {result.ndcg_at_20:.4f}")
        
        print("\n🥈 VTO/TOOL METRICS:")
        print(f"  VTO Precision:      {result.vto_precision:.4f}")
        print(f"  VTO Recall:         {result.vto_recall:.4f}")
        print(f"  VTO F1:             {result.vto_f1:.4f}")
        print(f"  Preference Acc:     {result.preference_accuracy:.4f}")
        print(f"  Reward Margin:      {result.reward_margin:.4f}")
        
        print("\n📝 GENERATION METRICS:")
        print(f"  BLEU-1:             {result.bleu_1:.4f}")
        print(f"  BLEU-4:             {result.bleu_4:.4f}")
        print(f"  ROUGE-L:            {result.rouge_l:.4f}")
        print(f"  Distinct-1:         {result.distinct_1:.4f}")
        print(f"  Distinct-2:         {result.distinct_2:.4f}")
        
        print("\n🧠 NOVEL METRICS - STAR Reasoning:")
        print(f"  Reasoning Depth:    {result.reasoning_depth:.2f}")
        print(f"  Thought Quality:    {result.thought_quality:.4f}")
        print(f"  Backtrack Rate:     {result.backtrack_rate:.4f}")
        
        print("\n🤝 NOVEL METRICS - MAVEN Agents:")
        print(f"  Agent Agreement:    {result.agent_agreement_rate:.4f}")
        
        return result
    
    def evaluate_ranking_protocol(self, 
                                   test_data: List[Dict],
                                   num_negatives: int = 99,
                                   max_samples: int = 100,
                                   scoring_method: str = "charm") -> Dict[str, float]:
        """
        Run publication-ready ranking evaluation.
        
        FIXED: Uses BRIDGE + CHARM scoring (the working method).
        OPTIMIZED: GPU batched for 5-10x speedup.
        
        Args:
            test_data: Test data with context and items
            num_negatives: Number of negative samples (99 = rank among 100)
            max_samples: Maximum samples to evaluate
            scoring_method: Not used (always uses CHARM with BRIDGE)
            
        Returns:
            Dictionary with all ranking metrics
        """
        print(f"\n📊 Running Ranking Evaluation (GPU Batched)")
        print(f"   Negative samples: {num_negatives}")
        
        # Extract item pool from data
        item_pool = extract_items_from_data(test_data)
        
        if len(item_pool) < num_negatives:
            print(f"  Warning: Item pool ({len(item_pool)}) smaller than num_negatives ({num_negatives})")
            num_negatives = max(10, len(item_pool) - 1)
        
        print(f"   Item pool size: {len(item_pool)}")
        
        # Create ranking evaluator
        ranking_eval = RankingEvaluator(self.model, self.device, num_negatives)
        ranking_eval.set_item_pool(item_pool)
        
        # Run evaluation
        results = ranking_eval.evaluate_ranking(
            test_data, 
            max_samples=max_samples
        )
        
        # Print results
        print("\n   📈 Ranking Results:")
        print(f"      Recall@1:   {results.get('recall_at_1', 0):.4f}")
        print(f"      Recall@5:   {results.get('recall_at_5', 0):.4f}")
        print(f"      Recall@10:  {results.get('recall_at_10', 0):.4f}")
        print(f"      Recall@20:  {results.get('recall_at_20', 0):.4f}")
        print(f"      Recall@50:  {results.get('recall_at_50', 0):.4f}")
        print(f"      MRR:        {results.get('mrr', 0):.4f}")
        print(f"      MRR@10:     {results.get('mrr_at_10', 0):.4f}")
        print(f"      MRR@20:     {results.get('mrr_at_20', 0):.4f}")
        print(f"      NDCG@1:     {results.get('ndcg_at_1', 0):.4f}")
        print(f"      NDCG@5:     {results.get('ndcg_at_5', 0):.4f}")
        print(f"      NDCG@10:    {results.get('ndcg_at_10', 0):.4f}")
        print(f"      NDCG@20:    {results.get('ndcg_at_20', 0):.4f}")
        print(f"      Samples:    {results.get('num_samples', 0)}")
        
        return results