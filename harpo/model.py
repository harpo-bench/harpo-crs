"""
HARPO: Unified Model Architecture

Optimizing Conversational Recommendation for User-Aligned Quality 
via Hierarchical Preference Learning

Integrates all novel components:
- CHARM: Contrastive Hierarchical Alignment with Reward Marginalization
- STAR: Structured Tree-of-thought Agentic Reasoning
- BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
- MAVEN: Multi-Agent Virtual Environment for Recommendations

Primary Objective: USER-ALIGNED RECOMMENDATION QUALITY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import math
import heapq

from .pooling import masked_mean_pool
from .retrieval import CrossEncoderReranker, TwoTowerRetriever
from .config import (
    VTO, Domain, AgentRole, ModelConfig, TrainingConfig,
    STARConfig, CHARMConfig, BRIDGEConfig, MAVENConfig, RetrievalConfig,
    SPECIAL_TOKENS, get_domain_token
)


@dataclass
class ModelOutput:
    """Unified output format"""
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    vto_logits: Optional[torch.Tensor] = None
    reward_scores: Optional[Dict[str, torch.Tensor]] = None
    recommendation_scores: Optional[torch.Tensor] = None
    reasoning_path: Optional[List] = None
    context_pooled: Optional[torch.Tensor] = None
    
    def to_dict(self):
        """Convert to dictionary for DataParallel compatibility"""
        return {
            'logits': self.logits,
            'loss': self.loss,
            'hidden_states': self.hidden_states,
            'vto_logits': self.vto_logits,
            'reward_scores': self.reward_scores,
            'recommendation_scores': self.recommendation_scores,
            'reasoning_path': self.reasoning_path,
            'context_pooled': self.context_pooled
        }
    
    @classmethod
    def from_dict(cls, d):
        """Create ModelOutput from dictionary"""
        return cls(**d)


# ============================================================================
# BRIDGE: Domain Adaptation Module
# ============================================================================

class GradientReversal(torch.autograd.Function):
    """Gradient reversal for adversarial domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class BRIDGE(nn.Module):
    """
    BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
    
    Creates domain-invariant representations while preserving task-specific info.
    Key innovations:
    - Adversarial training with domain-specific gates
    - Contrastive learning for cross-domain alignment (NEW)
    - Cross-attention fusion for recommendation relevance (NEW)
    """
    
    def __init__(self, hidden_size: int, config: BRIDGEConfig, 
                 num_domains: int = 6, num_vtos: int = 24):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Multi-head projection
        self.head_dim = hidden_size // config.num_projection_heads
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.GELU()
            ) for _ in range(config.num_projection_heads)
        ])
        self.head_combiner = nn.Linear(hidden_size, hidden_size)
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        # Task preserver (VTO prediction)
        self.task_preserver = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_vtos)
        )
        
        # Domain gates
        # One row per domain so a batch spanning several domains can be gated
        # per sample. The old ParameterDict keyed by Domain.value could only
        # apply a single domain to an entire batch.
        if config.use_domain_gates:
            self.domain_gates = nn.Parameter(
                torch.full((num_domains, hidden_size), float(config.gate_init)))
        
        # NEW: Contrastive projection for cross-domain alignment
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        self.contrastive_temp = nn.Parameter(torch.tensor(0.07))  # Learnable temperature
        
        # NEW: Cross-attention for recommendation-aware feature fusion
        self.cross_attn_query = nn.Linear(hidden_size, hidden_size // 4)
        self.cross_attn_key = nn.Linear(hidden_size, hidden_size // 4)
        self.cross_attn_value = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_out = nn.Linear(hidden_size, hidden_size)
        
        self.output_norm = nn.LayerNorm(hidden_size)
    
    @staticmethod
    def _as_domain_ids(domain: Union[Domain, torch.Tensor],
                       batch_size: int, device) -> torch.Tensor:
        """Normalise a Domain enum or an id tensor to a ``[batch]`` LongTensor."""
        if isinstance(domain, torch.Tensor):
            ids = domain.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1 and batch_size > 1:
                ids = ids.expand(batch_size)
            return ids
        return torch.full((batch_size,), list(Domain).index(domain),
                          dtype=torch.long, device=device)

    def forward(self, hidden_states: torch.Tensor,
                domain: Union[Domain, torch.Tensor],
                alpha: float = 1.0, vto_labels: Optional[torch.Tensor] = None,
                enable_contrastive: bool = True,
                attention_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for domain adaptation.
        
        Returns domain-invariant features and auxiliary losses.
        NEW: Includes contrastive projection and cross-attention fusion.
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Masked, so padding cannot leak into the representation.
        pooled = masked_mean_pool(hidden_states, attention_mask)
        batch_size = pooled.size(0)
        domain_ids = self._as_domain_ids(domain, batch_size, device)
        
        # Multi-head projection
        head_outputs = [head(pooled) for head in self.projection_heads]
        combined = torch.cat(head_outputs, dim=-1)
        projected = self.head_combiner(combined)
        
        # Apply domain gate
        if self.config.use_domain_gates:
            gate = torch.sigmoid(self.domain_gates[domain_ids])
            domain_invariant = gate * projected + (1 - gate) * pooled
        else:
            domain_invariant = projected
        
        # NEW: Cross-attention fusion for recommendation-aware features
        # Query from domain-invariant, Key/Value from original
        q = self.cross_attn_query(domain_invariant)
        k = self.cross_attn_key(pooled)
        v = self.cross_attn_value(pooled)
        
        # Scaled dot-product attention
        attn_scale = (q.size(-1)) ** 0.5
        attn_weights = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / attn_scale, dim=-1)
        attended = attn_weights * v
        fused = domain_invariant + 0.1 * self.cross_attn_out(attended)  # Residual connection
        
        domain_invariant = self.output_norm(fused)
        
        outputs = {"features": domain_invariant}
        
        # Domain confusion loss
        if self.config.gradient_reversal:
            reversed_features = gradient_reversal(domain_invariant, alpha)
        else:
            reversed_features = domain_invariant
        
        domain_logits = self.domain_discriminator(reversed_features)
        outputs["domain_loss"] = F.cross_entropy(domain_logits, domain_ids)
        
        # NEW: Contrastive projection for cross-domain alignment
        if enable_contrastive and batch_size > 1:
            contrastive_embeds = self.contrastive_proj(domain_invariant)
            contrastive_embeds = F.normalize(contrastive_embeds, dim=-1)
            outputs["contrastive_embeds"] = contrastive_embeds
            
            # InfoNCE-style loss (within-batch negatives)
            sim_matrix = torch.mm(contrastive_embeds, contrastive_embeds.t()) / self.contrastive_temp.clamp(min=0.01)
            labels = torch.arange(batch_size, device=device)
            outputs["contrastive_loss"] = F.cross_entropy(sim_matrix, labels)
        
        # Task preservation loss
        if vto_labels is not None:
            vto_logits = self.task_preserver(domain_invariant)
            outputs["task_loss"] = F.binary_cross_entropy_with_logits(vto_logits, vto_labels.float())
            outputs["vto_logits"] = vto_logits
        
        return outputs


# ============================================================================
# STAR: Tree-of-Thought Reasoning Module
# ============================================================================

@dataclass
class ReasoningState:
    """State in reasoning tree - tracks recommendation quality"""
    hidden_state: torch.Tensor
    thought: str
    vto_predictions: torch.Tensor
    value: float  # Predicted recommendation quality
    depth: int
    parent: Optional['ReasoningState'] = None
    children: List['ReasoningState'] = field(default_factory=list)
    item_scores: Dict[int, float] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    
    def __lt__(self, other):
        return self.value > other.value


class ValueNetwork(nn.Module):
    """
    Value network predicting RECOMMENDATION QUALITY (not just action correctness).
    
    Predicts: relevance, diversity, user satisfaction, engagement
    """
    
    def __init__(self, hidden_size: int, num_vtos: int = 24, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-component value (recommendation quality dimensions)
        self.value_heads = nn.ModuleDict({
            "relevance": nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            "diversity": nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            "user_satisfaction": nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            "engagement": nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # Learnable weights for combining
        self.value_weights = nn.Parameter(torch.tensor([0.35, 0.15, 0.35, 0.15]))
    
    def forward(self, hidden_state: torch.Tensor,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        
        encoded = self.state_encoder(hidden_state)
        
        components = {}
        values = []
        for name, head in self.value_heads.items():
            val = head(encoded).squeeze(-1)
            components[name] = val
            values.append(val)
        
        stacked = torch.stack(values, dim=-1)
        weights = F.softmax(self.value_weights, dim=0)
        total_value = (stacked * weights).sum(dim=-1)
        
        result = {"value": total_value, "weights": weights}
        if return_components:
            result.update(components)
        return result


class ThoughtGenerator(nn.Module):
    """Generates reasoning steps with VTO predictions"""
    
    def __init__(self, hidden_size: int, num_vtos: int = 24, 
                 max_candidates: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vtos = num_vtos
        self.max_candidates = max_candidates
        
        self.context_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Generate multiple candidates
        self.candidate_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * max_candidates),
            nn.LayerNorm(hidden_size * max_candidates),
            nn.GELU()
        )
        
        self.vto_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_vtos)
        )
        
        self.quality_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor, 
                num_candidates: int = 2) -> Dict[str, torch.Tensor]:
        batch_size = context.size(0) if context.dim() > 1 else 1
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        num_candidates = min(num_candidates, self.max_candidates)
        
        processed = self.context_processor(context)
        all_candidates = self.candidate_generator(processed)
        all_candidates = all_candidates.view(batch_size, self.max_candidates, self.hidden_size)
        candidates = all_candidates[:, :num_candidates, :]
        
        vto_logits = self.vto_predictor(candidates)
        quality_scores = self.quality_estimator(candidates).squeeze(-1)
        
        return {
            "thought_states": candidates,
            "vto_logits": vto_logits,
            "quality_scores": quality_scores
        }


class STAR(nn.Module):
    """
    STAR: Structured Tree-of-Thought Agentic Reasoning
    
    Tree search optimized for RECOMMENDATION QUALITY:
    - Value network predicts user satisfaction, not just accuracy
    - Paths evaluated on expected recommendation quality
    - Backtracking when paths lead to poor recommendations
    
    NEW IMPROVEMENTS:
    - Residual connections for better gradient flow
    - Uncertainty estimation for robust reasoning
    - Confidence-weighted path aggregation
    """
    
    def __init__(self, hidden_size: int, config: STARConfig,
                 num_vtos: int = 24, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.num_vtos = num_vtos
        
        self.value_network = ValueNetwork(hidden_size, num_vtos, dropout)
        self.thought_generator = ThoughtGenerator(hidden_size, num_vtos,
                                                   config.branching_factor, dropout)
        self.state_transition = nn.GRUCell(hidden_size, hidden_size)
        
        # Aggregator for final recommendation
        self.rec_aggregator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # NEW: Uncertainty estimator for robust reasoning
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Outputs uncertainty in [0, 1]
        )
        
        # NEW: Path confidence aggregator
        self.path_confidence = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # NEW: Residual gate for controlled information flow
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def create_root(self, context: torch.Tensor) -> ReasoningState:
        if context.dim() > 1:
            context = context.squeeze(0)
        
        value_result = self.value_network(context)
        return ReasoningState(
            hidden_state=context,
            thought="[START] Beginning recommendation reasoning",
            vto_predictions=torch.zeros(self.num_vtos, device=context.device),
            value=value_result["value"].item(),
            depth=0
        )
    
    def expand_node(self, state: ReasoningState) -> List[ReasoningState]:
        if state.depth >= self.config.max_depth:
            return []
        
        hidden = state.hidden_state.unsqueeze(0)
        gen_result = self.thought_generator(hidden, self.config.branching_factor)
        
        children = []
        for i in range(gen_result["thought_states"].size(1)):
            child_hidden = gen_result["thought_states"][0, i]
            child_vto = gen_result["vto_logits"][0, i]
            
            new_hidden = self.state_transition(child_hidden.unsqueeze(0), hidden).squeeze(0)
            value_result = self.value_network(new_hidden)
            
            child = ReasoningState(
                hidden_state=new_hidden,
                thought=f"[STEP {state.depth + 1}] Reasoning step {i + 1}",
                vto_predictions=child_vto,
                value=value_result["value"].item(),
                depth=state.depth + 1,
                parent=state
            )
            children.append(child)
            state.children.append(child)
        
        return children
    
    def search_beam(self, root: ReasoningState) -> ReasoningState:
        """Beam search for best recommendation path"""
        beam = [root]
        
        for _ in range(self.config.max_depth):
            all_children = []
            for state in beam:
                children = self.expand_node(state)
                all_children.extend(children)
            
            if not all_children:
                break
            
            beam = sorted(all_children, key=lambda x: x.value, reverse=True)[:self.config.beam_width]
        
        return max(beam, key=lambda x: x.value) if beam else root
    
    def forward(self, context: torch.Tensor,
                return_path: bool = False) -> Dict[str, Any]:
        if context.dim() == 2:
            context = context[0]
        
        root = self.create_root(context)
        final_state = self.search_beam(root)
        
        # Get reasoning path
        path = []
        current = final_state
        while current is not None:
            path.append(current)
            current = current.parent
        path = list(reversed(path))
        
        # Aggregate for final response with NEW residual gating
        if len(path) > 1:
            combined = torch.cat([root.hidden_state, final_state.hidden_state], dim=-1)
            base_aggregated = self.rec_aggregator(combined)
            
            # NEW: Residual gate for controlled mixing
            gate = self.residual_gate(combined)
            aggregated = gate * base_aggregated + (1 - gate) * context  # Residual connection
            
            # NEW: Compute path confidence
            confidence = self.path_confidence(combined)
        else:
            aggregated = final_state.hidden_state
            confidence = torch.tensor([0.5], device=context.device)
        
        # NEW: Estimate uncertainty
        uncertainty = self.uncertainty_head(aggregated if aggregated.dim() == 1 else aggregated.unsqueeze(0))
        
        result = {
            "final_hidden": aggregated,
            "vto_predictions": final_state.vto_predictions,
            "recommendation_value": final_state.value,
            "reasoning_depth": len(path),
            "uncertainty": uncertainty.item() if uncertainty.numel() == 1 else uncertainty.mean().item(),
            "path_confidence": confidence.item() if confidence.numel() == 1 else confidence.mean().item()
        }
        
        if return_path:
            result["reasoning_path"] = path
        
        return result


# ============================================================================
# CHARM: Hierarchical Preference Learning Module
# ============================================================================

class RewardHead(nn.Module):
    """Individual reward head for one quality dimension
    
    FIXED: Now outputs normalized scores in range [-1, 1] via tanh
    This prevents negative unbounded values during evaluation.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # FIXED: Bound outputs to [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class MetaLearner(nn.Module):
    """Learns to weight reward components based on context and domain"""
    
    def __init__(self, hidden_size: int, num_domains: int = 6, 
                 num_rewards: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_rewards = num_rewards
        
        self.domain_embedding = nn.Embedding(num_domains, hidden_size // 4)
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_rewards)
        )
        
        self.baseline_weights = nn.Parameter(torch.tensor([0.30, 0.20, 0.30, 0.20]))
    
    def forward(self, context: torch.Tensor,
                domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = context.size(0)
        device = context.device
        dtype = context.dtype  # CRITICAL FIX: Match input dtype
        
        ctx_encoded = self.context_encoder(context)
        
        if domain_id is not None:
            domain_emb = self.domain_embedding(domain_id)
        else:
            # CRITICAL FIX: Create zeros with same dtype as input
            domain_emb = torch.zeros(batch_size, ctx_encoded.size(-1), device=device, dtype=dtype)
        
        combined = torch.cat([ctx_encoded, domain_emb], dim=-1)
        weight_logits = self.weight_predictor(combined)
        
        # CRITICAL FIX: Ensure baseline_weights matches dtype
        baseline = self.baseline_weights.to(dtype=dtype)
        return F.softmax(weight_logits + baseline, dim=-1)


class CHARM(nn.Module):
    """
    CHARM: Contrastive Hierarchical Alignment with Reward Marginalization
    
    Optimizes for RECOMMENDATION QUALITY through:
    - Decomposed rewards: relevance, diversity, satisfaction, engagement
    - Meta-learned weights adapting to domain and context
    - Adaptive margin for robust preference learning
    """
    
    def __init__(self, hidden_size: int, config: CHARMConfig,
                 num_vtos: int = 24, num_domains: int = 6, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.beta = config.beta
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Hierarchical reward heads (recommendation quality dimensions)
        self.reward_heads = nn.ModuleDict({
            "relevance": RewardHead(hidden_size, dropout),
            "diversity": RewardHead(hidden_size, dropout),
            "user_satisfaction": RewardHead(hidden_size, dropout),
            "engagement": RewardHead(hidden_size, dropout)
        })
        
        self.meta_learner = MetaLearner(hidden_size, num_domains, 4, dropout)
        
        # Adaptive margin
        self.margin_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.base_margin = nn.Parameter(torch.tensor(0.5))
    
    def compute_rewards(self, hidden_states: torch.Tensor,
                        domain_id: Optional[torch.Tensor] = None,
                        return_components: bool = True) -> Dict[str, torch.Tensor]:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)
        
        # CRITICAL FIX: Ensure consistent dtype throughout
        dtype = hidden_states.dtype
        
        features = self.feature_extractor(hidden_states)
        
        rewards = {}
        reward_list = []
        for name, head in self.reward_heads.items():
            r = head(features)
            rewards[name] = r
            reward_list.append(r)
        
        stacked = torch.stack(reward_list, dim=-1)
        weights = self.meta_learner(features, domain_id)
        
        # CRITICAL FIX: Ensure weights and stacked have same dtype
        weights = weights.to(dtype=dtype)
        total = (stacked * weights).sum(dim=-1)
        
        result = {"total": total, "weights": weights, "features": features}
        if return_components:
            result.update(rewards)
        return result
    
    def compute_preference_loss(self, chosen_hidden: torch.Tensor,
                                 rejected_hidden: torch.Tensor,
                                 domain_id: Optional[torch.Tensor] = None
                                ) -> Dict[str, torch.Tensor]:
        chosen_rewards = self.compute_rewards(chosen_hidden, domain_id)
        rejected_rewards = self.compute_rewards(rejected_hidden, domain_id)
        
        reward_diff = chosen_rewards["total"] - rejected_rewards["total"]
        
        # Adaptive margin
        combined = torch.cat([chosen_rewards["features"], rejected_rewards["features"]], dim=-1)
        margin = self.base_margin + 0.5 * self.margin_net(combined).squeeze(-1)
        
        # Bradley-Terry loss with margin
        loss = -F.logsigmoid(self.beta * (reward_diff - margin)).mean()
        
        return {
            "total_loss": loss,
            "preference_loss": loss,
            "margin": margin.mean(),
            "reward_diff": reward_diff.mean(),
            "chosen_reward": chosen_rewards["total"].mean(),
            "rejected_reward": rejected_rewards["total"].mean()
        }
    
    def forward(self, hidden_states: torch.Tensor,
                domain_id: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:  # <--- Add **kwargs
        
        # Pass **kwargs (like return_components=True) to compute_rewards
        return self.compute_rewards(hidden_states, domain_id, **kwargs)


# ============================================================================
# MAVEN: Multi-Agent Module
# ============================================================================

class AgentModule(nn.Module):
    """Individual agent in MAVEN framework"""
    
    def __init__(self, hidden_size: int, role: AgentRole, dropout: float = 0.1):
        super().__init__()
        self.role = role
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(context)
        output = self.output_head(encoded)
        quality = self.quality_scorer(encoded)
        return {"output": output, "quality": quality.squeeze(-1)}


class MAVEN(nn.Module):
    """
    MAVEN: Multi-Agent Virtual Environment for Recommendations
    
    Multiple specialized agents collaborate for better recommendations:
    - Recommender: generates candidates
    - Critic: evaluates quality  
    - Explainer: provides explanations
    
    NEW IMPROVEMENTS:
    - Attention-based agent weighting for dynamic collaboration
    - Consensus mechanism for robust predictions
    - Iterative refinement through agent communication
    """
    
    def __init__(self, hidden_size: int, config: MAVENConfig, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        
        # Create agents
        self.agents = nn.ModuleDict({
            role.value: AgentModule(hidden_size, role, dropout)
            for role in config.agent_roles
        })
        
        # Orchestrator for combining outputs
        self.orchestrator = nn.Sequential(
            nn.Linear(hidden_size * len(config.agent_roles), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Agreement scorer
        self.agreement_scorer = nn.Sequential(
            nn.Linear(hidden_size * len(config.agent_roles), hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # NEW: Attention-based agent weighting
        self.agent_attention_query = nn.Linear(hidden_size, hidden_size // 4)
        self.agent_attention_key = nn.Linear(hidden_size, hidden_size // 4)
        self.agent_weight_mlp = nn.Sequential(
            nn.Linear(len(config.agent_roles), len(config.agent_roles) * 2),
            nn.GELU(),
            nn.Linear(len(config.agent_roles) * 2, len(config.agent_roles)),
            nn.Softmax(dim=-1)
        )
        
        # NEW: Consensus mechanism for robust predictions
        self.consensus_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # NEW: Iterative refinement layer
        self.refinement_layer = nn.GRUCell(hidden_size, hidden_size)
    
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        batch_size = context.size(0)
        
        # Get outputs from all agents
        agent_outputs = {}
        agent_qualities = {}
        all_outputs = []
        
        for role_name, agent in self.agents.items():
            result = agent(context)
            agent_outputs[role_name] = result["output"]
            agent_qualities[role_name] = result["quality"]
            all_outputs.append(result["output"])
        
        # Stack agent outputs for attention: [batch, num_agents, hidden]
        stacked_outputs = torch.stack(all_outputs, dim=1)
        
        # NEW: Attention-based agent weighting
        # Query from context, Key from agent outputs
        query = self.agent_attention_query(context)  # [batch, hidden/4]
        keys = self.agent_attention_key(stacked_outputs)  # [batch, num_agents, hidden/4]
        
        # Compute attention scores
        attn_scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # [batch, num_agents]
        
        # Combine with quality scores for final weights
        qualities = torch.stack(list(agent_qualities.values()), dim=-1)  # [batch, num_agents]
        combined_scores = attn_scores + qualities
        
        # Learnable weight refinement
        refined_weights = self.agent_weight_mlp(combined_scores)  # [batch, num_agents]
        
        # Weighted sum of agent outputs
        weighted_outputs = (stacked_outputs * refined_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]
        
        # Standard orchestrator output
        combined = torch.cat(all_outputs, dim=-1)
        orchestrator_output = self.orchestrator(combined)
        
        # NEW: Consensus mechanism - gate between weighted and orchestrator
        consensus_input = torch.cat([weighted_outputs, orchestrator_output], dim=-1)
        consensus_gate = self.consensus_gate(consensus_input)
        final_output = consensus_gate * weighted_outputs + (1 - consensus_gate) * orchestrator_output
        
        # NEW: Optional iterative refinement
        if self.config.max_communication_rounds > 1:
            for _ in range(self.config.max_communication_rounds - 1):
                final_output = self.refinement_layer(weighted_outputs, final_output)
        
        agreement = self.agreement_scorer(combined)
        
        return {
            "output": final_output,
            "agent_outputs": agent_outputs,
            "agent_qualities": agent_qualities,
            "agreement": agreement.squeeze(-1),
            "weights": refined_weights,
            "attention_scores": attn_scores
        }


# ============================================================================
# HARPO-MT v2: Unified Model
# ============================================================================

class HARPOMTv2(nn.Module):
    """
    HARPO-MT v2: Complete Model Architecture
    
    Integrates:
    - Base LLM (Qwen 0.5B for Mac)
    - BRIDGE: Domain adaptation
    - STAR: Tree-of-thought reasoning
    - CHARM: Hierarchical preference learning
    - MAVEN: Multi-agent collaboration
    
    Primary objective: RECOMMENDATION QUALITY
    """
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        
        self.base_model = None
        self.tokenizer = None
        self.device = None
        self._use_dataparallel_output = False  # Flag for DataParallel compatibility
        self._model_dtype = torch.float32  # CRITICAL FIX: Track model dtype
        
        hidden_size = model_config.hidden_size
        num_vtos = len(VTO)
        num_domains = len(Domain)
        
        # Novel components
        self.bridge = BRIDGE(hidden_size, training_config.bridge_config, num_domains, num_vtos)
        self.star = STAR(hidden_size, training_config.star_config, num_vtos)
        self.charm = CHARM(hidden_size, training_config.charm_config, num_vtos, num_domains)
        self.maven = MAVEN(hidden_size, training_config.maven_config)
        
        # VTO prediction head
        self.vto_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_vtos)
        )
        
        # Recommendation head (predicts item relevance)
        self.recommendation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Item retriever: the missing contrastive objective.
        retrieval_config = getattr(training_config, "retrieval_config", None)
        self.retrieval_config = retrieval_config
        if retrieval_config is not None and retrieval_config.enabled:
            self.retriever = TwoTowerRetriever(
                hidden_size, embed_dim=retrieval_config.embed_dim,
                dropout=retrieval_config.dropout,
                init_temperature=retrieval_config.temperature,
            )
            if retrieval_config.item_representation in ("id", "hybrid"):
                self.item_id_embedding = nn.Embedding(
                    retrieval_config.max_catalog_size, retrieval_config.embed_dim
                )
                nn.init.normal_(self.item_id_embedding.weight, std=0.02)
            else:
                self.item_id_embedding = None
            # Zero until a catalogue is attached, when it is set to log popularity.
            if getattr(retrieval_config, "item_bias", False):
                self.item_bias = nn.Embedding(retrieval_config.max_catalog_size, 1)
                nn.init.zeros_(self.item_bias.weight)
            else:
                self.item_bias = None
            self.reranker = (CrossEncoderReranker(hidden_size, retrieval_config.dropout)
                             if retrieval_config.use_reranker else None)
        else:
            self.retriever = None
            self.item_id_embedding = None
            self.item_bias = None
            self.reranker = None

    def encode_item_embeddings(self, item_pooled: torch.Tensor,
                               item_indices: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """Item embeddings, fusing the learned id table with the text tower.

        Items outside the table (index < 0) fall back to content only rather
        than erroring.
        """
        if self.retriever is None:
            raise RuntimeError("retrieval is disabled in this configuration")

        embeds = self.retriever.encode_item(item_pooled)
        cfg = self.retrieval_config
        if self.item_id_embedding is None or item_indices is None:
            return embeds
        if cfg.item_representation == "content":
            return embeds

        known = (item_indices >= 0) & (item_indices < self.item_id_embedding.num_embeddings)
        if not bool(known.any()):
            return embeds

        safe = item_indices.clamp(min=0, max=self.item_id_embedding.num_embeddings - 1)
        id_embeds = F.normalize(self.item_id_embedding(safe), dim=-1)

        if cfg.item_representation == "id":
            fused = torch.where(known.unsqueeze(-1), id_embeds, embeds)
        else:
            w = cfg.id_embedding_weight
            fused = torch.where(known.unsqueeze(-1),
                                (1.0 - w) * embeds + w * id_embeds, embeds)
        return F.normalize(fused, dim=-1)

    @staticmethod
    def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean NLL per example, ``[batch]``.

        ``outputs.loss`` is already reduced over the batch, so it cannot be a
        per-example signal; broadcasting it gave every row the same target.
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
        valid = (shift_labels != -100).to(nll.dtype)
        return (nll * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    
    def load_base_model(self, device: str = None):
        """Load base LLM with LoRA, Flash Attention 2, and proper special token handling
        
        CRITICAL FIXES for resume/offline mode:
        - Detects if loading from local checkpoint
        - Uses local_files_only=True for offline environments
        - Handles tokenizer and embedding size correctly
        - Doesn't re-add special tokens when resuming
        
        OPTIMIZED FOR PUBLICATION:
        - Flash Attention 2 for 2-3x speedup
        - Proper gradient checkpointing
        - Efficient memory usage for 7B model on 2x A100
        """
        import os
        import json
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        model_path = self.model_config.model_name
        
        # ===== Detect if loading from local checkpoint =====
        is_local_path = os.path.isdir(model_path)
        is_lora_checkpoint = is_local_path and os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        print(f"Loading base model: {model_path}")
        print(f"Device: {device}")
        if is_local_path:
            print(f"✓ Loading from LOCAL path (offline mode enabled)")
        if is_lora_checkpoint:
            print(f"✓ Detected LoRA adapter checkpoint")
        
        # ===== Set offline mode for HuggingFace =====
        # Only when genuinely loading from disk: hard-coding these made the
        # documented "fresh start (requires internet)" branch impossible.
        if is_local_path:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # ===== Determine dtype and check for Flash Attention 2 =====
        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        
        # Check Flash Attention 2 availability
        use_flash_attn = False
        if device.startswith("cuda") and getattr(self.model_config, 'use_flash_attention', True):
            try:
                import flash_attn
                use_flash_attn = True
                print("=" * 50)
                print("✓ FLASH ATTENTION 2 ENABLED - 2-3x speedup!")
                print("=" * 50)
            except ImportError:
                print("=" * 50)
                print("⚠ Flash Attention 2 NOT installed")
                print("  Install with: pip install flash-attn --no-build-isolation")
                print("  Training will work but be SLOWER (no flash attention)")
                print("=" * 50)
        
        # ===== Load tokenizer =====
        # A local path is not necessarily a HARPO checkpoint: a plain model
        # directory (a hub snapshot copied to a cluster node) has no special
        # tokens. Assuming it did skipped them entirely, so the same data was
        # tokenized differently depending on where the weights came from. The
        # missing-token check below is a no-op for checkpoints that have them.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=is_local_path
        )
        # Dialogues overflow from the start: the latest turns carry the request,
        # and right-truncation also cut the reply off 15% of SFT examples.
        self.tokenizer.truncation_side = "left"

        # ===== Add ALL special tokens used in training data =====
        special_tokens_to_add = [
            "<|vto_start|>", "<|vto_end|>",
            "<|tool_start|>", "<|tool_end|>",
            "<|think|>", "<|/think|>",
            "<|thought|>", "<|/thought|>",
            "<|response|>", "<|/response|>",
            "<|domain:fashion|>", "<|domain:movies|>", 
            "<|domain:electronics|>", "<|domain:general|>",
            "<|domain:food|>", "<|domain:books|>",
            "<|agent:recommender|>", "<|agent:critic|>", 
            "<|agent:explainer|>", "<|agent:orchestrator|>",
        ]

        # Check which tokens are truly new
        new_tokens = []
        for token in special_tokens_to_add:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)

        if new_tokens:
            num_added = self.tokenizer.add_special_tokens({
                'additional_special_tokens': new_tokens
            })
            print(f"✓ Added {num_added} special tokens to tokenizer")

        # Add pad token if missing (for both cases)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # ===== Load model =====
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "local_files_only": is_local_path,
        }
        
        # Add Flash Attention 2 if available
        if use_flash_attn:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        print(f"Loading model (this may take a moment for 7B params)...")
        
        if is_lora_checkpoint:
            # ===== CRITICAL: Loading from LoRA checkpoint requires special handling =====
            # 1. First load the BASE model from cache
            # 2. Resize embeddings to match tokenizer
            # 3. Then load LoRA adapter
            
            # Get the base model name from adapter config
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            
            print(f"Loading base model from cache: {base_model_name}")
            
            # Load base model from cache
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **load_kwargs
            )
            
            # Move to device BEFORE resizing
            if device.startswith("cuda"):
                target_device = device
                self.base_model = self.base_model.to(target_device)
            elif device == "mps":
                target_device = device
                self.base_model = self.base_model.to(device)
            else:
                target_device = device
            self._target_device = target_device
            
            # Resize embeddings to match tokenizer BEFORE loading LoRA
            old_vocab_size = self.base_model.get_input_embeddings().weight.shape[0]
            new_vocab_size = len(self.tokenizer)
            
            if old_vocab_size != new_vocab_size:
                self.base_model.resize_token_embeddings(new_vocab_size)
                print(f"✓ Resized embeddings: {old_vocab_size} -> {new_vocab_size}")
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {model_path}")
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                model_path,
                is_trainable=True,
                local_files_only=True
            )
            print(f"✓ Loaded LoRA adapter successfully")
            
            # Don't apply LoRA again - already loaded
            self._lora_applied = True
            
        else:
            # Fresh start or full model checkpoint
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Move to device
            if device.startswith("cuda"):
                target_device = device
                self.base_model = self.base_model.to(target_device)
            elif device == "mps":
                target_device = device
                self.base_model = self.base_model.to(device)
            else:
                target_device = device
            self._target_device = target_device
            
            # ===== Resize embeddings for new tokens =====
            if new_tokens:
                old_embeddings_size = self.base_model.get_input_embeddings().weight.shape[0]
                self.base_model.resize_token_embeddings(len(self.tokenizer))
                new_embeddings_size = self.base_model.get_input_embeddings().weight.shape[0]
                print(f"✓ Resized embeddings: {old_embeddings_size} -> {new_embeddings_size}")

                # Initialise the rows of the tokens actually added, by id rather
                # than by the change in matrix size. The size delta is the wrong
                # signal: Qwen2.5 pads its vocabulary to 151936 while holding
                # 151665 real tokens, so adding 20 tokens *shrinks* the matrix to
                # 151685 and range(old, new) is empty -- the old loop printed
                # success while initialising nothing, and the new ids silently
                # inherited unused reserved rows.
                new_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
                new_ids = [i for i in new_ids
                           if isinstance(i, int) and 0 <= i < new_embeddings_size]

                with torch.no_grad():
                    in_w = self.base_model.get_input_embeddings().weight
                    keep = torch.ones(new_embeddings_size, dtype=torch.bool)
                    if new_ids:
                        keep[torch.tensor(new_ids, dtype=torch.long)] = False
                    mean_embedding = in_w[keep].mean(dim=0)
                    for i in new_ids:
                        in_w[i] = (mean_embedding
                                   + torch.randn_like(mean_embedding) * 0.02).to(in_w.dtype)

                    # The untied output head needs the same treatment, or the
                    # model can never assign probability to the new tokens.
                    out_emb = self.base_model.get_output_embeddings()
                    if out_emb is not None and out_emb.weight is not in_w:
                        out_w = out_emb.weight
                        out_mean = out_w[keep].mean(dim=0)
                        for i in new_ids:
                            out_w[i] = (out_mean
                                        + torch.randn_like(out_mean) * 0.02).to(out_w.dtype)

                print(f"✓ Initialized {len(new_ids)} new token embeddings")
            
            # Apply LoRA for fresh start
            self._lora_applied = False
        
        # Enable gradient checkpointing for memory efficiency
        # CRITICAL FIX: Use use_reentrant=False for DDP compatibility
        if self.training_config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("✓ Gradient checkpointing enabled (non-reentrant mode for DDP)")
        
        # Apply LoRA if not already done
        if not getattr(self, '_lora_applied', False):
            self._apply_lora()
        
        # ===== CRITICAL FIX: Ensure PEFT modules_to_save are on correct device =====
        target_device = getattr(self, '_target_device', device)
        if hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'model'):
            base_transformer = self.base_model.base_model.model
            if hasattr(base_transformer, 'embed_tokens'):
                base_transformer.embed_tokens = base_transformer.embed_tokens.to(target_device)
        
        if hasattr(self.base_model, 'lm_head'):
            self.base_model.lm_head = self.base_model.lm_head.to(target_device)
        elif hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'lm_head'):
            self.base_model.base_model.lm_head = self.base_model.base_model.lm_head.to(target_device)
        
        # Move entire model to ensure all submodules are on correct device
        self.base_model = self.base_model.to(target_device)
        
        # For DataParallel: all components must be on same device as base model
        # For DataParallel: all components must be on same device as base model
        # For Accelerate: each process has its own device
        target_device = getattr(self, '_target_device', "cuda:0" if device.startswith("cuda") else device)
        
        # Move components to device AND convert to same dtype as base model
        self.bridge = self.bridge.to(target_device, dtype=model_dtype)
        self.star = self.star.to(target_device, dtype=model_dtype)
        self.charm = self.charm.to(target_device, dtype=model_dtype)
        self.maven = self.maven.to(target_device, dtype=model_dtype)
        self.vto_head = self.vto_head.to(target_device, dtype=model_dtype)
        self.recommendation_head = self.recommendation_head.to(target_device, dtype=model_dtype)
        # Omitting these left the towers on CPU while the backbone sat on the
        # accelerator, failing on any non-CPU device -- CUDA included.
        # Retrieval heads stay float32 on a bf16 backbone: an AdamW step (~lr)
        # is below bf16 resolution for most of their weights and is rounded
        # away -- a popularity bias near 6 would never move at all.
        for _name in ("retriever", "item_id_embedding", "item_bias", "reranker"):
            _mod = getattr(self, _name, None)
            if _mod is not None:
                setattr(self, _name, _mod.to(target_device, dtype=torch.float32))
        
        # Store dtype for _reinit_components
        self._model_dtype = model_dtype
        
        # Update hidden size from loaded model
        actual_hidden = self.base_model.config.hidden_size
        if actual_hidden != self.model_config.hidden_size:
            print(f"Updating hidden size: {self.model_config.hidden_size} -> {actual_hidden}")
            self.model_config.hidden_size = actual_hidden
            self._reinit_components(actual_hidden)
        
        print(f"✓ Model loaded. Trainable params: {self.count_parameters():,}")
        return self
    
    def _apply_lora(self):
        """Apply LoRA adapters
        
        CRITICAL FIX: Added modules_to_save to train embedding layers.
        Without this, new special tokens (<|think|>, <|tool_start|>, etc.) 
        would remain frozen and produce gibberish!
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.lora_target_modules,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # CRITICAL: Train embeddings for new special tokens!
                # Without this, tokens like <|think|> remain random noise
                # ~40% of a 0.5B model becomes trainable (two 151k x hidden
                # matrices), which dominates wall-clock on memory-bound devices.
                modules_to_save=(None if getattr(self, "_skip_modules_to_save", False)
                                 else ["embed_tokens", "lm_head"])
            )
            
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
            print("✓ LoRA applied with trainable embeddings for special tokens")
            
        except ImportError:
            print("PEFT not available, using full fine-tuning (last 2 layers)")
            for name, param in self.base_model.named_parameters():
                if "layers.23" not in name and "layers.22" not in name:
                    param.requires_grad = False
    
    def _reinit_components(self, hidden_size: int):
        """Reinitialize components with correct hidden size.
        
        CRITICAL FIX: Only reinitialize if dimensions ACTUALLY changed.
        Previously this was destroying learned weights on every call!
        """
        num_vtos = len(VTO)
        num_domains = len(Domain)
        
        # For DataParallel: use cuda:0 specifically
        # For Accelerate: use the process-specific device
        target_device = getattr(self, '_target_device', "cuda:0" if self.device.startswith("cuda") else self.device)
        
        # Use stored dtype (bfloat16 for A100)
        dtype = getattr(self, '_model_dtype', torch.float32)
        
        # CRITICAL FIX: Check if dimensions actually changed before reinitializing
        # This prevents destroying learned weights when loading checkpoints
        bridge_hidden = getattr(self.bridge, 'hidden_size', None)
        if bridge_hidden is None or bridge_hidden != hidden_size:
            print(f"  Reinitializing BRIDGE: {bridge_hidden} -> {hidden_size}")
            self.bridge = BRIDGE(hidden_size, self.training_config.bridge_config, num_domains, num_vtos).to(target_device, dtype=dtype)
        else:
            print(f"  BRIDGE dimensions match ({hidden_size}), preserving weights")
            self.bridge = self.bridge.to(target_device, dtype=dtype)
        
        star_hidden = getattr(self.star, 'hidden_size', None)
        if star_hidden is None or star_hidden != hidden_size:
            print(f"  Reinitializing STAR: {star_hidden} -> {hidden_size}")
            self.star = STAR(hidden_size, self.training_config.star_config, num_vtos).to(target_device, dtype=dtype)
        else:
            print(f"  STAR dimensions match ({hidden_size}), preserving weights")
            self.star = self.star.to(target_device, dtype=dtype)
        
        charm_hidden = getattr(self.charm, 'hidden_size', None)
        if charm_hidden is None or charm_hidden != hidden_size:
            print(f"  Reinitializing CHARM: {charm_hidden} -> {hidden_size}")
            self.charm = CHARM(hidden_size, self.training_config.charm_config, num_vtos, num_domains).to(target_device, dtype=dtype)
        else:
            print(f"  CHARM dimensions match ({hidden_size}), preserving weights")
            self.charm = self.charm.to(target_device, dtype=dtype)
        
        maven_hidden = getattr(self.maven, 'hidden_size', None)
        if maven_hidden is None or maven_hidden != hidden_size:
            print(f"  Reinitializing MAVEN: {maven_hidden} -> {hidden_size}")
            self.maven = MAVEN(hidden_size, self.training_config.maven_config).to(target_device, dtype=dtype)
        else:
            print(f"  MAVEN dimensions match ({hidden_size}), preserving weights")
            self.maven = self.maven.to(target_device, dtype=dtype)
        
        # VTO head - check input dimension
        vto_input_dim = None
        if hasattr(self.vto_head, '__getitem__') or hasattr(self.vto_head, '__iter__'):
            try:
                first_layer = self.vto_head[0]
                if hasattr(first_layer, 'in_features'):
                    vto_input_dim = first_layer.in_features
            except:
                pass
        
        if vto_input_dim is None or vto_input_dim != hidden_size:
            print(f"  Reinitializing vto_head: {vto_input_dim} -> {hidden_size}")
            self.vto_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_vtos)
            ).to(target_device, dtype=dtype)
        else:
            print(f"  vto_head dimensions match ({hidden_size}), preserving weights")
            self.vto_head = self.vto_head.to(target_device, dtype=dtype)
        
        # Recommendation head - check input dimension
        rec_input_dim = None
        if hasattr(self.recommendation_head, '__getitem__') or hasattr(self.recommendation_head, '__iter__'):
            try:
                first_layer = self.recommendation_head[0]
                if hasattr(first_layer, 'in_features'):
                    rec_input_dim = first_layer.in_features
            except:
                pass
        
        if rec_input_dim is None or rec_input_dim != hidden_size:
            print(f"  Reinitializing recommendation_head: {rec_input_dim} -> {hidden_size}")
            self.recommendation_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            ).to(target_device, dtype=dtype)
        else:
            print(f"  recommendation_head dimensions match ({hidden_size}), preserving weights")
            self.recommendation_head = self.recommendation_head.to(target_device, dtype=dtype)

        # Retrieval modules were left out of both the resize and the device move,
        # so they kept ModelConfig's default 3584 regardless of the backbone.
        if getattr(self, "retriever", None) is not None:
            cfg = self.retrieval_config
            if self.retriever.hidden_size != hidden_size:
                print(f"  Reinitializing retriever: {self.retriever.hidden_size} -> {hidden_size}")
                self.retriever = TwoTowerRetriever(
                    hidden_size, embed_dim=cfg.embed_dim, dropout=cfg.dropout,
                    init_temperature=cfg.temperature)
            # float32 regardless of backbone dtype; see load_base_model.
            self.retriever = self.retriever.to(target_device, dtype=torch.float32)
            if self.item_id_embedding is not None:
                self.item_id_embedding = self.item_id_embedding.to(target_device, dtype=torch.float32)
            if getattr(self, "item_bias", None) is not None:
                self.item_bias = self.item_bias.to(target_device, dtype=torch.float32)
            if self.reranker is not None:
                if self.reranker.scorer[0].in_features != hidden_size * 4:
                    self.reranker = CrossEncoderReranker(hidden_size, cfg.dropout)
                self.reranker = self.reranker.to(target_device, dtype=torch.float32)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_for_sft(self):
        """Freeze modules not used during SFT training for DDP compatibility.
        
        CRITICAL FIX: Train BRIDGE + recommendation_head + vto_head during SFT!
        
        During SFT we train:
        - base_model (LoRA adapters)
        - vto_head (VTO prediction)
        - recommendation_head (for ranking - trained via self-supervised signal)
        - BRIDGE (domain adaptation - critical for ranking evaluation)
        
        We freeze: star, maven, charm (these are trained in later stages)
        """
        # Freeze modules for later stages
        for param in self.star.parameters():
            param.requires_grad = False
        for param in self.maven.parameters():
            param.requires_grad = False
        for param in self.charm.parameters():
            param.requires_grad = False
        
        # CRITICAL FIX: Keep these trainable for SFT
        for param in self.vto_head.parameters():
            param.requires_grad = True
        for param in self.recommendation_head.parameters():
            param.requires_grad = True
        for param in self.bridge.parameters():
            param.requires_grad = True
            
        for _mod in (self.retriever, self.item_id_embedding,
                     getattr(self, "item_bias", None), self.reranker):
            if _mod is not None:
                for param in _mod.parameters():
                    param.requires_grad = True

        print("✓ Frozen STAR, MAVEN, CHARM for SFT stage")
        print("✓ Training: base_model + vto_head + recommendation_head + BRIDGE")
    
    def freeze_for_charm(self):
        """Freeze modules not used during CHARM training for DDP compatibility.
        
        CRITICAL FIX: Keep BRIDGE trainable during CHARM for domain-aware preference learning.
        
        During CHARM we train:
        - base_model (LoRA adapters)
        - charm (preference learning)
        - BRIDGE (domain adaptation continues to improve)
        
        We freeze: star, maven, vto_head, recommendation_head
        """
        for param in self.star.parameters():
            param.requires_grad = False
        for param in self.maven.parameters():
            param.requires_grad = False
        for param in self.vto_head.parameters():
            param.requires_grad = False
        for param in self.recommendation_head.parameters():
            param.requires_grad = False
        
        # Keep charm AND bridge trainable
        for param in self.charm.parameters():
            param.requires_grad = True
        for param in self.bridge.parameters():
            param.requires_grad = True
            
        print("✓ Frozen STAR, MAVEN, vto_head, recommendation_head for CHARM stage")
        print("✓ Training: base_model + CHARM + BRIDGE")
    
    def freeze_for_star(self):
        """Freeze modules not used during STAR training.
        
        During STAR, we use: base_model, star
        We freeze: maven, recommendation_head, and optionally others
        """
        for param in self.maven.parameters():
            param.requires_grad = False
        for param in self.recommendation_head.parameters():
            param.requires_grad = False
        for param in self.bridge.parameters():
            param.requires_grad = False
        for param in self.charm.parameters():
            param.requires_grad = False
        for param in self.vto_head.parameters():
            param.requires_grad = False
        # Unfreeze STAR
        for param in self.star.parameters():
            param.requires_grad = True
        print("✓ Frozen all except STAR for STAR stage")
    
    def freeze_for_maven(self):
        """Freeze modules not used during MAVEN training.
        
        During MAVEN, we use: base_model, maven, charm
        We freeze: star, recommendation_head, bridge, vto_head
        """
        for param in self.star.parameters():
            param.requires_grad = False
        for param in self.recommendation_head.parameters():
            param.requires_grad = False
        for param in self.bridge.parameters():
            param.requires_grad = False
        for param in self.vto_head.parameters():
            param.requires_grad = False
        # Unfreeze MAVEN and CHARM
        for param in self.maven.parameters():
            param.requires_grad = True
        for param in self.charm.parameters():
            param.requires_grad = True
        print("✓ Frozen all except MAVEN and CHARM for MAVEN stage")
    
    def unfreeze_all(self):
        """Unfreeze all auxiliary modules."""
        for param in self.star.parameters():
            param.requires_grad = True
        for param in self.maven.parameters():
            param.requires_grad = True
        for param in self.bridge.parameters():
            param.requires_grad = True
        for param in self.charm.parameters():
            param.requires_grad = True
        for param in self.vto_head.parameters():
            param.requires_grad = True
        for param in self.recommendation_head.parameters():
            param.requires_grad = True
        print("✓ Unfrozen all modules")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                domain: Union[Domain, torch.Tensor] = Domain.GENERAL,
                vto_labels: Optional[torch.Tensor] = None,
                use_star: bool = False,
                use_maven: bool = False,
                mode: str = "full",  # "full", "hidden_states", "charm_reward"
                training_stage: str = None,  # "sft", "charm", "star", "maven" - controls which modules are used
                context_mask: Optional[torch.Tensor] = None,
                **kwargs) -> ModelOutput:
        """
        Forward pass with optional STAR reasoning and MAVEN collaboration.
        
        Modes:
        - "full": Complete forward pass with all components
        - "hidden_states": Only return hidden states (for CHARM training)
        - "charm_reward": Return CHARM reward scores for hidden states
        
        Training Stages (for DDP compatibility - only uses modules that contribute to loss):
        - "sft": Uses base_model + vto_head only
        - "charm": Uses base_model + charm only
        - "star": Uses base_model + star only
        - "maven": Uses base_model + maven + charm
        - None: Uses all modules (inference mode)
        """
        if self.base_model is None:
            raise RuntimeError("Call load_base_model() first")
        
        # Mode: Just get hidden states (for CHARM preference training)
        if mode == "hidden_states":
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                logits_to_keep=1,  # hidden states only; skip full-vocab logits
                **kwargs
            )
            # Return pooled hidden states (BRIDGE is applied explicitly in training code)
            return masked_mean_pool(outputs.hidden_states[-1], attention_mask)
        
        # Mode: Get CHARM rewards for given hidden states
        if mode == "charm_reward":
            # input_ids is actually hidden_states in this mode
            hidden_states = input_ids
            rewards = self.charm(hidden_states, return_components=True)
            return rewards
        
        # Standard forward pass - controlled by training_stage for DDP compatibility
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]
        device = input_ids.device

        # Retrieval context: raw last-layer states over the dialogue tokens only.
        # The SFT sequence also holds the reply, which names the target item in
        # every ReDial example, and evaluation encodes the dialogue without
        # BRIDGE. Pooling the BRIDGE features of the whole sequence trained the
        # retriever to spot the answer in text it never sees at test time.
        context_pooled = (masked_mean_pool(hidden_states, context_mask)
                          if context_mask is not None else None)
        
        # Initialize outputs
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
        adapted_hidden = hidden_states  # Default: no adaptation
        vto_logits = None
        reward_result = None
        reasoning_result = None
        
        # ===== STAGE-SPECIFIC FORWARD PATHS =====
        # This ensures only modules that contribute to loss are called (DDP compatibility)
        
        if training_stage == "sft":
            # CRITICAL FIX: SFT now trains BRIDGE + recommendation_head + vto_head
            # This is essential for ranking evaluation to work!
            
            # Use BRIDGE for domain-invariant features
            domain_id = BRIDGE._as_domain_ids(domain, input_ids.size(0), device)
            bridge_out = self.bridge(hidden_states, domain, vto_labels=vto_labels,
                                     attention_mask=attention_mask)
            adapted_hidden = bridge_out["features"]
            
            # Pool adapted hidden states
            pooled = masked_mean_pool(adapted_hidden, attention_mask)
            
            # VTO prediction
            vto_logits = self.vto_head(pooled)
            
            # Recommendation head training (self-supervised via LM confidence)
            rec_scores = self.recommendation_head(pooled)
            
            # VTO loss
            if vto_labels is not None:
                vto_loss = F.binary_cross_entropy_with_logits(vto_logits, vto_labels.float())
                total_loss = total_loss + 0.3 * vto_loss
            
            # BRIDGE domain confusion loss
            if bridge_out.get("domain_loss") is not None:
                total_loss = total_loss + 0.1 * bridge_out["domain_loss"]
            
            # BRIDGE task preservation loss
            if bridge_out.get("task_loss") is not None:
                total_loss = total_loss + 0.1 * bridge_out["task_loss"]
            
            # NEW: BRIDGE contrastive loss for cross-domain alignment
            if bridge_out.get("contrastive_loss") is not None:
                total_loss = total_loss + 0.05 * bridge_out["contrastive_loss"]
            
            # Self-supervised recommendation head loss
            # Use LM confidence as target (higher confidence = better recommendation quality)
            # outputs.loss is the batch-mean LM loss: expanding it gave every
            # example the same target, so the head could only learn a constant.
            if labels is not None and outputs.logits is not None:
                rec_target = torch.exp(-self._per_example_nll(outputs.logits, labels))
                rec_target = rec_target.clamp(0.1, 0.9).detach()
                rec_loss = F.mse_loss(torch.sigmoid(rec_scores).squeeze(-1), rec_target)
                total_loss = total_loss + 0.1 * rec_loss
                
        elif training_stage == "charm":
            # CHARM: Use base_model + charm + BRIDGE for domain-aware preference learning
            # CRITICAL FIX: Use BRIDGE for domain adaptation during preference learning
            domain_id = BRIDGE._as_domain_ids(domain, input_ids.size(0), device)
            bridge_out = self.bridge(hidden_states, domain, attention_mask=attention_mask)
            adapted_hidden = bridge_out["features"]
            pooled = masked_mean_pool(adapted_hidden, attention_mask)
            
            reward_result = self.charm(pooled, domain_id)
            
            # Add BRIDGE domain loss during CHARM training
            if bridge_out.get("domain_loss") is not None:
                total_loss = total_loss + 0.05 * bridge_out["domain_loss"]
            # Note: Main CHARM loss is computed separately via compute_preference_loss()
            
        elif training_stage == "star":
            # STAR: Only use base_model + star for reasoning
            pooled = masked_mean_pool(hidden_states, attention_mask)
            reasoning_result = self.star(pooled, return_path=True)
            # STAR has its own loss computation in training loop
            
        elif training_stage == "maven":
            # MAVEN: Use base_model + maven + charm
            pooled = masked_mean_pool(hidden_states, attention_mask)
            domain_id = BRIDGE._as_domain_ids(domain, input_ids.size(0), device)
            maven_result = self.maven(pooled)
            pooled = maven_result["output"]
            reward_result = self.charm(pooled, domain_id)
            # MAVEN loss is computed in training loop
            
        else:
            # Full forward pass (inference or when training_stage is None)
            # BRIDGE: Domain adaptation
            domain_id = BRIDGE._as_domain_ids(domain, input_ids.size(0), device)
            bridge_out = self.bridge(hidden_states, domain, vto_labels=vto_labels,
                                     attention_mask=attention_mask)
            adapted_hidden = bridge_out["features"]
            
            # Pool for downstream tasks
            pooled = masked_mean_pool(adapted_hidden, attention_mask)
            
            # Optional STAR reasoning
            if use_star:
                reasoning_result = self.star(pooled, return_path=True)
                pooled = reasoning_result["final_hidden"]
            
            # Optional MAVEN collaboration
            if use_maven:
                maven_result = self.maven(pooled)
                pooled = maven_result["output"]
            
            # VTO prediction
            vto_logits = self.vto_head(pooled)
            
            # CHARM rewards (recommendation quality)
            reward_result = self.charm(pooled, domain_id)
            
            # Add domain loss from BRIDGE
            if bridge_out.get("domain_loss") is not None:
                total_loss = total_loss + 0.1 * bridge_out["domain_loss"]
            
            # Add VTO loss
            if vto_labels is not None:
                vto_loss = F.binary_cross_entropy_with_logits(vto_logits, vto_labels.float())
                total_loss = total_loss + 0.3 * vto_loss
        
        # Return dict for DataParallel compatibility, ModelOutput otherwise
        if self._use_dataparallel_output:
            return {
                'logits': outputs.logits,
                'loss': total_loss,
                'hidden_states': adapted_hidden,
                'vto_logits': vto_logits,
                'reward_scores': reward_result,
                'reasoning_path': reasoning_result.get("reasoning_path") if reasoning_result else None,
                'context_pooled': context_pooled
            }
        else:
            return ModelOutput(
                logits=outputs.logits,
                loss=total_loss,
                hidden_states=adapted_hidden,
                vto_logits=vto_logits,
                reward_scores=reward_result,
                reasoning_path=reasoning_result.get("reasoning_path") if reasoning_result else None,
                context_pooled=context_pooled
            )
    
    def compute_preference_loss(self, chosen_input_ids: torch.Tensor,
                                 chosen_attention_mask: torch.Tensor,
                                 rejected_input_ids: torch.Tensor,
                                 rejected_attention_mask: torch.Tensor,
                                 domain: Domain = Domain.GENERAL) -> Dict[str, torch.Tensor]:
        """Compute CHARM preference loss for chosen vs rejected.
        
        CRITICAL FIX: Uses BRIDGE for domain-adapted features to match training.
        """
        
        # Get hidden states for both
        chosen_out = self.base_model(chosen_input_ids, chosen_attention_mask, output_hidden_states=True)
        rejected_out = self.base_model(rejected_input_ids, rejected_attention_mask, output_hidden_states=True)
        
        chosen_hidden = masked_mean_pool(chosen_out.hidden_states[-1], chosen_attention_mask)
        rejected_hidden = masked_mean_pool(rejected_out.hidden_states[-1], rejected_attention_mask)
        
        # Use input tensor's device for DataParallel compatibility
        device = chosen_input_ids.device
        domain_id = BRIDGE._as_domain_ids(domain, chosen_input_ids.size(0), device)
        
        # CRITICAL FIX: Apply BRIDGE for domain-adapted features
        # This ensures preference learning uses the same feature space as SFT
        bridge_chosen = self.bridge(chosen_hidden, domain)
        bridge_rejected = self.bridge(rejected_hidden, domain)
        
        chosen_adapted = bridge_chosen["features"]
        rejected_adapted = bridge_rejected["features"]
        
        return self.charm.compute_preference_loss(chosen_adapted, rejected_adapted, domain_id)
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 256, temperature: float = 0.7,
                 do_sample: bool = True, force_think_token: bool = True, 
                 **kwargs) -> torch.Tensor:
        """Generate response with proper structured output format
        
        Uses constrained generation to ensure proper format.
        """
        if self.base_model is None:
            raise RuntimeError("Call load_base_model() first")
        
        # Remove conflicting kwargs
        kwargs.pop('pad_token_id', None)
        kwargs.pop('eos_token_id', None)
        kwargs.pop('max_new_tokens', None)
        kwargs.pop('temperature', None)
        kwargs.pop('do_sample', None)
        
        # If force_think_token, prepend think token
        if force_think_token and self.tokenizer is not None:
            think_token = "<|think|>"
            think_ids = self.tokenizer.encode(think_token, add_special_tokens=False)
            if think_ids:
                think_tensor = torch.tensor([think_ids], device=input_ids.device)
                think_tensor = think_tensor.expand(input_ids.size(0), -1)
                
                input_ids = torch.cat([input_ids, think_tensor], dim=1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(attention_mask.size(0), len(think_ids), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)
        
        # Generation parameters optimized for structured output
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        
        gen_kwargs.update(kwargs)
        
        return self.base_model.generate(**gen_kwargs)