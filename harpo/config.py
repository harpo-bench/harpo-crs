"""
HARPO: Optimizing Conversational Recommendation for User-Aligned Quality 
       via Hierarchical Preference Learning

Configuration and Data Structures

Novel Components:
- CHARM: Contrastive Hierarchical Alignment with Reward Marginalization  
- STAR: Structured Tree-of-thought Agentic Reasoning
- BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
- MAVEN: Multi-Agent Virtual Environment for Recommendations

ACL 2025 Submission
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum, auto

# ============================================================================
# VIRTUAL TOOL OPERATIONS (VTOs) - Domain Agnostic Abstractions
# ============================================================================

class VTO(Enum):
    """Virtual Tool Operations - Domain-agnostic reasoning primitives."""
    # Information Extraction
    ANALYZE_SENTIMENT = "analyze_sentiment"
    EXTRACT_CONTEXT = "extract_context"
    EXTRACT_ENTITIES = "extract_entities"
    
    # User Modeling
    RETRIEVE_PREFERENCES = "retrieve_preferences"
    IDENTIFY_CONSTRAINTS = "identify_constraints"
    MODEL_USER_STATE = "model_user_state"
    
    # Search & Retrieval
    SEARCH_CANDIDATES = "search_candidates"
    FILTER_RESULTS = "filter_results"
    MATCH_ATTRIBUTES = "match_attributes"
    
    # Ranking & Selection
    RANK_OPTIONS = "rank_options"
    COMPARE_OPTIONS = "compare_options"
    SELECT_BEST = "select_best"
    
    # Knowledge & Reasoning
    QUERY_KNOWLEDGE = "query_knowledge"
    REASON_OVER_GRAPH = "reason_over_graph"
    INFER_IMPLICIT = "infer_implicit"
    
    # Interaction
    EXPLAIN_CHOICE = "explain_choice"
    REFINE_QUERY = "refine_query"
    HANDLE_REJECTION = "handle_rejection"
    
    # Memory
    TRACK_HISTORY = "track_history"
    UPDATE_BELIEFS = "update_beliefs"
    RECALL_CONTEXT = "recall_context"


VTO_DESCRIPTIONS = {
    VTO.ANALYZE_SENTIMENT: "Understand user emotions, mood, feelings from message",
    VTO.EXTRACT_CONTEXT: "Extract situation, occasion, setting from conversation",
    VTO.EXTRACT_ENTITIES: "Identify named entities (products, brands, categories)",
    VTO.RETRIEVE_PREFERENCES: "Get user's stated or inferred preferences",
    VTO.IDENTIFY_CONSTRAINTS: "Find hard requirements (budget, time, size, etc.)",
    VTO.MODEL_USER_STATE: "Build/update model of user's current needs",
    VTO.SEARCH_CANDIDATES: "Search for matching items/recommendations",
    VTO.FILTER_RESULTS: "Apply filters to narrow down candidates",
    VTO.MATCH_ATTRIBUTES: "Check compatibility with preferences",
    VTO.RANK_OPTIONS: "Order candidates by relevance/preference",
    VTO.COMPARE_OPTIONS: "Compare multiple alternatives side by side",
    VTO.SELECT_BEST: "Select optimal item(s) from ranked list",
    VTO.QUERY_KNOWLEDGE: "Access domain-specific facts and information",
    VTO.REASON_OVER_GRAPH: "Perform multi-hop reasoning on knowledge graph",
    VTO.INFER_IMPLICIT: "Infer unstated preferences or requirements",
    VTO.EXPLAIN_CHOICE: "Generate explanation for recommendation",
    VTO.REFINE_QUERY: "Ask clarifying questions or refine search",
    VTO.HANDLE_REJECTION: "Process and learn from user rejection",
    VTO.TRACK_HISTORY: "Remember and use conversation history",
    VTO.UPDATE_BELIEFS: "Update belief state based on new information",
    VTO.RECALL_CONTEXT: "Retrieve relevant past context",
}

# VTO Categories for hierarchical reasoning
VTO_CATEGORIES = {
    "extraction": [VTO.ANALYZE_SENTIMENT, VTO.EXTRACT_CONTEXT, VTO.EXTRACT_ENTITIES],
    "user_modeling": [VTO.RETRIEVE_PREFERENCES, VTO.IDENTIFY_CONSTRAINTS, VTO.MODEL_USER_STATE],
    "retrieval": [VTO.SEARCH_CANDIDATES, VTO.FILTER_RESULTS, VTO.MATCH_ATTRIBUTES],
    "ranking": [VTO.RANK_OPTIONS, VTO.COMPARE_OPTIONS, VTO.SELECT_BEST],
    "reasoning": [VTO.QUERY_KNOWLEDGE, VTO.REASON_OVER_GRAPH, VTO.INFER_IMPLICIT],
    "interaction": [VTO.EXPLAIN_CHOICE, VTO.REFINE_QUERY, VTO.HANDLE_REJECTION],
    "memory": [VTO.TRACK_HISTORY, VTO.UPDATE_BELIEFS, VTO.RECALL_CONTEXT],
}

# ============================================================================
# DOMAIN CONFIGURATIONS
# ============================================================================

class Domain(Enum):
    FASHION = "fashion"
    MOVIES = "movies"
    ELECTRONICS = "electronics"
    GENERAL = "general"
    FOOD = "food"
    BOOKS = "books"


@dataclass
class DomainConfig:
    """Configuration for each domain."""
    name: str
    has_explicit_tools: bool
    has_images: bool
    item_type: str
    typical_tools: List[str]
    knowledge_sources: List[str]
    vto_priors: Dict[str, float] = field(default_factory=dict)


DOMAIN_CONFIGS = {
    Domain.FASHION: DomainConfig(
        name="fashion",
        has_explicit_tools=True,
        has_images=True,
        item_type="clothing",
        typical_tools=["search_products", "visual_match", "style_check"],
        knowledge_sources=["fashion_kg", "trend_db", "catalog"],
        vto_priors={"analyze_sentiment": 0.8, "match_attributes": 0.9}
    ),
    Domain.MOVIES: DomainConfig(
        name="movies",
        has_explicit_tools=False,
        has_images=False,
        item_type="movie",
        typical_tools=[],
        knowledge_sources=["movie_db", "reviews", "cast_info"],
        vto_priors={"query_knowledge": 0.9, "compare_options": 0.8}
    ),
    Domain.ELECTRONICS: DomainConfig(
        name="electronics",
        has_explicit_tools=True,
        has_images=True,
        item_type="device",
        typical_tools=["search", "compare", "filter"],
        knowledge_sources=["product_db", "reviews", "specs"],
        vto_priors={"identify_constraints": 0.9, "compare_options": 0.9}
    ),
    Domain.GENERAL: DomainConfig(
        name="general",
        has_explicit_tools=True,
        has_images=False,
        item_type="item",
        typical_tools=["search", "filter", "compare", "recommend"],
        knowledge_sources=["knowledge_graph"],
        vto_priors={}
    ),
    Domain.FOOD: DomainConfig(
        name="food",
        has_explicit_tools=True,
        has_images=True,
        item_type="restaurant/dish",
        typical_tools=["search", "filter", "recommend"],
        knowledge_sources=["restaurant_db", "menu_db"],
        vto_priors={"analyze_sentiment": 0.7, "identify_constraints": 0.9}
    ),
    Domain.BOOKS: DomainConfig(
        name="books",
        has_explicit_tools=False,
        has_images=False,
        item_type="book",
        typical_tools=[],
        knowledge_sources=["book_db", "reviews"],
        vto_priors={"query_knowledge": 0.8, "retrieve_preferences": 0.9}
    ),
}

# ============================================================================
# AGENT CONFIGURATION (MAVEN)
# ============================================================================

class AgentRole(Enum):
    """Agent roles in MAVEN framework."""
    RECOMMENDER = "recommender"
    CRITIC = "critic"
    EXPLAINER = "explainer"
    USER_SIMULATOR = "user_simulator"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    role: AgentRole
    description: str
    capabilities: List[VTO]
    priority: float = 1.0


AGENT_CONFIGS = {
    AgentRole.RECOMMENDER: AgentConfig(
        role=AgentRole.RECOMMENDER,
        description="Main agent for generating recommendations",
        capabilities=[VTO.SEARCH_CANDIDATES, VTO.RANK_OPTIONS, VTO.SELECT_BEST],
        priority=1.0
    ),
    AgentRole.CRITIC: AgentConfig(
        role=AgentRole.CRITIC,
        description="Evaluates recommendation quality",
        capabilities=[VTO.COMPARE_OPTIONS, VTO.IDENTIFY_CONSTRAINTS],
        priority=0.9
    ),
    AgentRole.EXPLAINER: AgentConfig(
        role=AgentRole.EXPLAINER,
        description="Generates explanations for recommendations",
        capabilities=[VTO.EXPLAIN_CHOICE, VTO.QUERY_KNOWLEDGE],
        priority=0.8
    ),
    AgentRole.USER_SIMULATOR: AgentConfig(
        role=AgentRole.USER_SIMULATOR,
        description="Simulates user behavior for self-play",
        capabilities=[VTO.MODEL_USER_STATE, VTO.RETRIEVE_PREFERENCES],
        priority=0.7
    ),
    AgentRole.ORCHESTRATOR: AgentConfig(
        role=AgentRole.ORCHESTRATOR,
        description="Coordinates agent interactions",
        capabilities=list(VTO),
        priority=1.0
    ),
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the base model.
    
    PUBLICATION OPTIMIZED: Using DeepSeek-R1-Distill-Qwen-7B
    - Excellent reasoning capability (R1 distillation)
    - 7B parameters - good balance of quality and speed
    - Works well with 2x A100 80GB
    """
    # Base model - DeepSeek R1 Distill for strong reasoning
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    # Model dimensions (auto-detected from model, defaults for 7B Qwen)
    hidden_size: int = 3584  # DeepSeek-R1-Distill-Qwen-7B hidden size
    num_attention_heads: int = 28
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    
    # LoRA configuration - Optimized for 7B model
    lora_r: int = 128  # Higher rank for 7B model capacity
    lora_alpha: int = 256  # 2x lora_r for stable training
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Learning rates - CRITICAL: Conservative for stable training on 7B model
    # High LR causes catastrophic forgetting (gibberish output)
    sft_learning_rate: float = 5e-5  # FIXED: Reduced from 2e-4 to prevent forgetting
    charm_learning_rate: float = 2e-5  # FIXED: Reduced for stable preference learning
    star_learning_rate: float = 1e-5  # FIXED: Very conservative for reasoning module
    maven_learning_rate: float = 1e-5  # FIXED: Very conservative for multi-agent
    
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10  # FIXED: Longer warmup for training stability
    max_grad_norm: float = 1.0
    
    # Flash Attention 2 - 2-3x speedup on A100
    use_flash_attention: bool = True


# ============================================================================
# STAR (Tree-of-Thought) CONFIGURATION
# ============================================================================

@dataclass
class STARConfig:
    """Configuration for STAR reasoning module."""
    max_depth: int = 3
    branching_factor: int = 2
    beam_width: int = 2
    search_strategy: str = "beam"  # "greedy", "best_first", "beam", "mcts"
    use_value_network: bool = True
    value_weight: float = 0.5
    enable_backtracking: bool = True
    backtrack_threshold: float = 0.3
    thought_max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9


# ============================================================================
# CHARM (Preference) CONFIGURATION
# ============================================================================

@dataclass  
class CHARMConfig:
    """Configuration for CHARM preference learning."""
    beta: float = 0.5  # FIXED: Increased from 0.1 for stronger preference learning
    reference_free: bool = True  # SimPO-style
    gamma: float = 0.3  # FIXED: Reduced target margin for smoother optimization
    length_normalization: bool = True
    num_reward_heads: int = 4
    reward_head_names: List[str] = field(default_factory=lambda: [
        "relevance", "diversity", "user_satisfaction", "engagement"
    ])
    meta_learning: bool = True
    meta_learning_rate: float = 1e-3
    contrastive_temperature: float = 0.1
    hard_negative_ratio: float = 0.3


# ============================================================================
# BRIDGE (Domain Adaptation) CONFIGURATION
# ============================================================================

@dataclass
class BRIDGEConfig:
    """Configuration for BRIDGE domain adapter."""
    adapter_hidden_size: int = 256
    num_projection_heads: int = 4
    adversarial_alpha: float = 1.0
    gradient_reversal: bool = True
    domain_confusion_weight: float = 0.1
    use_domain_gates: bool = True
    gate_init: float = 0.5


# ============================================================================
# MAVEN (Multi-Agent) CONFIGURATION
# ============================================================================

@dataclass
class MAVENConfig:
    """Configuration for MAVEN multi-agent framework."""
    num_agents: int = 3  # Reduced for efficiency
    agent_roles: List[AgentRole] = field(default_factory=lambda: [
        AgentRole.RECOMMENDER, AgentRole.CRITIC, AgentRole.EXPLAINER
    ])
    max_communication_rounds: int = 2
    message_max_tokens: int = 64
    self_play_iterations: int = 2
    user_simulation_temperature: float = 0.9
    voting_strategy: str = "weighted"
    conflict_resolution: str = "orchestrator"


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for the item retriever.

    No counterpart existed in the original code: nothing in the four-stage
    curriculum trained the model to rank items, despite Recall@K being the
    headline metric.
    """
    enabled: bool = True
    embed_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    loss_weight: float = 1.0

    # Content-only ignores the collaborative signal in ~10k ReDial dialogues;
    # id-only cannot generalise to unseen items. "hybrid" fuses both.
    item_representation: str = "hybrid"      # "content" | "id" | "hybrid"
    id_embedding_weight: float = 0.5
    max_catalog_size: int = 20000
    item_max_length: int = 48

    # In-batch negatives alone are weak and popularity-biased.
    num_hard_negatives: int = 4
    hard_negative_start_epoch: int = 0
    popularity_debias: float = 0.5

    # Per-item logit bias initialised to log training frequency (logit
    # adjustment). Ranking then starts at the popularity baseline -- which beats
    # every dialogue-blind scorer on ReDial -- and the retriever only has to
    # learn what the dialogue adds on top.
    item_bias: bool = True

    use_reranker: bool = False
    rerank_top_k: int = 50


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for all stages.
    
    OPTIMIZED FOR: 2x A100 80GB with DeepSeek-R1-Distill-Qwen-7B
    Expected training time: ~1-1.5 hours (OPTIMIZED from ~2-3 hours)
    """
    # Data
    # OPTIMIZED: Reduced from 768 to 512 based on data analysis
    # 95th percentile = 355 tokens, 99th = 471 tokens
    # Only 0.5% of samples truncated at 512
    max_seq_length: int = 512  # OPTIMIZED: Was 768
    batch_size: int = 12  # OPTIMIZED: Can increase due to shorter sequences
    gradient_accumulation_steps: int = 4  # Effective batch = 12 * 4 * 2 GPUs = 96
    
    # Stage epochs - FIXED: More epochs for thorough learning
    sft_epochs: int = 1  # FIXED: Increased from 3 for better VTO learning
    charm_epochs: int = 1  # FIXED: Increased from 2 for better preference learning
    star_epochs: int = 1
    maven_epochs: int = 1
    
    # Evaluation
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 50
    
    # Optimization - CRITICAL for stable training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10  # FIXED: Longer warmup for stable convergence
    max_grad_norm: float = 1.0
    
    # Hardware - optimized for 2x A100 80GB
    fp16: bool = False
    bf16: bool = True  # BF16 for A100
    gradient_checkpointing: bool = True  # Enable for 7B model memory efficiency
    
    # Acceleration settings
    use_accelerate: bool = True  # Use HuggingFace Accelerate for multi-GPU
    dataloader_num_workers: int = 8  # OPTIMIZED: Increased from 4 for better throughput
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4  # OPTIMIZED: Added prefetch factor
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Module-specific configs
    star_config: STARConfig = field(default_factory=STARConfig)
    charm_config: CHARMConfig = field(default_factory=CHARMConfig)
    bridge_config: BRIDGEConfig = field(default_factory=BRIDGEConfig)
    maven_config: MAVENConfig = field(default_factory=MAVENConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ToolCall:
    """Represents a tool call."""
    tool_name: str
    arguments: Dict[str, Any]
    vto_type: VTO = VTO.EXTRACT_CONTEXT
    result: Optional[str] = None
    is_explicit: bool = True
    confidence: float = 1.0


@dataclass
class ThoughtNode:
    """Node in the tree-of-thought (STAR)."""
    node_id: str
    parent_id: Optional[str]
    vto_sequence: List[VTO]
    thought_text: str
    value_score: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    turn_id: int
    user_input: str
    system_response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    vto_sequence: List[VTO] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    intent: str = ""


@dataclass
class Conversation:
    """Complete conversation."""
    conversation_id: str
    domain: Domain
    turns: List[ConversationTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """Preference pair for CHARM training."""
    conversation_id: str
    context: str
    chosen_response: str
    rejected_response: str
    chosen_vtos: List[VTO] = field(default_factory=list)
    rejected_vtos: List[VTO] = field(default_factory=list)
    chosen_tools: List[ToolCall] = field(default_factory=list)
    rejected_tools: List[ToolCall] = field(default_factory=list)
    reward_margin: float = 0.0
    hierarchical_rewards: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    domain: Domain = Domain.GENERAL


@dataclass
class AgentMessage:
    """Message between agents in MAVEN."""
    sender: AgentRole
    receiver: AgentRole
    content: str
    message_type: str = "query"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation metrics."""
    dataset: str
    domain: Domain = Domain.GENERAL
    
    # Recommendation metrics (PRIMARY - what we optimize for)
    # Standard metrics used in ReDial, KBRD, UniCRS, KGSF papers
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    mrr: float = 0.0  # Overall MRR (standard in papers)
    mrr_at_10: float = 0.0
    mrr_at_20: float = 0.0
    ndcg_at_1: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    
    # User satisfaction (PRIMARY)
    user_satisfaction: float = 0.0
    engagement_score: float = 0.0
    
    # VTO/Tool metrics (SECONDARY)
    vto_precision: float = 0.0
    vto_recall: float = 0.0
    vto_f1: float = 0.0
    tool_selection_accuracy: float = 0.0
    tool_execution_accuracy: float = 0.0
    
    # Generation metrics
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    perplexity: float = 0.0
    
    # Alignment metrics
    preference_accuracy: float = 0.0
    reward_margin: float = 0.0
    
    # STAR metrics (novel)
    reasoning_depth: float = 0.0
    backtrack_rate: float = 0.0
    thought_quality: float = 0.0
    
    # MAVEN metrics (novel)
    agent_agreement_rate: float = 0.0
    explanation_quality: float = 0.0


# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are HARPO-MT, an intelligent recommendation assistant that reasons step-by-step to provide helpful, accurate, and personalized recommendations.

When processing user requests:
1. First, analyze the user's intent and context
2. Identify relevant constraints and preferences  
3. Search for appropriate options
4. Rank and select the best recommendations
5. Explain your choices clearly

Always be helpful, accurate, and respectful."""

VTO_ANNOTATION_PROMPT = """Analyze this conversation turn and identify which Virtual Tool Operations (VTOs) the system performed.

VTO Types:
{vto_descriptions}

Context: {context}
User: {user_input}
System: {system_response}

Output JSON: {{"vtos": ["vto_name1", "vto_name2"], "reasoning": "brief explanation"}}
"""

STAR_THOUGHT_PROMPT = """Given the context, generate a reasoning step for recommendation.

Context: {context}
User Query: {user_input}
Previous Thoughts: {previous_thoughts}

Consider: What information do we need? What VTOs should we apply?

Thought:"""

CHARM_PREFERENCE_PROMPT = """Generate a {quality} quality response for preference learning.

Context: {context}
User: {user_input}
Domain: {domain}

If "high": Natural, helpful, relevant recommendations
If "low": Misses context, wrong approach, unhelpful

Response:"""

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

SPECIAL_TOKENS = {
    "vto_start": "<|vto_start|>",
    "vto_end": "<|vto_end|>",
    "tool_start": "<|tool_start|>",
    "tool_end": "<|tool_end|>",
    "think_start": "<|think|>",
    "think_end": "<|/think|>",
    "thought_start": "<|thought|>",
    "thought_end": "<|/thought|>",
    "agent_start": "<|agent:",
    "agent_end": "|>",
    "domain_fashion": "<|domain:fashion|>",
    "domain_movies": "<|domain:movies|>",
    "domain_electronics": "<|domain:electronics|>",
    "domain_general": "<|domain:general|>",
    "domain_food": "<|domain:food|>",
    "domain_books": "<|domain:books|>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_domain_token(domain: Domain) -> str:
    """Get special token for domain."""
    token_key = f"domain_{domain.value}"
    return SPECIAL_TOKENS.get(token_key, SPECIAL_TOKENS["domain_general"])


def get_vto_description(vto: VTO) -> str:
    """Get description for a VTO."""
    return VTO_DESCRIPTIONS.get(vto, "Unknown operation")


def get_vto_category(vto: VTO) -> str:
    """Get category for a VTO."""
    for category, vtos in VTO_CATEGORIES.items():
        if vto in vtos:
            return category
    return "unknown"


def format_tool_call(tool_name: str, arguments: Dict) -> str:
    """Format tool call as string."""
    import json
    return f"{SPECIAL_TOKENS['tool_start']}{json.dumps({'tool': tool_name, 'args': arguments})}{SPECIAL_TOKENS['tool_end']}"


def format_thought(thought: str, vtos: List[VTO]) -> str:
    """Format thought node as string."""
    vto_str = ", ".join([v.value for v in vtos])
    return f"{SPECIAL_TOKENS['thought_start']}[{vto_str}] {thought}{SPECIAL_TOKENS['thought_end']}"