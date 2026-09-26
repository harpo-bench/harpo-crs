"""
HARPO: Hierarchical Agentic Reasoning with Preference Optimization
for Conversational Recommendation

Novel Components:
- CHARM: Contrastive Hierarchical Alignment with Reward Marginalization
- STAR: Structured Tree-of-thought Agentic Reasoning  
- BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
- MAVEN: Multi-Agent Virtual Environment for Recommendations
"""

from .config import (
    VTO, Domain, AgentRole,
    ModelConfig, TrainingConfig,
    STARConfig, CHARMConfig, BRIDGEConfig, MAVENConfig,
    Conversation, ConversationTurn, PreferencePair, EvaluationResult
)

from .model import HARPOMTv2
from .api import Evaluator, Comparator, Explainer, ScoreResult, ComparisonResult, ExplanationResult
from .plugins import EvaluatorPlugin, VTOPlugin, PluginManager, register_evaluator, register_vto, get_plugin_manager

__version__ = "1.0.0"
__all__ = [
    "HARPOMTv2",
    "VTO", "Domain", "AgentRole",
    "ModelConfig", "TrainingConfig",
    "STARConfig", "CHARMConfig", "BRIDGEConfig", "MAVENConfig",
    "Conversation", "ConversationTurn", "PreferencePair", "EvaluationResult"
]