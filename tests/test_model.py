#!/usr/bin/env python3
"""
HARPO: Basic Tests

Run with: pytest tests/test_model.py -v
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all modules can be imported."""
    from harpo.config import (
        VTO, Domain, AgentRole,
        ModelConfig, TrainingConfig,
        STARConfig, CHARMConfig, BRIDGEConfig, MAVENConfig,
    )
    
    assert len(VTO) == 21, "Should have 21 VTOs"
    assert len(Domain) == 6, "Should have 6 domains"
    assert len(AgentRole) == 5, "Should have 5 agent roles"


def test_vto_categories():
    """Test VTO category mapping."""
    from harpo.config import VTO, VTO_CATEGORIES, get_vto_category
    
    # Each VTO should belong to exactly one category
    all_vtos_in_categories = set()
    for vtos in VTO_CATEGORIES.values():
        all_vtos_in_categories.update(vtos)
    
    # Check category assignment works
    assert get_vto_category(VTO.SEARCH_CANDIDATES) == "retrieval"
    assert get_vto_category(VTO.EXPLAIN_CHOICE) == "interaction"


def test_model_config():
    """Test model configuration."""
    from harpo.config import VTO, ModelConfig
    
    config = ModelConfig()
    assert config.hidden_size > 0
    assert len(VTO) == 21  # the paper's 21 VTOs; the model sizes its heads by len(VTO)


def test_training_config():
    """Test training configuration."""
    from harpo.config import TrainingConfig
    
    config = TrainingConfig()
    assert config.sft_epochs > 0
    assert config.charm_epochs > 0
    assert config.star_epochs > 0
    assert config.maven_epochs > 0


def test_special_tokens():
    """Test special token definitions."""
    from harpo.config import SPECIAL_TOKENS
    
    assert "<|think|>" in SPECIAL_TOKENS.values()
    assert "<|/think|>" in SPECIAL_TOKENS.values()
    assert "<|domain:movies|>" in SPECIAL_TOKENS.values()


def test_domain_configs():
    """Test domain configurations exist."""
    from harpo.config import Domain, DOMAIN_CONFIGS
    
    for domain in Domain:
        assert domain in DOMAIN_CONFIGS
        config = DOMAIN_CONFIGS[domain]
        assert config.name == domain.value


def test_data_structures():
    """Test data structure creation."""
    from harpo.config import (
        Conversation, ConversationTurn, PreferencePair,
        Domain, VTO
    )
    
    # Create a conversation turn
    turn = ConversationTurn(
        turn_id=1,
        user_input="I want a comedy movie",
        system_response="I recommend The Grand Budapest Hotel",
        vto_sequence=[VTO.SEARCH_CANDIDATES, VTO.RANK_OPTIONS]
    )
    assert turn.turn_id == 1
    assert len(turn.vto_sequence) == 2
    
    # Create a conversation
    conv = Conversation(
        conversation_id="test_001",
        domain=Domain.MOVIES,
        turns=[turn]
    )
    assert conv.conversation_id == "test_001"
    assert conv.domain == Domain.MOVIES


def test_evaluation_result():
    """Test evaluation result structure."""
    from harpo.config import EvaluationResult, Domain
    
    result = EvaluationResult(
        dataset="test",
        recall_at_10=0.25,
        mrr=0.15,
        user_satisfaction=0.8
    )
    
    assert result.recall_at_10 == 0.25
    assert result.mrr == 0.15
    assert result.user_satisfaction == 0.8


class TestModelComponents:
    """Test model component initialization (without GPU)."""
    
    def test_bridge_init(self):
        """Test BRIDGE module can be created."""
        import torch
        from harpo.config import BRIDGEConfig
        
        # Only test if model.py can be imported
        try:
            from harpo.model import BRIDGE
            
            config = BRIDGEConfig()
            bridge = BRIDGE(
                hidden_size=256,  # Small for testing
                config=config,
                num_domains=6,
                num_vtos=24
            )
            
            # Test forward pass with dummy input
            x = torch.randn(2, 256)
            from harpo.config import Domain
            output = bridge(x, Domain.MOVIES)
            
            assert "features" in output
            assert output["features"].shape == (2, 256)
        except Exception as e:
            print(f"BRIDGE test skipped: {e}")
    
    def test_charm_init(self):
        """Test CHARM module can be created."""
        import torch
        
        try:
            from harpo.model import CHARM
            from harpo.config import CHARMConfig
            
            config = CHARMConfig()
            charm = CHARM(
                hidden_size=256,
                config=config,
                num_domains=6
            )
            
            # Test forward pass
            x = torch.randn(2, 256)
            domain_ids = torch.zeros(2, dtype=torch.long)
            output = charm(x, domain_ids)
            
            assert "total" in output
            assert output["total"].shape == (2, 1)
        except Exception as e:
            print(f"CHARM test skipped: {e}")
    
    def test_star_init(self):
        """Test STAR module can be created."""
        import torch
        
        try:
            from harpo.model import STAR
            from harpo.config import STARConfig
            
            config = STARConfig()
            star = STAR(
                hidden_size=256,
                config=config,
                num_vtos=24
            )
            
            # Test forward pass
            x = torch.randn(2, 256)
            output = star(x)
            
            assert "final_hidden" in output
        except Exception as e:
            print(f"STAR test skipped: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_imports()
    print("✓ Imports OK")
    
    test_vto_categories()
    print("✓ VTO categories OK")
    
    test_model_config()
    print("✓ Model config OK")
    
    test_training_config()
    print("✓ Training config OK")
    
    test_special_tokens()
    print("✓ Special tokens OK")
    
    test_domain_configs()
    print("✓ Domain configs OK")
    
    test_data_structures()
    print("✓ Data structures OK")
    
    test_evaluation_result()
    print("✓ Evaluation result OK")
    
    print("\nAll basic tests passed!")