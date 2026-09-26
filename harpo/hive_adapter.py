"""
Component Configuration for HARPO

Each module can be enabled/disabled independently for:
- Ablation studies
- Performance tuning
- HIVE Adapt phase integration
- Incremental deployment
"""

import yaml
from typing import Dict, Any


class ComponentConfig:
    """Runtime component control"""
    
    def __init__(self):
        self.configs = {
            "charm_enabled": True,      # Contrastive Hierarchical Alignment
            "star_enabled": True,       # Structured Tree-of-thought Agentic Reasoning
            "bridge_enabled": True,     # Bidirectional Domain-Generalized Embeddings
            "maven_enabled": True,      # Multi-Agent Virtual Environment
            "vto_enabled": True,        # Virtual Tool Objects
        }
    
    def enable_charm_only(self):
        """For evaluation: Use CHARM module only"""
        self.configs = {
            "charm_enabled": True,
            "star_enabled": False,
            "bridge_enabled": False,
            "maven_enabled": False,
            "vto_enabled": False,
        }
        return self.configs
    
    def enable_star_only(self):
        """For evaluation: Use STAR module only"""
        self.configs = {
            "charm_enabled": False,
            "star_enabled": True,
            "bridge_enabled": False,
            "maven_enabled": False,
            "vto_enabled": True,
        }
        return self.configs
    
    def enable_bridge_only(self):
        """For evaluation: Use BRIDGE module only"""
        self.configs = {
            "charm_enabled": False,
            "star_enabled": False,
            "bridge_enabled": True,
            "maven_enabled": False,
            "vto_enabled": False,
        }
        return self.configs
    
    def enable_maven_only(self):
        """For evaluation: Use MAVEN module only"""
        self.configs = {
            "charm_enabled": False,
            "star_enabled": False,
            "bridge_enabled": False,
            "maven_enabled": True,
            "vto_enabled": True,
        }
        return self.configs
    
    def enable_all(self):
        """Full HARPO system"""
        self.configs = {
            "charm_enabled": True,
            "star_enabled": True,
            "bridge_enabled": True,
            "maven_enabled": True,
            "vto_enabled": True,
        }
        return self.configs
    
    def custom_config(self, **kwargs):
        """Custom component combination for A/B testing"""
        self.configs.update(kwargs)
        return self.configs
    
    def to_dict(self) -> Dict[str, bool]:
        """Get current configuration"""
        return self.configs.copy()
    
    def to_yaml(self) -> str:
        """Export as YAML"""
        return yaml.dump(self.configs, default_flow_style=False)
    
    @staticmethod
    def from_yaml(yaml_content: str) -> 'ComponentConfig':
        """Load from YAML"""
        cfg = ComponentConfig()
        cfg.configs = yaml.safe_load(yaml_content)
        return cfg


# HIVE Supervision & Adaptation
class AdaptationMetrics:
    """Metrics for HIVE's Adapt phase"""
    
    def __init__(self):
        self.metrics = {
            "charm_contribution": 0.0,      # % improvement from CHARM
            "star_contribution": 0.0,       # % improvement from STAR
            "bridge_contribution": 0.0,     # % improvement from BRIDGE
            "maven_contribution": 0.0,      # % improvement from MAVEN
            "total_quality_score": 0.0,     # Overall score
            "performance_per_module": {},   # Per-module performance
            "recommended_adaptation": "",   # AI recommendation for next step
        }
    
    def set_metrics(self, **kwargs):
        """Update metrics from evaluation"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key] = value
    
    def recommend_adaptation(self) -> str:
        """Generate recommendation for HIVE Supervision"""
        charm = self.metrics["charm_contribution"]
        star = self.metrics["star_contribution"]
        bridge = self.metrics["bridge_contribution"]
        maven = self.metrics["maven_contribution"]
        
        total = charm + star + bridge + maven
        if total == 0:
            return "All modules performing equally. Consider balanced resources."
        
        contributions = {
            "CHARM": charm / total,
            "STAR": star / total,
            "BRIDGE": bridge / total,
            "MAVEN": maven / total,
        }
        
        best_module = max(contributions, key=contributions.get)
        return f"Increase focus on {best_module} ({contributions[best_module]*100:.1f}% contribution). Reduce emphasis on others."
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics"""
        self.metrics["recommended_adaptation"] = self.recommend_adaptation()
        return self.metrics


if __name__ == "__main__":
    # Example: CHARM-only evaluation
    config = ComponentConfig()
    charm_only = config.enable_charm_only()
    print("CHARM-Only Config:", charm_only)
    
    # Example: Custom A/B test
    ab_config = config.custom_config(charm_enabled=True, star_enabled=False, bridge_enabled=True)
    print("A/B Test Config:", ab_config)
    
    # Example: Adaptation metrics
    metrics = AdaptationMetrics()
    metrics.set_metrics(
        charm_contribution=0.25,
        star_contribution=0.35,
        bridge_contribution=0.20,
        maven_contribution=0.20,
        total_quality_score=0.85
    )
    print("Recommendation:", metrics.recommend_adaptation())
    print("Metrics:", metrics.to_dict())
