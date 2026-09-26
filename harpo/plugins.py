"""
HARPO Plugin System for Extensibility

Allows registration of custom evaluators, VTO plugins, and external evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PluginRegistry:
    """Registry for custom plugins"""
    evaluators: Dict[str, "EvaluatorPlugin"] = None
    vtos: Dict[str, "VTOPlugin"] = None
    
    def __post_init__(self):
        if self.evaluators is None:
            self.evaluators = {}
        if self.vtos is None:
            self.vtos = {}


class EvaluatorPlugin(ABC):
    """
    Base class for external evaluators.
    
    Example:
        class MyEvaluator(EvaluatorPlugin):
            def score(self, context: str, response: str) -> Dict[str, float]:
                return {"quality": 0.85}
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier"""
        pass
    
    @abstractmethod
    def score(self, context: str, response: str) -> Dict[str, float]:
        """
        Score a response in context.
        
        Args:
            context: Conversation context
            response: Model response to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        pass


class VTOPlugin(ABC):
    """
    Base class for Virtual Tool Objects (VTOs).
    
    Example:
        class SearchVTO(VTOPlugin):
            name = "search"
            input_schema = {"query": "string"}
            output_schema = {"results": "array"}
            
            def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
                return {"results": [...]}
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """VTO name identifier"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for inputs"""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """JSON schema for outputs"""
        pass
    
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VTO operation"""
        pass


class PluginManager:
    """Manages plugin registration and lifecycle"""
    
    def __init__(self):
        self.evaluators: Dict[str, EvaluatorPlugin] = {}
        self.vtos: Dict[str, VTOPlugin] = {}
    
    def register_evaluator(self, plugin: EvaluatorPlugin) -> None:
        """Register external evaluator plugin"""
        self.evaluators[plugin.name] = plugin
    
    def register_vto(self, plugin: VTOPlugin) -> None:
        """Register VTO plugin"""
        self.vtos[plugin.name] = plugin
    
    def get_evaluator(self, name: str) -> Optional[EvaluatorPlugin]:
        """Retrieve registered evaluator"""
        return self.evaluators.get(name)
    
    def get_vto(self, name: str) -> Optional[VTOPlugin]:
        """Retrieve registered VTO"""
        return self.vtos.get(name)
    
    def list_evaluators(self) -> List[str]:
        """List all registered evaluators"""
        return list(self.evaluators.keys())
    
    def list_vtos(self) -> List[str]:
        """List all registered VTOs"""
        return list(self.vtos.keys())


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager"""
    return _plugin_manager


def register_evaluator(plugin: EvaluatorPlugin) -> None:
    """Register evaluator plugin"""
    _plugin_manager.register_evaluator(plugin)


def register_vto(plugin: VTOPlugin) -> None:
    """Register VTO plugin"""
    _plugin_manager.register_vto(plugin)
