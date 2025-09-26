"""
Learning and adaptation capabilities for autonomous agents.
"""
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from . import AGUIProtocol, AGUIEvent, EventType
from .autonomous import AutonomousAgent, TaskContext, Task

@dataclass
class LearningContext:
    """Context for agent learning and adaptation."""
    session_id: str
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class AdaptiveAgent(AutonomousAgent):
    """Agent capable of learning and adapting from interactions."""
    
    def __init__(self, protocol: AGUIProtocol):
        super().__init__(protocol)
        self.learning_contexts: Dict[str, LearningContext] = {}
        self.preference_weights: Dict[str, float] = {}
    
    def get_learning_context(self, user_id: str) -> LearningContext:
        """Get or create learning context for a user."""
        if user_id not in self.learning_contexts:
            self.learning_contexts[user_id] = LearningContext(
                session_id=f"learning_{user_id}",
                user_id=user_id
            )
        return self.learning_contexts[user_id]
    
    async def update_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ):
        """Update user preferences."""
        context = self.get_learning_context(user_id)
        context.preferences.update(preferences)
        
        await self.protocol.emit(AGUIEvent(
            type=EventType.STATE_SYNC,
            data={
                "user_id": user_id,
                "preferences": context.preferences
            }
        ))
    
    async def record_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Record user interaction for learning."""
        context = self.get_learning_context(user_id)
        
        interaction = {
            **interaction_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        context.history.append(interaction)
        await self._update_metrics(context)
    
    async def _update_metrics(self, context: LearningContext):
        """Update learning metrics based on interaction history."""
        # Example metrics calculation
        total_interactions = len(context.history)
        success_count = sum(
            1 for interaction in context.history
            if interaction.get("success", False)
        )
        
        context.metrics.update({
            "total_interactions": total_interactions,
            "success_rate": success_count / total_interactions if total_interactions > 0 else 0
        })
        
        await self.protocol.emit(AGUIEvent(
            type=EventType.STATE_SYNC,
            data={
                "user_id": context.user_id,
                "metrics": context.metrics
            }
        ))
    
    async def adapt_behavior(self, user_id: str) -> Dict[str, Any]:
        """Adapt agent behavior based on learning context."""
        context = self.get_learning_context(user_id)
        
        # Analyze interaction history
        recent_interactions = context.history[-10:]  # Last 10 interactions
        
        # Calculate success patterns
        success_patterns = self._analyze_success_patterns(recent_interactions)
        
        # Update weights based on successful patterns
        self._update_weights(success_patterns)
        
        return {
            "adapted_weights": self.preference_weights,
            "success_patterns": success_patterns
        }
    
    def _analyze_success_patterns(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze patterns in successful interactions."""
        patterns = {}
        
        for interaction in interactions:
            if interaction.get("success", False):
                # Extract features that led to success
                features = interaction.get("features", {})
                for feature, value in features.items():
                    if feature not in patterns:
                        patterns[feature] = {"count": 0, "sum": 0}
                    patterns[feature]["count"] += 1
                    patterns[feature]["sum"] += value
        
        # Calculate average effectiveness of each pattern
        return {
            feature: data["sum"] / data["count"]
            for feature, data in patterns.items()
        }
    
    def _update_weights(self, success_patterns: Dict[str, float]):
        """Update preference weights based on success patterns."""
        # Simple moving average update
        alpha = 0.3  # Learning rate
        
        for feature, effectiveness in success_patterns.items():
            current_weight = self.preference_weights.get(feature, 0.5)
            self.preference_weights[feature] = (
                (1 - alpha) * current_weight + alpha * effectiveness
            )

class LearningPipeline:
    """Pipeline for processing and learning from interactions."""
    
    def __init__(self, agent: AdaptiveAgent):
        self.agent = agent
        self.steps: List[callable] = []
    
    def add_learning_step(self, step: callable):
        """Add a learning step to the pipeline."""
        self.steps.append(step)
    
    async def process_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Process an interaction through the learning pipeline."""
        # Record the interaction
        await self.agent.record_interaction(user_id, interaction_data)
        
        # Process through learning steps
        current_data = interaction_data
        for step in self.steps:
            current_data = await step(current_data)
        
        # Adapt agent behavior
        adaptation_result = await self.agent.adapt_behavior(user_id)
        
        # Emit learning update event
        await self.agent.protocol.emit(AGUIEvent(
            type=EventType.STATE_SYNC,
            data={
                "user_id": user_id,
                "learning_update": {
                    "processed_data": current_data,
                    "adaptation_result": adaptation_result
                }
            }
        ))
        
        return adaptation_result