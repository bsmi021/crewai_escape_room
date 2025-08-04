"""
MoralDilemmaEngine for managing ethical choices in competitive scenarios.
Presents context-appropriate dilemmas and tracks ethical burden of choices.
"""
from datetime import datetime
from typing import Dict, List, Optional
from .models import MoralDilemma, MoralChoice, ChoiceConsequences


class MoralDilemmaEngine:
    """Manages moral dilemmas and ethical choice tracking for competitive scenarios."""
    
    def __init__(self, dilemmas: List[MoralDilemma]):
        """Initialize MoralDilemmaEngine with list of available dilemmas."""
        self.dilemmas = dilemmas
        self.choices_made: Dict[str, List[ChoiceConsequences]] = {}
        self.ethical_scores: Dict[str, float] = {}
    
    def present_dilemma(self, agent_id: str, context: Dict) -> Optional[MoralDilemma]:
        """Present a context-appropriate moral dilemma to an agent."""
        self._validate_present_dilemma_inputs(agent_id, context)
        
        # Find first dilemma that matches the context
        for dilemma in self.dilemmas:
            if dilemma.applies_to_context(context):
                return dilemma
        
        return None  # No matching dilemma found
    
    def _validate_present_dilemma_inputs(self, agent_id: str, context: Dict):
        """Validate inputs for present_dilemma method."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if context is None:
            raise ValueError("Context cannot be None")
    
    def process_choice(self, agent_id: str, choice: MoralChoice) -> ChoiceConsequences:
        """Process a moral choice and return its consequences."""
        self._validate_process_choice_inputs(agent_id, choice)
        
        # Create consequences based on the choice
        consequences = ChoiceConsequences(
            agent_id=agent_id,
            choice_made=choice,
            survival_benefit_applied=choice.survival_benefit,
            ethical_cost_applied=choice.ethical_cost,
            trust_impacts_applied=choice.trust_impact.copy(),
            consequences_triggered=choice.consequences.copy(),
            timestamp=datetime.now()
        )
        
        # Record choice in agent's history
        if agent_id not in self.choices_made:
            self.choices_made[agent_id] = []
        self.choices_made[agent_id].append(consequences)
        
        # Update agent's ethical score
        if agent_id not in self.ethical_scores:
            self.ethical_scores[agent_id] = 0.0
        self.ethical_scores[agent_id] += choice.ethical_cost
        
        return consequences
    
    def _validate_process_choice_inputs(self, agent_id: str, choice: MoralChoice):
        """Validate inputs for process_choice method."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if choice is None:
            raise ValueError("Choice cannot be None")
    
    def calculate_ethical_burden(self, agent_id: str) -> float:
        """Calculate the cumulative ethical burden for an agent."""
        self._validate_ethical_burden_input(agent_id)
        
        return self.ethical_scores.get(agent_id, 0.0)
    
    def _validate_ethical_burden_input(self, agent_id: str):
        """Validate input for calculate_ethical_burden method."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
    
    def get_agent_choice_history(self, agent_id: str) -> List[ChoiceConsequences]:
        """Get the complete choice history for an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        return self.choices_made.get(agent_id, [])
    
    def get_choice_count(self, agent_id: str) -> int:
        """Get the total number of choices made by an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        return len(self.choices_made.get(agent_id, []))
    
    def get_selfish_choice_count(self, agent_id: str) -> int:
        """Get the number of selfish choices made by an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        history = self.choices_made.get(agent_id, [])
        return sum(1 for consequences in history if consequences.was_selfish_choice())
    
    def get_altruistic_choice_count(self, agent_id: str) -> int:
        """Get the number of altruistic choices made by an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        total_choices = self.get_choice_count(agent_id)
        selfish_choices = self.get_selfish_choice_count(agent_id)
        return total_choices - selfish_choices
    
    def get_moral_alignment(self, agent_id: str) -> str:
        """Get the moral alignment of an agent based on their choices."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        total_choices = self.get_choice_count(agent_id)
        if total_choices == 0:
            return "neutral"
        
        selfish_choices = self.get_selfish_choice_count(agent_id)
        selfish_ratio = selfish_choices / total_choices
        
        if selfish_ratio >= 0.8:
            return "highly_selfish"
        elif selfish_ratio >= 0.6:
            return "selfish"
        elif selfish_ratio >= 0.4:
            return "mixed"
        elif selfish_ratio >= 0.2:
            return "altruistic"
        else:
            return "highly_altruistic"
    
    def get_available_dilemmas_for_context(self, context: Dict) -> List[MoralDilemma]:
        """Get all dilemmas that apply to the given context."""
        if context is None:
            raise ValueError("Context cannot be None")
        
        return [dilemma for dilemma in self.dilemmas if dilemma.applies_to_context(context)]
    
    def get_all_agents_with_choices(self) -> List[str]:
        """Get list of all agents who have made moral choices."""
        return list(self.choices_made.keys())
    
    def get_ethical_burden_ranking(self) -> List[tuple]:
        """Get agents ranked by their ethical burden (highest first)."""
        return sorted(self.ethical_scores.items(), key=lambda x: x[1], reverse=True)
    
    def reset_agent_history(self, agent_id: str):
        """Reset the choice history and ethical score for an agent."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        if agent_id in self.choices_made:
            del self.choices_made[agent_id]
        if agent_id in self.ethical_scores:
            del self.ethical_scores[agent_id]
    
    def get_engine_statistics(self) -> Dict:
        """Get comprehensive statistics about the moral dilemma engine."""
        total_agents = len(self.choices_made)
        total_choices = sum(len(history) for history in self.choices_made.values())
        total_selfish = sum(self.get_selfish_choice_count(agent) 
                           for agent in self.choices_made.keys())
        
        return {
            "total_dilemmas_available": len(self.dilemmas),
            "total_agents_with_choices": total_agents,
            "total_choices_made": total_choices,
            "total_selfish_choices": total_selfish,
            "total_altruistic_choices": total_choices - total_selfish,
            "average_choices_per_agent": total_choices / total_agents if total_agents > 0 else 0,
            "selfish_choice_ratio": total_selfish / total_choices if total_choices > 0 else 0
        }