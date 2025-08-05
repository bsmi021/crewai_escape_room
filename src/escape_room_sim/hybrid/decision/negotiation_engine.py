"""
Multi-Agent Negotiation Engine

Implements negotiation protocols, conflict resolution, and coordination
mechanisms for multi-agent decision making in the escape room scenario.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..core_architecture import PerceptionData, DecisionData
from ..llm.client import OptimizedLLMClient

logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Types of negotiation proposals"""
    COORDINATION = "coordination"
    RESOURCE_SHARING = "resource_sharing"
    TASK_ALLOCATION = "task_allocation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TIMING_SYNC = "timing_sync"


@dataclass
class NegotiationProposal:
    """Represents a negotiation proposal between agents"""
    proposer: str
    target: str
    proposal_type: ProposalType
    resource_requirements: Dict[str, float]
    expected_outcome: str
    priority: float
    timestamp: datetime
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


@dataclass
class NegotiationOutcome:
    """Represents the outcome of a negotiation"""
    participants: List[str]
    agreement_reached: bool
    agreed_actions: Dict[str, str]
    resource_allocation: Dict[str, Dict[str, float]]
    compromise_level: float
    execution_order: List[str]
    timestamp: datetime
    negotiation_rounds: int = 0
    consensus_score: float = 0.0


class MultiAgentNegotiationEngine:
    """
    Handles multi-agent negotiation and coordination protocols
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize negotiation engine
        
        Args:
            config: Configuration for negotiation parameters
        """
        self.negotiation_timeout = config.get("negotiation_timeout", 10.0)
        self.max_rounds = config.get("max_negotiation_rounds", 3)
        self.consensus_threshold = config.get("consensus_threshold", 0.7)
        self.conflict_resolution = config.get("conflict_resolution", "majority_vote")
        
        # Agent priority weights for different action types
        self.priority_weights = config.get("priority_weights", {})
        
        # LLM client for negotiation facilitation
        llm_config = config.get("llm", {})
        self.llm_client = OptimizedLLMClient(llm_config) if llm_config else None
        
        # Negotiation history
        self.negotiation_history: List[Dict[str, Any]] = []
    
    async def generate_proposals(self, perceptions: Dict[str, PerceptionData]) -> List[NegotiationProposal]:
        """
        Generate negotiation proposals based on agent perceptions
        
        Args:
            perceptions: Current perceptions for all agents
            
        Returns:
            List of negotiation proposals
        """
        proposals = []
        
        for agent_id, perception in perceptions.items():
            agent_proposals = self._generate_agent_proposals(agent_id, perception, perceptions)
            proposals.extend(agent_proposals)
        
        return proposals
    
    def _generate_agent_proposals(self, agent_id: str, perception: PerceptionData, 
                                all_perceptions: Dict[str, PerceptionData]) -> List[NegotiationProposal]:
        """Generate proposals for a specific agent"""
        proposals = []
        
        # Coordination proposals
        if perception.nearby_agents:
            coordination_proposal = NegotiationProposal(
                proposer=agent_id,
                target="team",
                proposal_type=ProposalType.COORDINATION,
                resource_requirements={"time": 3.0, "energy": 0.1},
                expected_outcome="coordinated_action",
                priority=self._calculate_proposal_priority(agent_id, "coordination", perception),
                timestamp=datetime.now(),
                conditions={"nearby_agents": perception.nearby_agents}
            )
            proposals.append(coordination_proposal)
        
        # Resource sharing proposals
        agent_tools = perception.resources.get("tools", [])
        if agent_tools:
            for other_agent in perception.nearby_agents:
                other_perception = all_perceptions.get(other_agent)
                if other_perception:
                    other_tools = other_perception.resources.get("tools", [])
                    if len(agent_tools) > len(other_tools):  # Has more tools
                        resource_proposal = NegotiationProposal(
                            proposer=agent_id,
                            target=other_agent,
                            proposal_type=ProposalType.RESOURCE_SHARING,
                            resource_requirements={"tools": 1.0},
                            expected_outcome="resource_optimization",
                            priority=self._calculate_proposal_priority(agent_id, "resource_sharing", perception),
                            timestamp=datetime.now(),
                            conditions={"shared_tools": agent_tools[:1]}  # Share one tool
                        )
                        proposals.append(resource_proposal)
        
        # Task allocation proposals
        high_priority_actions = self._get_high_priority_actions(agent_id, perception)
        if high_priority_actions:
            for action in high_priority_actions:
                task_proposal = NegotiationProposal(
                    proposer=agent_id,
                    target="team",
                    proposal_type=ProposalType.TASK_ALLOCATION,
                    resource_requirements=self._estimate_action_cost(action, perception),
                    expected_outcome=f"efficient_{action}",
                    priority=self._calculate_proposal_priority(agent_id, "task_allocation", perception),
                    timestamp=datetime.now(),
                    conditions={"preferred_action": action}
                )
                proposals.append(task_proposal)
        
        return proposals
    
    def _calculate_proposal_priority(self, agent_id: str, proposal_type: str, 
                                   perception: PerceptionData) -> float:
        """Calculate priority for a proposal based on agent weights and context"""
        base_priority = 0.5
        
        # Get agent-specific weights
        agent_weights = self.priority_weights.get(agent_id, {})
        type_weight = agent_weights.get(proposal_type, 0.5)
        
        # Adjust based on environmental factors
        time_pressure = perception.environmental_state.get("time_pressure", 0.0)
        energy_level = perception.resources.get("energy", 1.0)
        
        # Higher time pressure increases coordination priority
        if proposal_type == "coordination" and time_pressure > 0.7:
            type_weight += 0.2
        
        # Low energy decreases resource sharing priority
        if proposal_type == "resource_sharing" and energy_level < 0.3:
            type_weight -= 0.1
        
        return min(1.0, max(0.0, base_priority + type_weight))
    
    def _get_high_priority_actions(self, agent_id: str, perception: PerceptionData) -> List[str]:
        """Get high priority actions for an agent based on their role"""
        available_actions = perception.available_actions
        
        if "strategist" in agent_id.lower():
            priority_actions = ["analyze", "assess_risk", "plan"]
        elif "mediator" in agent_id.lower():
            priority_actions = ["coordinate", "mediate", "facilitate"]
        elif "survivor" in agent_id.lower():
            priority_actions = ["use_tool", "escape_attempt", "survive"]
        else:
            priority_actions = ["examine", "move"]
        
        return [action for action in priority_actions if action in available_actions]
    
    def _estimate_action_cost(self, action: str, perception: PerceptionData) -> Dict[str, float]:
        """Estimate resource cost for an action"""
        base_costs = {
            "move": {"energy": 0.1, "time": 1.0},
            "examine": {"energy": 0.05, "time": 2.0},
            "analyze": {"energy": 0.2, "time": 5.0},
            "coordinate": {"energy": 0.1, "time": 3.0},
            "use_tool": {"energy": 0.15, "time": 2.5},
            "communicate": {"energy": 0.05, "time": 1.5}
        }
        
        return base_costs.get(action, {"energy": 0.1, "time": 2.0})
    
    def detect_conflicts(self, proposals: List[NegotiationProposal]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between proposals
        
        Args:
            proposals: List of negotiation proposals
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group proposals by target resource or agent
        resource_groups: Dict[str, List[NegotiationProposal]] = {}
        
        for proposal in proposals:
            target = proposal.target
            if target not in resource_groups:
                resource_groups[target] = []
            resource_groups[target].append(proposal)
        
        # Detect conflicts within groups
        for resource, group_proposals in resource_groups.items():
            if len(group_proposals) > 1:
                # Check for resource conflicts
                total_requirements = {}
                for proposal in group_proposals:
                    for resource_type, amount in proposal.resource_requirements.items():
                        total_requirements[resource_type] = total_requirements.get(resource_type, 0) + amount
                
                # Check if any resource is over-allocated
                for resource_type, total_needed in total_requirements.items():
                    if total_needed > 1.0:  # Assuming 1.0 is max available
                        conflict = {
                            "type": "resource_conflict",
                            "resource": resource,
                            "resource_type": resource_type,
                            "agents": [p.proposer for p in group_proposals],
                            "total_demand": total_needed,
                            "proposals": group_proposals
                        }
                        conflicts.append(conflict)
        
        return conflicts
    
    async def execute_negotiation_round(self, proposals: List[NegotiationProposal],
                                      perceptions: Dict[str, PerceptionData]) -> Dict[str, Any]:
        """
        Execute a single round of negotiation
        
        Args:
            proposals: Proposals to negotiate
            perceptions: Current agent perceptions
            
        Returns:
            Round results including responses and consensus
        """
        if not self.llm_client:
            # Fallback to simple rule-based negotiation
            return self._execute_rule_based_round(proposals, perceptions)
        
        # Create negotiation prompts for each agent
        prompts = {}
        for agent_id in perceptions.keys():
            prompt = self._create_negotiation_prompt(agent_id, proposals, perceptions[agent_id])
            prompts[agent_id] = prompt
        
        try:
            # Get responses from all agents
            responses = await self.llm_client.generate_decisions_batch(prompts)
            
            # Calculate consensus
            consensus_score = self.calculate_consensus(responses)
            consensus_reached = consensus_score >= self.consensus_threshold
            
            return {
                "round_number": 1,
                "responses": responses,
                "consensus_score": consensus_score,
                "consensus_reached": consensus_reached,
                "proposals": proposals
            }
            
        except Exception as e:
            logger.warning(f"LLM negotiation failed: {e}, falling back to rule-based")
            return self._execute_rule_based_round(proposals, perceptions)
    
    def _execute_rule_based_round(self, proposals: List[NegotiationProposal],
                                perceptions: Dict[str, PerceptionData]) -> Dict[str, Any]:
        """Execute rule-based negotiation round"""
        responses = {}
        
        for agent_id in perceptions.keys():
            # Simple rule: agree to coordination, negotiate resource sharing
            relevant_proposals = [p for p in proposals if p.target == agent_id or p.target == "team"]
            
            if relevant_proposals:
                highest_priority = max(relevant_proposals, key=lambda p: p.priority)
                if highest_priority.proposal_type == ProposalType.COORDINATION:
                    responses[agent_id] = f"I agree to coordinate for {highest_priority.expected_outcome}"
                else:
                    responses[agent_id] = f"I'm willing to negotiate {highest_priority.proposal_type.value}"
            else:
                responses[agent_id] = "I'll observe and follow team decisions"
        
        consensus_score = self.calculate_consensus(responses)
        
        return {
            "round_number": 1,
            "responses": responses,
            "consensus_score": consensus_score,
            "consensus_reached": consensus_score >= self.consensus_threshold,
            "proposals": proposals
        }
    
    def _create_negotiation_prompt(self, agent_id: str, proposals: List[NegotiationProposal],
                                 perception: PerceptionData) -> str:
        """Create negotiation prompt for an agent"""
        relevant_proposals = [
            p for p in proposals 
            if p.target == agent_id or p.target == "team" or p.proposer == agent_id
        ]
        
        prompt_parts = [
            f"You are a {agent_id.replace('_', ' ').title()} in a team negotiation.",
            "Current proposals on the table:"
        ]
        
        for i, proposal in enumerate(relevant_proposals):
            prompt_parts.append(
                f"{i+1}. {proposal.proposer} proposes {proposal.proposal_type.value} "
                f"for {proposal.expected_outcome} (priority: {proposal.priority:.2f})"
            )
        
        prompt_parts.extend([
            "",
            f"Your current situation:",
            f"- Energy: {perception.resources.get('energy', 1.0):.2f}",
            f"- Available tools: {len(perception.resources.get('tools', []))}",
            f"- Nearby agents: {', '.join(perception.nearby_agents)}",
            "",
            "Respond with your position on these proposals. Be specific about what you agree to, "
            "what you want to modify, and what you oppose. Consider the team's success."
        ])
        
        return "\n".join(prompt_parts)
    
    def calculate_consensus(self, responses: Dict[str, str]) -> float:
        """
        Calculate consensus level from agent responses
        
        Args:
            responses: Agent responses to proposals
            
        Returns:
            Consensus score between 0.0 and 1.0
        """
        if not responses:
            return 0.0
        
        # Simple consensus calculation based on positive sentiment
        positive_words = ["agree", "support", "yes", "approve", "accept", "willing"]
        negative_words = ["disagree", "oppose", "no", "reject", "refuse", "against"]
        
        positive_count = 0
        negative_count = 0
        
        for response in responses.values():
            response_lower = response.lower()
            
            pos_score = sum(1 for word in positive_words if word in response_lower)
            neg_score = sum(1 for word in negative_words if word in response_lower)
            
            if pos_score > neg_score:
                positive_count += 1
            elif neg_score > pos_score:
                negative_count += 1
            # Neutral responses don't count either way
        
        total_responses = len(responses)
        if total_responses == 0:
            return 0.0
        
        # Consensus based on positive sentiment ratio
        return positive_count / total_responses
    
    async def conduct_negotiation(self, proposals: List[NegotiationProposal],
                                perceptions: Dict[str, PerceptionData]) -> NegotiationOutcome:
        """
        Conduct full negotiation process
        
        Args:
            proposals: Initial proposals
            perceptions: Current perceptions
            
        Returns:
            Final negotiation outcome
        """
        start_time = datetime.now()
        
        try:
            round_results = []
            current_proposals = proposals.copy()
            
            for round_num in range(1, self.max_rounds + 1):
                round_result = await asyncio.wait_for(
                    self.execute_negotiation_round(current_proposals, perceptions),
                    timeout=self.negotiation_timeout / self.max_rounds
                )
                
                round_result["round_number"] = round_num
                round_results.append(round_result)
                
                if round_result["consensus_reached"]:
                    break
                
                # Modify proposals for next round (simplified)
                current_proposals = self._modify_proposals_for_next_round(
                    current_proposals, round_result
                )
            
            # Create final outcome
            final_round = round_results[-1]
            outcome = self._create_negotiation_outcome(
                final_round, perceptions, len(round_results)
            )
            
            # Store in history
            self.negotiation_history.append({
                "timestamp": start_time,
                "duration": (datetime.now() - start_time).total_seconds(),
                "outcome": outcome,
                "rounds": round_results
            })
            
            return outcome
            
        except asyncio.TimeoutError:
            logger.warning("Negotiation timed out")
            
            # Return failed negotiation outcome
            return NegotiationOutcome(
                participants=list(perceptions.keys()),
                agreement_reached=False,
                agreed_actions={"status": "negotiation_timeout"},
                resource_allocation={},
                compromise_level=0.0,
                execution_order=[],
                timestamp=datetime.now(),
                negotiation_rounds=0,
                consensus_score=0.0
            )
    
    def _modify_proposals_for_next_round(self, proposals: List[NegotiationProposal],
                                       round_result: Dict[str, Any]) -> List[NegotiationProposal]:
        """Modify proposals based on round results"""
        # Simple modification: adjust priorities based on consensus
        modified_proposals = []
        
        for proposal in proposals:
            # Lower priority for proposals that weren't well received
            if round_result["consensus_score"] < 0.5:
                proposal.priority *= 0.8
            
            modified_proposals.append(proposal)
        
        return modified_proposals
    
    def _create_negotiation_outcome(self, final_round: Dict[str, Any],
                                  perceptions: Dict[str, PerceptionData],
                                  total_rounds: int) -> NegotiationOutcome:
        """Create final negotiation outcome"""
        participants = list(perceptions.keys())
        agreement_reached = final_round["consensus_reached"]
        
        # Extract agreed actions from responses
        agreed_actions = {}
        if agreement_reached:
            for agent_id, response in final_round["responses"].items():
                # Simple extraction - in practice this would be more sophisticated
                if "coordinate" in response.lower():
                    agreed_actions[agent_id] = "coordinate"
                elif "analyze" in response.lower():
                    agreed_actions[agent_id] = "analyze"
                elif "search" in response.lower():
                    agreed_actions[agent_id] = "search"
                else:
                    agreed_actions[agent_id] = "observe"
        
        # Simple resource allocation
        resource_allocation = {}
        for agent_id in participants:
            energy = perceptions[agent_id].resources.get("energy", 1.0)
            resource_allocation[agent_id] = {"energy": min(0.3, energy * 0.3)}
        
        # Simple execution order
        execution_order = participants.copy()
        
        # Calculate compromise level
        compromise_level = 1.0 - final_round["consensus_score"]
        
        return NegotiationOutcome(
            participants=participants,
            agreement_reached=agreement_reached,
            agreed_actions=agreed_actions,
            resource_allocation=resource_allocation,
            compromise_level=compromise_level,
            execution_order=execution_order,
            timestamp=datetime.now(),
            negotiation_rounds=total_rounds,
            consensus_score=final_round["consensus_score"]
        )
    
    def resolve_conflict_majority_vote(self, proposals: List[NegotiationProposal],
                                     votes: Dict[str, str]) -> Dict[str, Any]:
        """Resolve conflict using majority vote"""
        vote_counts = {}
        
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        if not vote_counts:
            return {"winner": None, "vote_count": {}}
        
        winner = max(vote_counts, key=vote_counts.get)
        
        return {
            "winner": winner,
            "vote_count": vote_counts,
            "total_votes": len(votes)
        }
    
    async def negotiate_resource_allocation(self, available_resources: Dict[str, Any],
                                          resource_requests: Dict[str, Dict[str, float]],
                                          perceptions: Dict[str, PerceptionData]) -> Dict[str, Dict[str, Any]]:
        """
        Negotiate resource allocation between agents
        
        Args:
            available_resources: Total available resources
            resource_requests: Agent requests for resources
            perceptions: Current perceptions
            
        Returns:
            Allocation of resources to agents
        """
        allocation = {}
        
        # Simple proportional allocation
        for resource_type, total_available in available_resources.items():
            if isinstance(total_available, (int, float)):
                # Calculate total requested
                total_requested = sum(
                    request.get(resource_type, 0) 
                    for request in resource_requests.values()
                )
                
                if total_requested <= total_available:
                    # Grant all requests
                    for agent_id, request in resource_requests.items():
                        if agent_id not in allocation:
                            allocation[agent_id] = {}
                        allocation[agent_id][resource_type] = request.get(resource_type, 0)
                else:
                    # Proportional allocation
                    ratio = total_available / total_requested
                    for agent_id, request in resource_requests.items():
                        if agent_id not in allocation:
                            allocation[agent_id] = {}
                        requested = request.get(resource_type, 0)
                        allocation[agent_id][resource_type] = requested * ratio
            
            elif isinstance(total_available, list):
                # Discrete resources (e.g., tools)
                remaining_items = total_available.copy()
                
                for agent_id, request in resource_requests.items():
                    if agent_id not in allocation:
                        allocation[agent_id] = {}
                    
                    requested_items = request.get(resource_type, [])
                    if isinstance(requested_items, list):
                        allocated_items = []
                        for item in requested_items:
                            if item in remaining_items:
                                allocated_items.append(item)
                                remaining_items.remove(item)
                        allocation[agent_id][resource_type] = allocated_items
        
        return allocation


# Utility functions

def create_negotiation_outcome(data: Dict[str, Any]) -> NegotiationOutcome:
    """Create negotiation outcome from data dictionary"""
    return NegotiationOutcome(
        participants=data["participants"],
        agreement_reached=data.get("consensus_score", 0.0) > 0.7,
        agreed_actions=data.get("agreed_actions", {}),
        resource_allocation=data.get("resource_allocation", {}),
        compromise_level=1.0 - data.get("consensus_score", 0.0),
        execution_order=data.get("participants", []),
        timestamp=datetime.now(),
        negotiation_rounds=1,
        consensus_score=data.get("consensus_score", 0.0)
    )


def optimize_execution_order(agreed_actions: Dict[str, str],
                           dependencies: Dict[str, List[str]]) -> List[str]:
    """Optimize execution order based on action dependencies"""
    # Simple topological sort
    execution_order = []
    remaining_actions = agreed_actions.copy()
    
    while remaining_actions:
        # Find actions with no dependencies
        ready_agents = []
        for agent_id, action in remaining_actions.items():
            deps = dependencies.get(action, [])
            if not deps or all(dep not in remaining_actions.values() for dep in deps):
                ready_agents.append(agent_id)
        
        if not ready_agents:
            # No progress possible - add remaining in arbitrary order
            ready_agents = list(remaining_actions.keys())
        
        # Add first ready agent to execution order
        next_agent = ready_agents[0]
        execution_order.append(next_agent)
        del remaining_actions[next_agent]
    
    return execution_order


def calculate_compromise_level(initial_proposals: Dict[str, Dict[str, Any]],
                             final_agreement: Dict[str, Dict[str, Any]]) -> float:
    """Calculate the level of compromise in the final agreement"""
    if not initial_proposals or not final_agreement:
        return 0.0
    
    total_compromise = 0.0
    agent_count = 0
    
    for agent_id in initial_proposals.keys():
        if agent_id in final_agreement:
            initial = initial_proposals[agent_id]
            final = final_agreement[agent_id]
            
            # Calculate compromise as difference in priority/resources
            initial_priority = initial.get("priority", 0.5)
            final_priority = final.get("priority", 0.5)
            
            compromise = abs(initial_priority - final_priority) / initial_priority if initial_priority > 0 else 0
            total_compromise += compromise
            agent_count += 1
    
    return total_compromise / agent_count if agent_count > 0 else 0.0