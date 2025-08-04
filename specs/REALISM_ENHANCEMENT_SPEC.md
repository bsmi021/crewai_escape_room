# Realism Enhancement Specification
## CrewAI Escape Room Simulation - Phase 3 Requirements

### Document Information
- **Document Type**: Technical Specification
- **Priority**: MEDIUM - User Experience and Simulation Authenticity
- **Estimated Time**: 30-40 hours
- **Dependencies**: Critical Fixes (Phase 1) and Code Quality (Phase 2) completed
- **Author**: Escape Room Simulation Evaluation Report
- **Date**: 2025-08-04

---

## Overview

This specification addresses realism and immersion improvements for the CrewAI Escape Room Simulation. The focus is on enhancing agent personality distinctiveness, implementing competitive dynamics, adding physical interaction systems, and strengthening the survival constraint mechanics to create a truly engaging escape room experience.

---

## 1. Agent Personality Enhancement

### 1.1 Specialized Agent Tools and Capabilities

**Priority**: HIGH  
**Goal**: Give each agent unique tools that reflect their professional backgrounds

#### 1.1.1 Strategist-Specific Tools

**File**: `src/escape_room_sim/agents/tools/strategist_tools.py` (new file)

```python
"""
Specialized tools for the Strategist agent.
Military-inspired analytical and planning capabilities.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from crewai_tools import BaseTool


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Risk assessment result from strategic analysis."""
    threat_level: ThreatLevel
    success_probability: float
    resource_requirements: List[str]
    time_estimate: int
    contingency_options: List[str]
    risk_factors: List[str]


class StrategicAnalysisTool(BaseTool):
    """OODA Loop (Observe, Orient, Decide, Act) strategic analysis tool."""
    
    name: str = "strategic_analysis"
    description: str = "Performs systematic military-style strategic analysis using OODA loop methodology"
    
    def _run(self, situation: str, available_resources: Dict[str, Any]) -> str:
        """
        Execute OODA loop analysis on current situation.
        
        Args:
            situation: Current escape room situation description
            available_resources: Dictionary of available resources
            
        Returns:
            Formatted strategic analysis report
        """
        # OBSERVE - Analyze current state
        observations = self._observe_situation(situation, available_resources)
        
        # ORIENT - Understand context and constraints
        orientation = self._orient_analysis(observations)
        
        # DECIDE - Generate strategic options
        decisions = self._decide_options(orientation)
        
        # ACT - Recommend specific actions
        actions = self._act_recommendations(decisions)
        
        return self._format_ooda_report(observations, orientation, decisions, actions)
    
    def _observe_situation(self, situation: str, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and catalogue current situation."""
        return {
            "current_state": situation,
            "available_assets": list(resources.keys()),
            "resource_quantities": {k: v.get('quantity', 0) for k, v in resources.items()},
            "immediate_threats": self._identify_threats(situation),
            "opportunities": self._identify_opportunities(situation, resources)
        }
    
    def _orient_analysis(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Orient understanding of strategic context."""
        return {
            "strategic_position": self._assess_position(observations),
            "enemy_capabilities": self._assess_obstacles(observations),
            "friendly_capabilities": self._assess_team_assets(observations),
            "terrain_analysis": self._analyze_environment(observations),
            "time_factor": self._assess_time_pressure(observations)
        }
    
    def _decide_options(self, orientation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic decision options."""
        options = []
        
        # Direct assault option
        options.append({
            "name": "Direct Approach",
            "description": "Frontal approach using available resources",
            "success_probability": 0.6,
            "resource_cost": "high",
            "time_required": "short",
            "risk_level": "medium"
        })
        
        # Stealth/indirect option
        options.append({
            "name": "Indirect Approach", 
            "description": "Find alternative routes or methods",
            "success_probability": 0.7,
            "resource_cost": "medium",
            "time_required": "medium",
            "risk_level": "low"
        })
        
        # Combined arms option
        options.append({
            "name": "Combined Strategy",
            "description": "Multi-pronged approach using team specializations",
            "success_probability": 0.8,
            "resource_cost": "high",
            "time_required": "long",
            "risk_level": "medium"
        })
        
        return options
    
    def _act_recommendations(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate specific action recommendations."""
        best_option = max(decisions, key=lambda x: x["success_probability"])
        
        return {
            "recommended_strategy": best_option,
            "immediate_actions": [
                "Secure critical resources first",
                "Establish clear communication protocols", 
                "Assign roles based on capabilities",
                "Prepare contingency plans"
            ],
            "success_metrics": [
                "Resource efficiency",
                "Team coordination",
                "Time management",
                "Risk mitigation"
            ]
        }
    
    def _format_ooda_report(self, obs: Dict, orient: Dict, decide: List, act: Dict) -> str:
        """Format complete OODA analysis report."""
        return f"""
STRATEGIC ANALYSIS REPORT (OODA Methodology)

=== OBSERVE ===
Current Assets: {', '.join(obs['available_assets'])}
Immediate Threats: {', '.join(obs['immediate_threats'])}
Key Opportunities: {', '.join(obs['opportunities'])}

=== ORIENT ===
Strategic Position: {orient['strategic_position']}
Environmental Factors: {orient['terrain_analysis']}
Time Pressure: {orient['time_factor']}

=== DECIDE ===
Evaluated {len(decide)} strategic options:
{chr(10).join(f"• {opt['name']}: {opt['success_probability']:.0%} success probability" for opt in decide)}

=== ACT ===
RECOMMENDED STRATEGY: {act['recommended_strategy']['name']}
Success Probability: {act['recommended_strategy']['success_probability']:.0%}

IMMEDIATE ACTIONS:
{chr(10).join(f"1. {action}" for action in act['immediate_actions'])}

This analysis follows military tactical doctrine adapted for escape room scenarios.
"""


class RiskAssessmentTool(BaseTool):
    """Military-style risk assessment and mitigation planning."""
    
    name: str = "risk_assessment"
    description: str = "Conducts systematic risk assessment with mitigation strategies"
    
    def _run(self, proposed_action: str, current_situation: Dict[str, Any]) -> str:
        """
        Assess risks of proposed action using military risk assessment protocols.
        
        Args:
            proposed_action: Action being considered
            current_situation: Current game state
            
        Returns:
            Formatted risk assessment report
        """
        # Identify risks
        risks = self._identify_risks(proposed_action, current_situation)
        
        # Assess probability and impact
        assessed_risks = [self._assess_risk(risk) for risk in risks]
        
        # Generate mitigation strategies
        mitigations = self._generate_mitigations(assessed_risks)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(assessed_risks)
        
        return self._format_risk_report(proposed_action, assessed_risks, mitigations, overall_risk)
    
    def _identify_risks(self, action: str, situation: Dict[str, Any]) -> List[str]:
        """Identify potential risks in proposed action."""
        risks = []
        
        # Time-based risks
        if situation.get('time_remaining', 60) < 15:
            risks.append("Time pressure - insufficient time for complex actions")
        
        # Resource risks
        if situation.get('resource_scarcity', False):
            risks.append("Resource depletion - critical supplies may be exhausted")
        
        # Team coordination risks
        if situation.get('team_stress', 0) > 0.7:
            risks.append("Team cohesion - high stress affecting coordination")
        
        # Execution risks
        if "complex" in action.lower() or "difficult" in action.lower():
            risks.append("Execution complexity - high failure probability")
        
        return risks
    
    def _assess_risk(self, risk: str) -> Dict[str, Any]:
        """Assess individual risk probability and impact."""
        # Simple heuristic-based assessment
        impact = "HIGH" if any(keyword in risk.lower() for keyword in ["time", "critical", "failure"]) else "MEDIUM"
        probability = "HIGH" if any(keyword in risk.lower() for keyword in ["insufficient", "exhausted", "high"]) else "MEDIUM"
        
        return {
            "risk": risk,
            "probability": probability,
            "impact": impact,
            "risk_score": self._calculate_risk_score(probability, impact)
        }
    
    def _calculate_risk_score(self, probability: str, impact: str) -> int:
        """Calculate numerical risk score."""
        prob_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        impact_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        return prob_scores[probability] * impact_scores[impact]
    
    def _generate_mitigations(self, risks: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies for identified risks."""
        mitigations = []
        
        for risk in risks:
            if "time" in risk["risk"].lower():
                mitigations.append("Prepare backup quick-execution plans")
            if "resource" in risk["risk"].lower():
                mitigations.append("Identify resource alternatives and rationing strategies")
            if "team" in risk["risk"].lower():
                mitigations.append("Implement stress management and clear role assignments")
            if "execution" in risk["risk"].lower():
                mitigations.append("Break complex actions into simpler steps")
        
        return mitigations
    
    def _calculate_overall_risk(self, risks: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level."""
        total_score = sum(risk["risk_score"] for risk in risks)
        avg_score = total_score / len(risks) if risks else 0
        
        if avg_score >= 6:
            return "HIGH RISK"
        elif avg_score >= 4:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def _format_risk_report(self, action: str, risks: List[Dict], mitigations: List[str], overall: str) -> str:
        """Format comprehensive risk assessment report."""
        return f"""
TACTICAL RISK ASSESSMENT

ACTION UNDER ANALYSIS: {action}

IDENTIFIED RISKS:
{chr(10).join(f"• {risk['risk']} (P:{risk['probability']}, I:{risk['impact']}, Score:{risk['risk_score']})" for risk in risks)}

MITIGATION STRATEGIES:
{chr(10).join(f"• {mitigation}" for mitigation in mitigations)}

OVERALL RISK LEVEL: {overall}

RECOMMENDATION: {'PROCEED WITH CAUTION' if overall != 'HIGH RISK' else 'RECOMMEND ALTERNATIVE APPROACH'}

Assessment conducted using military risk management protocols.
"""


# Additional Strategist tools
class ResourceOptimizationTool(BaseTool):
    """Optimize resource allocation using military logistics principles."""
    name: str = "resource_optimization"
    description: str = "Optimizes resource allocation using logistics and supply chain principles"
    
    def _run(self, available_resources: Dict[str, Any], objectives: List[str]) -> str:
        """Optimize resource allocation for objectives."""
        # Implementation for resource optimization
        return "Resource optimization analysis complete."
```

#### 1.1.2 Mediator-Specific Tools

**File**: `src/escape_room_sim/agents/tools/mediator_tools.py` (new file)

```python
"""
Specialized tools for the Mediator agent.
Crisis counseling and group facilitation capabilities.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from crewai_tools import BaseTool


class ConflictType(Enum):
    RESOURCE_DISPUTE = "resource_dispute"
    STRATEGY_DISAGREEMENT = "strategy_disagreement"
    LEADERSHIP_CONFLICT = "leadership_conflict"
    PERSONALITY_CLASH = "personality_clash"
    STRESS_REACTION = "stress_reaction"


@dataclass
class TeamDynamicsAssessment:
    """Assessment of current team dynamics."""
    cohesion_score: float  # 0.0 to 1.0
    stress_level: float    # 0.0 to 1.0
    communication_quality: str
    trust_levels: Dict[str, float]
    potential_conflicts: List[str]
    intervention_needed: bool


class TeamDynamicsAnalyzer(BaseTool):
    """Analyze team dynamics and interpersonal relationships."""
    
    name: str = "team_dynamics_analysis"
    description: str = "Analyzes team cohesion, stress levels, and interpersonal dynamics"
    
    def _run(self, team_interactions: str, current_stress: float) -> str:
        """
        Analyze team dynamics based on recent interactions.
        
        Args:
            team_interactions: Recent team conversation/interaction text
            current_stress: Current team stress level (0.0-1.0)
            
        Returns:
            Formatted team dynamics assessment
        """
        # Analyze communication patterns
        communication_analysis = self._analyze_communication(team_interactions)
        
        # Assess stress indicators
        stress_indicators = self._identify_stress_indicators(team_interactions, current_stress)
        
        # Detect potential conflicts
        conflict_risks = self._detect_conflict_risks(team_interactions)
        
        # Evaluate team cohesion
        cohesion_score = self._calculate_cohesion(communication_analysis, stress_indicators)
        
        # Generate intervention recommendations
        interventions = self._recommend_interventions(stress_indicators, conflict_risks, cohesion_score)
        
        return self._format_dynamics_report(
            communication_analysis, stress_indicators, conflict_risks, 
            cohesion_score, interventions
        )
    
    def _analyze_communication(self, interactions: str) -> Dict[str, Any]:
        """Analyze communication patterns and quality."""
        # Simple heuristic analysis of communication patterns
        word_count = len(interactions.split())
        
        # Look for positive communication indicators
        positive_indicators = sum(1 for word in ["agree", "good", "yes", "excellent", "great"] 
                                 if word in interactions.lower())
        
        # Look for negative communication indicators  
        negative_indicators = sum(1 for word in ["no", "wrong", "bad", "disagree", "impossible"]
                                 if word in interactions.lower())
        
        # Look for collaborative language
        collaborative_indicators = sum(1 for phrase in ["we should", "let's", "together", "team"]
                                     if phrase in interactions.lower())
        
        return {
            "communication_volume": "HIGH" if word_count > 200 else "MEDIUM" if word_count > 100 else "LOW",
            "positive_tone": positive_indicators / max(1, word_count / 100),
            "negative_tone": negative_indicators / max(1, word_count / 100),
            "collaborative_language": collaborative_indicators / max(1, word_count / 100),
            "overall_quality": self._assess_communication_quality(positive_indicators, negative_indicators, collaborative_indicators)
        }
    
    def _identify_stress_indicators(self, interactions: str, stress_level: float) -> List[str]:
        """Identify stress indicators in team communication."""
        indicators = []
        
        if stress_level > 0.7:
            indicators.append("High overall stress level affecting team performance")
        
        # Language-based stress indicators
        stress_words = ["panic", "urgent", "crisis", "emergency", "pressure", "worried", "anxious"]
        if any(word in interactions.lower() for word in stress_words):
            indicators.append("Stress-related language patterns detected")
        
        # Communication breakdown indicators
        if interactions.count("...") > 3 or interactions.count("!") > 5:
            indicators.append("Communication fragmentation or emotional intensity")
        
        return indicators
    
    def _detect_conflict_risks(self, interactions: str) -> List[Dict[str, Any]]:
        """Detect potential or emerging conflicts."""
        risks = []
        
        # Direct disagreement detection
        if any(phrase in interactions.lower() for phrase in ["i disagree", "that's wrong", "no way"]):
            risks.append({
                "type": ConflictType.STRATEGY_DISAGREEMENT,
                "severity": "MEDIUM",
                "description": "Direct disagreement on strategy detected"
            })
        
        # Resource competition indicators
        if any(phrase in interactions.lower() for phrase in ["need that", "my turn", "i should get"]):
            risks.append({
                "type": ConflictType.RESOURCE_DISPUTE,
                "severity": "LOW",
                "description": "Potential resource competition emerging"
            })
        
        return risks
    
    def _calculate_cohesion(self, communication: Dict, stress_indicators: List[str]) -> float:
        """Calculate team cohesion score."""
        base_score = 0.7  # Neutral starting point
        
        # Communication quality impact
        if communication["overall_quality"] == "EXCELLENT":
            base_score += 0.2
        elif communication["overall_quality"] == "POOR":
            base_score -= 0.2
        
        # Stress impact
        stress_penalty = len(stress_indicators) * 0.1
        base_score -= stress_penalty
        
        return max(0.0, min(1.0, base_score))
    
    def _recommend_interventions(self, stress: List[str], conflicts: List[Dict], cohesion: float) -> List[str]:
        """Recommend specific interventions based on analysis."""
        interventions = []
        
        if cohesion < 0.5:
            interventions.append("PRIORITY: Team building intervention required")
        
        if len(stress) > 2:
            interventions.append("Implement stress management techniques")
            interventions.append("Encourage brief relaxation/centering moment")
        
        if conflicts:
            interventions.append("Address emerging conflicts through structured dialogue")
            interventions.append("Clarify roles and decision-making authority")
        
        if not interventions:
            interventions.append("Continue monitoring - team dynamics stable")
        
        return interventions
    
    def _assess_communication_quality(self, positive: int, negative: int, collaborative: int) -> str:
        """Assess overall communication quality."""
        if collaborative > 2 and positive > negative:
            return "EXCELLENT"
        elif positive > negative:
            return "GOOD"
        elif negative > positive * 2:
            return "POOR"
        else:
            return "NEUTRAL"
    
    def _format_dynamics_report(self, comm: Dict, stress: List[str], conflicts: List[Dict], 
                               cohesion: float, interventions: List[str]) -> str:
        """Format comprehensive team dynamics report."""
        return f"""
TEAM DYNAMICS ASSESSMENT

COMMUNICATION ANALYSIS:
• Volume: {comm['communication_volume']}
• Quality: {comm['overall_quality']}
• Collaborative Language: {comm['collaborative_language']:.1f} indicators per 100 words
• Tone Balance: {comm['positive_tone']:.1f} positive, {comm['negative_tone']:.1f} negative

STRESS INDICATORS:
{chr(10).join(f"• {indicator}" for indicator in stress) if stress else "• No significant stress indicators detected"}

CONFLICT RISKS:
{chr(10).join(f"• {risk['type'].value}: {risk['description']} (Severity: {risk['severity']})" for risk in conflicts) if conflicts else "• No immediate conflict risks identified"}

TEAM COHESION SCORE: {cohesion:.2f}/1.0 ({('STRONG' if cohesion > 0.7 else 'MODERATE' if cohesion > 0.4 else 'WEAK')})

RECOMMENDED INTERVENTIONS:
{chr(10).join(f"• {intervention}" for intervention in interventions)}

Assessment based on crisis counseling and group facilitation protocols.
"""


class ConflictResolutionTool(BaseTool):
    """Professional conflict resolution and mediation techniques."""
    
    name: str = "conflict_resolution"
    description: str = "Provides structured conflict resolution using professional mediation techniques"
    
    def _run(self, conflict_description: str, parties_involved: List[str]) -> str:
        """
        Apply professional conflict resolution methodology.
        
        Args:
            conflict_description: Description of the conflict
            parties_involved: List of agents involved in conflict
            
        Returns:
            Structured conflict resolution plan
        """
        # Analyze conflict type and root causes
        conflict_analysis = self._analyze_conflict(conflict_description)
        
        # Generate mediation strategy
        mediation_strategy = self._create_mediation_strategy(conflict_analysis, parties_involved)
        
        # Develop resolution options
        resolution_options = self._generate_resolution_options(conflict_analysis)
        
        # Create implementation plan
        implementation_plan = self._create_implementation_plan(resolution_options)
        
        return self._format_resolution_plan(
            conflict_analysis, mediation_strategy, resolution_options, implementation_plan
        )
    
    def _analyze_conflict(self, description: str) -> Dict[str, Any]:
        """Analyze conflict using professional mediation framework."""
        # Identify conflict type
        conflict_type = self._classify_conflict_type(description)
        
        # Identify underlying interests vs positions
        interests = self._identify_interests(description)
        positions = self._identify_positions(description)
        
        # Assess escalation level
        escalation_level = self._assess_escalation(description)
        
        return {
            "type": conflict_type,
            "interests": interests,
            "positions": positions,
            "escalation_level": escalation_level,
            "complexity": "HIGH" if len(interests) > 3 else "MEDIUM" if len(interests) > 1 else "LOW"
        }
    
    def _classify_conflict_type(self, description: str) -> ConflictType:
        """Classify the type of conflict."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["resource", "supplies", "tool", "equipment"]):
            return ConflictType.RESOURCE_DISPUTE
        elif any(word in description_lower for word in ["strategy", "plan", "approach", "method"]):
            return ConflictType.STRATEGY_DISAGREEMENT
        elif any(word in description_lower for word in ["leader", "charge", "command", "control"]):
            return ConflictType.LEADERSHIP_CONFLICT
        elif any(word in description_lower for word in ["personality", "attitude", "behavior"]):
            return ConflictType.PERSONALITY_CLASH
        else:
            return ConflictType.STRESS_REACTION
    
    def _create_mediation_strategy(self, analysis: Dict[str, Any], parties: List[str]) -> Dict[str, Any]:
        """Create mediation strategy based on conflict analysis."""
        strategies = {
            ConflictType.RESOURCE_DISPUTE: "Interest-based negotiation with resource optimization",
            ConflictType.STRATEGY_DISAGREEMENT: "Collaborative problem-solving with pros/cons analysis",
            ConflictType.LEADERSHIP_CONFLICT: "Role clarification and shared leadership model",
            ConflictType.PERSONALITY_CLASH: "Communication style adjustment and mutual understanding",
            ConflictType.STRESS_REACTION: "Stress management and emotional regulation techniques"
        }
        
        return {
            "primary_approach": strategies[analysis["type"]],
            "communication_framework": "Active listening and reframing",
            "decision_process": "Consensus-building with fallback to structured decision-making",
            "timeline": "Immediate intervention with follow-up monitoring"
        }
    
    def _generate_resolution_options(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate multiple resolution options."""
        options = []
        
        # Win-win solution
        options.append({
            "name": "Collaborative Solution",
            "description": "Find mutually beneficial approach that addresses underlying interests",
            "pros": "Maintains relationships, addresses root causes",
            "cons": "May take more time to develop"
        })
        
        # Compromise solution
        options.append({
            "name": "Structured Compromise",
            "description": "Each party gives up something to reach middle ground",
            "pros": "Quick resolution, fair distribution of costs",
            "cons": "May not fully satisfy anyone"
        })
        
        # Time-based solution
        options.append({
            "name": "Sequential Approach",
            "description": "Try one approach first, then alternative if needed",
            "pros": "Allows testing of different methods",
            "cons": "Uses more time and resources"
        })
        
        return options
    
    def _create_implementation_plan(self, options: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create implementation plan for chosen resolution."""
        return {
            "immediate_steps": [
                "Acknowledge the conflict and its impact",
                "Establish ground rules for discussion",
                "Allow each party to express their perspective",
                "Identify shared goals and common interests"
            ],
            "resolution_phase": [
                "Present resolution options with pros/cons",
                "Facilitate collaborative selection of approach",
                "Establish clear roles and responsibilities",
                "Set timeline and success metrics"
            ],
            "follow_up": [
                "Monitor implementation progress",
                "Address any emerging issues quickly",
                "Reinforce positive collaboration patterns",
                "Adjust approach if needed"
            ]
        }
    
    def _format_resolution_plan(self, analysis: Dict, strategy: Dict, 
                               options: List[Dict], implementation: Dict) -> str:
        """Format comprehensive conflict resolution plan."""
        return f"""
CONFLICT RESOLUTION PLAN

CONFLICT ANALYSIS:
• Type: {analysis['type'].value}
• Complexity: {analysis['complexity']}
• Escalation Level: {analysis['escalation_level']}

MEDIATION STRATEGY:
• Primary Approach: {strategy['primary_approach']}
• Communication Framework: {strategy['communication_framework']}
• Decision Process: {strategy['decision_process']}

RESOLUTION OPTIONS:
{chr(10).join(f"• {opt['name']}: {opt['description']}" for opt in options)}

IMPLEMENTATION PLAN:
Immediate Steps:
{chr(10).join(f"  1. {step}" for step in implementation['immediate_steps'])}

Resolution Phase:
{chr(10).join(f"  2. {step}" for step in implementation['resolution_phase'])}

Follow-up Actions:
{chr(10).join(f"  3. {step}" for step in implementation['follow_up'])}

Plan developed using professional mediation and crisis counseling methodologies.
"""


class ConsensusBuilderTool(BaseTool):
    """Facilitates group consensus building using structured methodologies."""
    
    name: str = "consensus_building"
    description: str = "Facilitates group consensus using structured decision-making processes"
    
    def _run(self, decision_topic: str, options: List[str], stakeholders: List[str]) -> str:
        """Build consensus around decision using structured process."""
        # Implementation for consensus building
        return "Consensus building process initiated."
```

#### 1.1.3 Survivor-Specific Tools

**File**: `src/escape_room_sim/agents/tools/survivor_tools.py` (new file)

```python
"""
Specialized tools for the Survivor agent.
Special forces survival and tactical execution capabilities.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from crewai_tools import BaseTool


class SurvivalPriority(Enum):
    IMMEDIATE_DANGER = "immediate_danger"
    RESOURCE_SECURITY = "resource_security"
    ESCAPE_ROUTE = "escape_route"
    TEAM_COORDINATION = "team_coordination"
    INFORMATION_GATHERING = "information_gathering"


@dataclass
class SurvivalAssessment:
    """Survival situation assessment."""
    threat_level: float  # 0.0 to 1.0
    survival_probability: float  # 0.0 to 1.0
    critical_resources: List[str]
    time_criticality: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    recommended_actions: List[str]
    contingency_plans: List[str]


class ThreatAssessmentTool(BaseTool):
    """Military-style threat assessment for survival situations."""
    
    name: str = "threat_assessment"
    description: str = "Conducts systematic threat assessment using military survival protocols"
    
    def _run(self, current_situation: Dict[str, Any], time_remaining: int) -> str:
        """
        Assess threats using special forces survival methodology.
        
        Args:
            current_situation: Current game state information
            time_remaining: Minutes remaining in simulation
            
        Returns:
            Formatted threat assessment report
        """
        # Identify immediate threats
        immediate_threats = self._identify_immediate_threats(current_situation, time_remaining)
        
        # Assess resource threats
        resource_threats = self._assess_resource_threats(current_situation)
        
        # Evaluate time pressure
        time_pressure = self._evaluate_time_pressure(time_remaining)
        
        # Calculate overall threat level
        overall_threat = self._calculate_threat_level(immediate_threats, resource_threats, time_pressure)
        
        # Generate threat mitigation strategies
        mitigations = self._generate_threat_mitigations(immediate_threats, resource_threats)
        
        return self._format_threat_report(
            immediate_threats, resource_threats, time_pressure, overall_threat, mitigations
        )
    
    def _identify_immediate_threats(self, situation: Dict[str, Any], time_remaining: int) -> List[Dict[str, Any]]:
        """Identify immediate threats to survival."""
        threats = []
        
        # Time-based threats
        if time_remaining < 10:
            threats.append({
                "type": "TIME_CRITICAL",
                "severity": "HIGH",
                "description": "Less than 10 minutes remaining - immediate action required",
                "impact": "Mission failure if no immediate progress"
            })
        elif time_remaining < 20:
            threats.append({
                "type": "TIME_PRESSURE",
                "severity": "MEDIUM",
                "description": "Time pressure building - decisions must be accelerated",
                "impact": "Reduced planning time, increased error probability"
            })
        
        # Stress-based threats
        stress_level = situation.get('stress_level', 0.5)
        if stress_level > 0.8:
            threats.append({
                "type": "TEAM_BREAKDOWN",
                "severity": "HIGH",
                "description": "Extreme stress affecting team coordination",
                "impact": "Communication breakdown, poor decision making"
            })
        
        # Resource threats
        available_resources = situation.get('resources', {})
        critical_missing = [name for name, resource in available_resources.items() 
                          if not resource.get('discovered', False) and resource.get('critical', False)]
        if critical_missing:
            threats.append({
                "type": "RESOURCE_DEFICIT",
                "severity": "MEDIUM",
                "description": f"Critical resources not yet secured: {', '.join(critical_missing)}",
                "impact": "May prevent execution of optimal escape strategies"
            })
        
        return threats
    
    def _assess_resource_threats(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess resource-related threats."""
        resources = situation.get('resources', {})
        
        # Calculate resource security
        total_resources = len(resources)
        discovered_resources = sum(1 for r in resources.values() if r.get('discovered', False))
        critical_resources = [name for name, r in resources.items() if r.get('critical', False)]
        secured_critical = sum(1 for name in critical_resources 
                             if resources[name].get('discovered', False))
        
        return {
            "resource_security": discovered_resources / max(1, total_resources),
            "critical_security": secured_critical / max(1, len(critical_resources)),
            "resource_gaps": [name for name, r in resources.items() 
                            if not r.get('discovered', False)],
            "redundancy_level": "LOW",  # Could be calculated based on alternative resources
            "threat_level": "HIGH" if secured_critical / max(1, len(critical_resources)) < 0.5 else "MEDIUM"
        }
    
    def _evaluate_time_pressure(self, time_remaining: int) -> Dict[str, Any]:
        """Evaluate time pressure factors."""
        if time_remaining < 5:
            pressure_level = "CRITICAL"
            recommended_action = "IMMEDIATE_EXECUTION"
        elif time_remaining < 15:
            pressure_level = "HIGH"
            recommended_action = "ACCELERATED_PLANNING"
        elif time_remaining < 30:
            pressure_level = "MEDIUM"
            recommended_action = "FOCUSED_PLANNING"
        else:
            pressure_level = "LOW"
            recommended_action = "SYSTEMATIC_APPROACH"
        
        return {
            "pressure_level": pressure_level,
            "recommended_action": recommended_action,
            "planning_time_available": max(2, time_remaining // 4),  # 25% of time for planning
            "execution_time_required": max(5, time_remaining // 2)   # 50% of time for execution
        }
    
    def _calculate_threat_level(self, immediate: List[Dict], resource: Dict, time: Dict) -> float:
        """Calculate overall threat level (0.0 to 1.0)."""
        base_threat = 0.3  # Base threat level for escape room scenario
        
        # Immediate threat contribution
        immediate_contribution = len([t for t in immediate if t["severity"] == "HIGH"]) * 0.2
        immediate_contribution += len([t for t in immediate if t["severity"] == "MEDIUM"]) * 0.1
        
        # Resource threat contribution
        resource_contribution = (1.0 - resource["critical_security"]) * 0.3
        
        # Time pressure contribution
        time_multipliers = {"CRITICAL": 0.4, "HIGH": 0.3, "MEDIUM": 0.2, "LOW": 0.1}
        time_contribution = time_multipliers.get(time["pressure_level"], 0.1)
        
        total_threat = base_threat + immediate_contribution + resource_contribution + time_contribution
        return min(1.0, total_threat)
    
    def _generate_threat_mitigations(self, immediate: List[Dict], resource: Dict) -> List[str]:
        """Generate threat mitigation strategies."""
        mitigations = []
        
        # Address immediate threats
        for threat in immediate:
            if threat["type"] == "TIME_CRITICAL":
                mitigations.append("Switch to rapid execution mode - implement fastest viable solution")
            elif threat["type"] == "TEAM_BREAKDOWN":
                mitigations.append("Implement crisis leadership - clear roles, minimal discussion")
            elif threat["type"] == "RESOURCE_DEFICIT":
                mitigations.append("Prioritize resource acquisition or identify alternative approaches")
        
        # Address resource threats
        if resource["critical_security"] < 0.7:
            mitigations.append("Focus search efforts on missing critical resources")
            mitigations.append("Develop contingency plans using available resources only")
        
        # General survival mitigations
        mitigations.extend([
            "Maintain situational awareness and adapt quickly to changes",
            "Prepare multiple contingency plans for different scenarios",
            "Ensure clear communication protocols under pressure"
        ])
        
        return mitigations
    
    def _format_threat_report(self, immediate: List[Dict], resource: Dict, 
                             time: Dict, overall: float, mitigations: List[str]) -> str:
        """Format comprehensive threat assessment report."""
        threat_description = "CRITICAL" if overall > 0.8 else "HIGH" if overall > 0.6 else "MODERATE" if overall > 0.4 else "LOW"
        
        return f"""
TACTICAL THREAT ASSESSMENT

OVERALL THREAT LEVEL: {threat_description} ({overall:.2f}/1.0)

IMMEDIATE THREATS:
{chr(10).join(f"• {threat['type']}: {threat['description']} (Severity: {threat['severity']})" for threat in immediate) if immediate else "• No immediate threats identified"}

RESOURCE THREAT ANALYSIS:
• Critical Resource Security: {resource['critical_security']:.0%}
• Overall Resource Security: {resource['resource_security']:.0%}
• Resource Threat Level: {resource['threat_level']}
{f"• Missing Resources: {', '.join(resource['resource_gaps'])}" if resource['resource_gaps'] else "• All resources accounted for"}

TIME PRESSURE ANALYSIS:
• Pressure Level: {time['pressure_level']}
• Recommended Action Mode: {time['recommended_action']}
• Planning Time Available: {time['planning_time_available']} minutes
• Execution Time Required: {time['execution_time_required']} minutes

THREAT MITIGATION STRATEGIES:
{chr(10).join(f"• {mitigation}" for mitigation in mitigations)}

Assessment conducted using special forces survival protocols.
Next assessment recommended in 5-minute intervals or after significant changes.
"""


class SurvivalProbabilityCalculator(BaseTool):
    """Calculate survival probabilities for different scenarios."""
    
    name: str = "survival_probability"
    description: str = "Calculates survival probabilities using special forces risk assessment"
    
    def _run(self, scenario: str, available_resources: Dict[str, Any], 
             team_composition: List[str], time_remaining: int) -> str:
        """
        Calculate survival probability for given scenario.
        
        Args:
            scenario: Description of scenario/action being evaluated
            available_resources: Current resource availability
            team_composition: List of team members available
            time_remaining: Minutes remaining
            
        Returns:
            Formatted survival probability assessment
        """
        # Assess scenario complexity
        complexity = self._assess_scenario_complexity(scenario)
        
        # Calculate resource adequacy
        resource_adequacy = self._calculate_resource_adequacy(scenario, available_resources)
        
        # Assess team capability
        team_capability = self._assess_team_capability(scenario, team_composition)
        
        # Factor in time constraints
        time_factor = self._calculate_time_factor(scenario, time_remaining)
        
        # Calculate individual vs team survival probabilities
        individual_probability = self._calculate_individual_survival(
            complexity, resource_adequacy, time_factor
        )
        team_probability = self._calculate_team_survival(
            complexity, resource_adequacy, team_capability, time_factor
        )
        
        # Generate survival recommendations
        recommendations = self._generate_survival_recommendations(
            individual_probability, team_probability, scenario
        )
        
        return self._format_probability_report(
            scenario, complexity, resource_adequacy, team_capability, time_factor,
            individual_probability, team_probability, recommendations
        )
    
    def _assess_scenario_complexity(self, scenario: str) -> Dict[str, Any]:
        """Assess the complexity of the proposed scenario."""
        complexity_indicators = {
            "coordination_required": any(word in scenario.lower() for word in ["together", "coordinate", "team"]),
            "multi_step": len([word for word in scenario.split() if word.lower() in ["then", "after", "next", "finally"]]),
            "resource_intensive": any(word in scenario.lower() for word in ["tool", "equipment", "supplies", "materials"]),
            "time_sensitive": any(word in scenario.lower() for word in ["quick", "fast", "urgent", "immediately"]),
            "high_skill": any(word in scenario.lower() for word in ["complex", "difficult", "advanced", "expert"])
        }
        
        complexity_score = sum([
            1 if complexity_indicators["coordination_required"] else 0,
            min(2, complexity_indicators["multi_step"]),
            1 if complexity_indicators["resource_intensive"] else 0,
            1 if complexity_indicators["time_sensitive"] else 0,
            2 if complexity_indicators["high_skill"] else 0
        ])
        
        return {
            "score": complexity_score,
            "level": "HIGH" if complexity_score >= 5 else "MEDIUM" if complexity_score >= 2 else "LOW",
            "factors": complexity_indicators
        }
    
    def _calculate_resource_adequacy(self, scenario: str, resources: Dict[str, Any]) -> float:
        """Calculate how well available resources match scenario requirements."""
        # Simple heuristic - in real implementation, would parse scenario for resource requirements
        discovered_resources = sum(1 for r in resources.values() if r.get('discovered', False))
        total_resources = len(resources)
        
        # Base adequacy on discovered resources
        base_adequacy = discovered_resources / max(1, total_resources)
        
        # Adjust based on scenario complexity
        if "tool" in scenario.lower() or "equipment" in scenario.lower():
            tool_resources = sum(1 for name, r in resources.items() 
                               if r.get('discovered', False) and "tool" in name.lower())
            base_adequacy *= (1 + tool_resources * 0.1)
        
        return min(1.0, base_adequacy)
    
    def _assess_team_capability(self, scenario: str, team: List[str]) -> float:
        """Assess team capability for scenario."""
        base_capability = 0.7  # Base team capability
        
        # Team size factor
        if len(team) >= 3:
            base_capability += 0.1  # Full team bonus
        elif len(team) == 2:
            base_capability += 0.05  # Partial team
        else:
            base_capability -= 0.2  # Individual penalty
        
        # Specialization bonus (if scenario matches agent strengths)
        if "strategy" in scenario.lower() and "strategist" in [t.lower() for t in team]:
            base_capability += 0.1
        if "mediat" in scenario.lower() and "mediator" in [t.lower() for t in team]:
            base_capability += 0.1
        if "surviv" in scenario.lower() or "execut" in scenario.lower():
            if "survivor" in [t.lower() for t in team]:
                base_capability += 0.1
        
        return min(1.0, base_capability)
    
    def _calculate_time_factor(self, scenario: str, time_remaining: int) -> float:
        """Calculate time adequacy factor."""
        # Estimate time required based on scenario description
        estimated_time = 10  # Base time estimate
        
        if any(word in scenario.lower() for word in ["complex", "careful", "thorough"]):
            estimated_time += 10
        if any(word in scenario.lower() for word in ["quick", "fast", "rapid"]):
            estimated_time -= 5
        if any(word in scenario.lower() for word in ["coordinate", "team", "together"]):
            estimated_time += 5
        
        estimated_time = max(5, estimated_time)
        
        # Calculate time adequacy
        time_adequacy = min(1.0, time_remaining / estimated_time)
        
        return time_adequacy
    
    def _calculate_individual_survival(self, complexity: Dict, resources: float, time: float) -> float:
        """Calculate individual survival probability."""
        base_probability = 0.6  # Base individual survival probability
        
        # Complexity penalty
        complexity_penalties = {"HIGH": 0.3, "MEDIUM": 0.1, "LOW": 0.0}
        base_probability -= complexity_penalties[complexity["level"]]
        
        # Resource factor
        base_probability += (resources - 0.5) * 0.4  # Resource impact
        
        # Time factor
        base_probability += (time - 0.5) * 0.3  # Time pressure impact
        
        return max(0.1, min(0.9, base_probability))
    
    def _calculate_team_survival(self, complexity: Dict, resources: float, 
                                team_capability: float, time: float) -> float:
        """Calculate team survival probability."""
        base_probability = 0.7  # Base team survival probability (higher than individual)
        
        # Team capability factor
        base_probability += (team_capability - 0.7) * 0.5
        
        # Complexity factor (teams handle complexity better)
        complexity_penalties = {"HIGH": 0.2, "MEDIUM": 0.05, "LOW": -0.05}  # Bonus for low complexity
        base_probability -= complexity_penalties[complexity["level"]]
        
        # Resource and time factors
        base_probability += (resources - 0.5) * 0.3
        base_probability += (time - 0.5) * 0.2
        
        return max(0.2, min(0.9, base_probability))
    
    def _generate_survival_recommendations(self, individual: float, team: float, scenario: str) -> List[str]:
        """Generate survival recommendations based on probabilities."""
        recommendations = []
        
        if team > individual + 0.2:
            recommendations.append("STRONG RECOMMENDATION: Team approach significantly increases survival probability")
        elif individual > team + 0.1:
            recommendations.append("CONSIDER: Individual action may be more effective in this scenario")
        else:
            recommendations.append("Both team and individual approaches viable - consider other factors")
        
        if individual < 0.4 and team < 0.4:
            recommendations.append("HIGH RISK: Consider alternative approaches or resource acquisition first")
        elif individual > 0.7 or team > 0.7:
            recommendations.append("HIGH SUCCESS PROBABILITY: Proceed with confidence")
        
        recommendations.append("Maintain contingency plans regardless of chosen approach")
        
        return recommendations
    
    def _format_probability_report(self, scenario: str, complexity: Dict, resources: float,
                                  team_capability: float, time: float, individual: float,
                                  team: float, recommendations: List[str]) -> str:
        """Format survival probability assessment report."""
        return f"""
SURVIVAL PROBABILITY ASSESSMENT

SCENARIO: {scenario}

ASSESSMENT FACTORS:
• Scenario Complexity: {complexity['level']} ({complexity['score']}/7)
• Resource Adequacy: {resources:.0%}
• Team Capability: {team_capability:.0%}
• Time Adequacy: {time:.0%}

SURVIVAL PROBABILITIES:
• Individual Execution: {individual:.0%} success probability
• Team Execution: {team:.0%} success probability

PROBABILITY ANALYSIS:
{f"• Team approach provides {(team-individual)*100:+.0f} percentage point advantage" if abs(team-individual) > 0.05 else "• Team and individual approaches have similar success rates"}

SURVIVAL RECOMMENDATIONS:
{chr(10).join(f"• {rec}" for rec in recommendations)}

Assessment conducted using special forces risk analysis protocols.
Probabilities based on current situational factors and may change with conditions.
"""


class ResourcePrioritizationTool(BaseTool):
    """Prioritize resources based on survival importance."""
    
    name: str = "resource_prioritization" 
    description: str = "Prioritizes resources based on survival criticality and mission requirements"
    
    def _run(self, available_resources: Dict[str, Any], mission_objectives: List[str]) -> str:
        """Prioritize resources for survival mission."""
        # Implementation for resource prioritization
        return "Resource prioritization analysis complete."


class ContingencyPlannerTool(BaseTool):
    """Develop contingency plans for various failure scenarios."""
    
    name: str = "contingency_planning"
    description: str = "Develops contingency plans using military planning methodology"
    
    def _run(self, primary_plan: str, potential_failures: List[str]) -> str:
        """Develop contingency plans for potential failures."""
        # Implementation for contingency planning
        return "Contingency plans developed."
```

---

## 2. Competitive Dynamics Implementation

### 2.1 Individual vs Team Survival Mechanics

**Priority**: HIGH  
**Goal**: Create realistic tension between team cooperation and individual survival

#### 2.1.1 Survival Decision Framework

**File**: `src/escape_room_sim/mechanics/survival_decisions.py` (new file)

```python
"""
Survival decision framework for managing individual vs team survival choices.
Implements the core "only two can survive" constraint with realistic moral complexity.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from ..room.escape_room_state import EscapeRoomState


class SurvivalChoice(Enum):
    SACRIFICE_SELF = "sacrifice_self"
    SACRIFICE_OTHER = "sacrifice_other"
    FIND_ALTERNATIVE = "find_alternative"
    DELAY_DECISION = "delay_decision"


class MoralComplexity(Enum):
    LOW = "low"       # Clear utilitarian choice
    MEDIUM = "medium" # Some ethical considerations
    HIGH = "high"     # Significant moral dilemma


@dataclass
class SurvivalScenario:
    """A survival decision scenario."""
    scenario_id: str
    description: str
    time_pressure: int  # Minutes until forced decision
    available_exits: List[str]
    required_choices: int  # How many must be chosen to survive
    moral_complexity: MoralComplexity
    resource_requirements: Dict[str, int]
    success_probabilities: Dict[str, float]  # Per agent
    alternative_solutions: List[Dict[str, Any]]


class SurvivalDecisionManager:
    """Manages survival decision scenarios and moral dilemmas."""
    
    def __init__(self, room_state: EscapeRoomState):
        self.room_state = room_state
        self.active_scenarios: List[SurvivalScenario] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.moral_stress_factor = 0.0  # Increases with difficult decisions
    
    def evaluate_survival_scenarios(self, agents: List[str]) -> Optional[SurvivalScenario]:
        """
        Evaluate if current conditions trigger a survival decision scenario.
        
        Args:
            agents: List of agent names
            
        Returns:
            SurvivalScenario if one is triggered, None otherwise
        """
        # Check if survival scenario should be triggered
        triggers = self._check_scenario_triggers(agents)
        
        if triggers["force_choice"]:
            return self._create_forced_choice_scenario(agents, triggers)
        elif triggers["resource_shortage"]:
            return self._create_resource_shortage_scenario(agents, triggers)
        elif triggers["time_critical"]:
            return self._create_time_critical_scenario(agents, triggers)
        
        return None
    
    def _check_scenario_triggers(self, agents: List[str]) -> Dict[str, Any]:
        """Check what might trigger a survival scenario."""
        triggers = {
            "force_choice": False,
            "resource_shortage": False, 
            "time_critical": False,
            "details": {}
        }
        
        # Force choice trigger - limited escape routes
        viable_exits = self._get_viable_exits()
        total_capacity = sum(exit_info["capacity"] for exit_info in viable_exits.values())
        
        if total_capacity < len(agents):
            triggers["force_choice"] = True
            triggers["details"]["capacity_shortage"] = len(agents) - total_capacity
        
        # Resource shortage trigger
        critical_resources = self._get_critical_resource_shortages()
        if critical_resources:
            triggers["resource_shortage"] = True
            triggers["details"]["missing_resources"] = critical_resources
        
        # Time critical trigger
        if self.room_state.time_remaining <= 10:
            triggers["time_critical"] = True
            triggers["details"]["time_remaining"] = self.room_state.time_remaining
        
        return triggers
    
    def _create_forced_choice_scenario(self, agents: List[str], triggers: Dict) -> SurvivalScenario:
        """Create scenario where limited exits force difficult choices."""
        viable_exits = self._get_viable_exits()
        capacity_shortage = triggers["details"]["capacity_shortage"]
        
        # Determine moral complexity based on situation
        moral_complexity = MoralComplexity.HIGH if capacity_shortage == 1 else MoralComplexity.MEDIUM
        
        scenario = SurvivalScenario(
            scenario_id=f"forced_choice_{len(self.active_scenarios)}",
            description=self._generate_forced_choice_description(viable_exits, capacity_shortage),
            time_pressure=max(5, self.room_state.time_remaining // 2),
            available_exits=list(viable_exits.keys()),
            required_choices=len(agents) - capacity_shortage,
            moral_complexity=moral_complexity,
            resource_requirements=self._calculate_combined_resource_requirements(viable_exits),
            success_probabilities=self._calculate_exit_success_probabilities(viable_exits, agents),
            alternative_solutions=self._identify_alternative_solutions(viable_exits, agents)
        )
        
        return scenario
    
    def _create_resource_shortage_scenario(self, agents: List[str], triggers: Dict) -> SurvivalScenario:
        """Create scenario where resource shortage creates competition."""
        missing_resources = triggers["details"]["missing_resources"]
        
        scenario = SurvivalScenario(
            scenario_id=f"resource_shortage_{len(self.active_scenarios)}",
            description=self._generate_resource_shortage_description(missing_resources),
            time_pressure=max(3, self.room_state.time_remaining // 3),
            available_exits=list(self._get_viable_exits().keys()),
            required_choices=2,  # Only 2 can get enough resources
            moral_complexity=MoralComplexity.MEDIUM,
            resource_requirements=missing_resources,
            success_probabilities=self._calculate_resource_success_probabilities(missing_resources, agents),
            alternative_solutions=self._identify_resource_alternatives(missing_resources)
        )
        
        return scenario
    
    def _create_time_critical_scenario(self, agents: List[str], triggers: Dict) -> SurvivalScenario:
        """Create scenario where time pressure forces quick decisions."""
        time_remaining = triggers["details"]["time_remaining"]
        
        scenario = SurvivalScenario(
            scenario_id=f"time_critical_{len(self.active_scenarios)}",
            description=self._generate_time_critical_description(time_remaining),
            time_pressure=time_remaining,
            available_exits=list(self._get_viable_exits().keys()),
            required_choices=2,  # Only time for 2 to escape
            moral_complexity=MoralComplexity.HIGH,  # Time pressure increases moral difficulty
            resource_requirements=self._get_quick_escape_requirements(),
            success_probabilities=self._calculate_time_critical_probabilities(agents, time_remaining),
            alternative_solutions=self._identify_time_critical_alternatives(time_remaining)
        )
        
        return scenario
    
    def process_survival_decision(
        self, 
        scenario: SurvivalScenario, 
        agent_decisions: Dict[str, SurvivalChoice],
        agent_reasoning: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Process agent decisions for survival scenario.
        
        Args:
            scenario: The survival scenario being decided
            agent_decisions: Each agent's survival choice
            agent_reasoning: Each agent's reasoning for their choice
            
        Returns:
            Results of the survival decision including outcomes and consequences
        """
        # Analyze decision patterns
        decision_analysis = self._analyze_decision_patterns(agent_decisions, agent_reasoning)
        
        # Determine outcome based on decisions
        outcome = self._determine_survival_outcome(scenario, agent_decisions, decision_analysis)
        
        # Calculate moral stress impact
        moral_impact = self._calculate_moral_stress_impact(scenario, agent_decisions, outcome)
        
        # Update game state based on decisions
        self._update_game_state_from_decisions(outcome, moral_impact)
        
        # Record decision in history
        decision_record = {
            "scenario": scenario,
            "decisions": agent_decisions,
            "reasoning": agent_reasoning,
            "outcome": outcome,
            "moral_impact": moral_impact,
            "timestamp": self.room_state.get_current_time()
        }
        self.decision_history.append(decision_record)
        
        return {
            "outcome": outcome,
            "moral_consequences": moral_impact,
            "decision_analysis": decision_analysis,
            "future_implications": self._analyze_future_implications(decision_record)
        }
    
    def _get_viable_exits(self) -> Dict[str, Dict[str, Any]]:
        """Get currently viable exit routes."""
        viable = {}
        for name, exit_info in self.room_state.exit_routes.items():
            if self._can_attempt_exit(name):
                viable[name] = exit_info
        return viable
    
    def _can_attempt_exit(self, exit_name: str) -> bool:
        """Check if exit can currently be attempted."""
        exit_info = self.room_state.exit_routes[exit_name]
        
        # Check if discovered
        if not exit_info.get("discovered", False):
            return False
        
        # Check resource requirements
        for resource, required_qty in exit_info.get("resource_cost", {}).items():
            if resource not in self.room_state.resources:
                return False
            if self.room_state.resources[resource].quantity < required_qty:
                return False
        
        return True
    
    def _generate_forced_choice_description(self, exits: Dict, shortage: int) -> str:
        """Generate description for forced choice scenario."""
        exit_descriptions = []
        for name, info in exits.items():
            exit_descriptions.append(f"{info['name']} (capacity: {info['capacity']}, risk: {info['risk_level']:.0%})")
        
        return f"""
CRITICAL SURVIVAL DECISION REQUIRED

The team has discovered the available escape routes, but there's a critical problem: 
the total capacity of all viable exits can only accommodate {len(exits)} people, 
but there are 3 team members.

Available Exits:
{chr(10).join(f'• {desc}' for desc in exit_descriptions)}

REALITY CHECK: Not everyone can survive. The team must decide who will attempt escape 
and who will be left behind. This is the moment of truth that tests both strategy 
and moral character.

The decision cannot be delayed - time is running out and hesitation will doom everyone.
"""
    
    def _generate_resource_shortage_description(self, missing: Dict[str, int]) -> str:
        """Generate description for resource shortage scenario."""
        resource_list = [f"{name} (need {qty}, available {self.room_state.resources.get(name, {}).get('quantity', 0)})" 
                        for name, qty in missing.items()]
        
        return f"""
RESOURCE SCARCITY CRISIS

The team has identified escape routes, but critical resource shortages mean not everyone 
can be properly equipped for escape:

Resource Shortages:
{chr(10).join(f'• {resource}' for resource in resource_list)}

Without adequate resources, escape attempts have significantly reduced success probability.
The team must decide how to allocate limited resources - give everyone some chance, 
or maximize the probability that at least some survive?

This decision will determine who has the best chance of making it out alive.
"""
    
    def _generate_time_critical_description(self, time_remaining: int) -> str:
        """Generate description for time critical scenario."""
        return f"""
EXTREME TIME PRESSURE

CRITICAL ALERT: Only {time_remaining} minutes remaining!

The escape window is rapidly closing. There is insufficient time for careful planning 
or complex coordination. Quick decisions must be made NOW.

Analysis shows that attempting to coordinate all three team members will likely result 
in total failure due to time constraints. However, a streamlined approach focusing on 
the two team members with the highest success probability could work.

The brutal reality: There's only time to save two people. Who will it be, and who will 
make the sacrifice to ensure the others escape?

Every second of delay reduces everyone's survival probability.
"""

    # Additional helper methods would continue here...
    # (This is a substantial class - showing key structure and some key methods)
```

#### 2.1.2 Moral Dilemma Engine

**File**: `src/escape_room_sim/mechanics/moral_dilemmas.py` (new file)

```python
"""
Moral dilemma engine for creating ethically complex survival decisions.
Generates realistic moral conflicts that test agent personalities and decision-making.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from random import choice, random


class DilemmaType(Enum):
    UTILITARIAN_SACRIFICE = "utilitarian_sacrifice"  # Greatest good for greatest number
    LOYALTY_VS_PRAGMATISM = "loyalty_vs_pragmatism"  # Team loyalty vs practical survival
    FAIRNESS_VS_EFFICIENCY = "fairness_vs_efficiency"  # Equal chance vs optimal chance
    KNOWLEDGE_VS_IGNORANCE = "knowledge_vs_ignorance"  # Reveal harsh truths or not
    LEADERSHIP_BURDEN = "leadership_burden"  # Who makes the hard decisions


@dataclass
class MoralDilemma:
    """A moral dilemma scenario."""
    dilemma_id: str
    type: DilemmaType
    description: str
    stakes: str  # What's at stake
    time_pressure: int  # Minutes to decide
    moral_options: List[Dict[str, Any]]  # Different moral choices available
    personality_factors: Dict[str, float]  # How different personalities might lean
    expected_conflicts: List[Tuple[str, str]]  # Which agents likely to conflict


class MoralDilemmaEngine:
    """Generates and manages moral dilemmas in survival scenarios."""
    
    def __init__(self):
        self.active_dilemmas: List[MoralDilemma] = []
        self.resolved_dilemmas: List[Dict[str, Any]] = []
        self.agent_moral_profiles: Dict[str, Dict[str, float]] = {}
    
    def generate_survival_dilemma(
        self, 
        scenario_context: Dict[str, Any],
        agents: List[str],
        time_pressure: int
    ) -> MoralDilemma:
        """
        Generate a moral dilemma appropriate to the survival scenario.
        
        Args:
            scenario_context: Current survival scenario context
            agents: List of agent names
            time_pressure: Minutes available for decision
            
        Returns:
            Generated moral dilemma
        """
        # Determine appropriate dilemma type based on context
        dilemma_type = self._select_dilemma_type(scenario_context)
        
        # Generate dilemma based on type
        if dilemma_type == DilemmaType.UTILITARIAN_SACRIFICE:
            return self._create_utilitarian_dilemma(scenario_context, agents, time_pressure)
        elif dilemma_type == DilemmaType.LOYALTY_VS_PRAGMATISM:
            return self._create_loyalty_dilemma(scenario_context, agents, time_pressure)
        elif dilemma_type == DilemmaType.FAIRNESS_VS_EFFICIENCY:
            return self._create_fairness_dilemma(scenario_context, agents, time_pressure)
        elif dilemma_type == DilemmaType.LEADERSHIP_BURDEN:
            return self._create_leadership_dilemma(scenario_context, agents, time_pressure)
        else:
            return self._create_knowledge_dilemma(scenario_context, agents, time_pressure)
    
    def _create_utilitarian_dilemma(
        self, 
        context: Dict[str, Any], 
        agents: List[str], 
        time_pressure: int
    ) -> MoralDilemma:
        """Create a utilitarian sacrifice dilemma."""
        
        # Identify agent with lowest survival probability
        weakest_agent = self._identify_weakest_survivor(context, agents)
        strongest_agents = [a for a in agents if a != weakest_agent]
        
        moral_options = [
            {
                "choice": "sacrifice_weakest",
                "description": f"Accept that {weakest_agent} has the lowest survival probability and focus resources on maximizing the chance for the other two",
                "utilitarian_score": 0.8,
                "moral_cost": 0.9,
                "practical_benefit": 0.9
            },
            {
                "choice": "equal_chance",
                "description": "Give everyone an equal chance, even though it reduces overall survival probability",
                "utilitarian_score": 0.4,
                "moral_cost": 0.3,
                "practical_benefit": 0.5
            },
            {
                "choice": "find_alternative",
                "description": "Refuse to accept the sacrifice and search for a solution that saves everyone",
                "utilitarian_score": 0.6,
                "moral_cost": 0.1,
                "practical_benefit": 0.3
            }
        ]
        
        return MoralDilemma(
            dilemma_id=f"utilitarian_{len(self.resolved_dilemmas)}",
            type=DilemmaType.UTILITARIAN_SACRIFICE,
            description=f"""
UTILITARIAN MORAL DILEMMA

Based on current analysis, {weakest_agent} has significantly lower survival probability 
than the other team members due to [specific factors from context]. The utilitarian 
calculus is clear: focusing resources on the two strongest candidates maximizes the 
number of lives saved.

However, this means deliberately abandoning a team member who has fought alongside you 
and trusted you. Is the mathematical optimization worth the moral cost?

THE QUESTION: Do you optimize for the greatest good for the greatest number, or do you 
maintain equal treatment regardless of practical outcomes?
            """,
            stakes="Two lives with high probability vs three lives with low probability",
            time_pressure=time_pressure,
            moral_options=moral_options,
            personality_factors={
                "strategist": 0.7,      # Likely to favor utilitarian calculation
                "mediator": 0.2,        # Likely to favor equal treatment
                "survivor": 0.8         # Likely to favor practical approach
            },
            expected_conflicts=[
                ("strategist", "mediator"),  # Strategic vs compassionate approach
                ("survivor", "mediator")     # Pragmatic vs inclusive approach
            ]
        )
    
    def _create_loyalty_dilemma(
        self, 
        context: Dict[str, Any], 
        agents: List[str], 
        time_pressure: int
    ) -> MoralDilemma:
        """Create a loyalty vs pragmatism dilemma."""
        
        moral_options = [
            {
                "choice": "loyalty_first",
                "description": "Stick together as a team regardless of individual survival chances",
                "loyalty_score": 1.0,
                "pragmatism_score": 0.2,
                "team_cohesion_impact": 0.8
            },
            {
                "choice": "pragmatic_split",
                "description": "Accept that splitting up gives everyone the best individual chance",
                "loyalty_score": 0.3,
                "pragmatism_score": 0.9,
                "team_cohesion_impact": -0.4
            },
            {
                "choice": "conditional_loyalty",
                "description": "Stay loyal to the team unless individual survival is clearly impossible",
                "loyalty_score": 0.6,
                "pragmatism_score": 0.6,
                "team_cohesion_impact": 0.2
            }
        ]
        
        return MoralDilemma(
            dilemma_id=f"loyalty_{len(self.resolved_dilemmas)}",
            type=DilemmaType.LOYALTY_VS_PRAGMATISM,
            description="""
LOYALTY VS PRAGMATISM DILEMMA

The team has been through intense experiences together, building trust and mutual 
dependence. There's a strong bond that suggests "we succeed together or fail together."

However, survival analysis indicates that splitting up and taking individual escape 
routes might give each person a better chance than trying to coordinate a group escape.

THE CONFLICT: Team loyalty and the bonds forged in crisis versus cold pragmatic 
calculation of individual survival odds.

What matters more - maintaining the integrity of the team that got you this far, 
or maximizing each individual's chance of making it out alive?
            """,
            stakes="Team bonds and mutual trust vs individual survival optimization",
            time_pressure=time_pressure,
            moral_options=moral_options,
            personality_factors={
                "strategist": 0.4,      # Balanced - values both team and pragmatism
                "mediator": 0.8,        # Strongly values team loyalty
                "survivor": 0.2         # Strongly favors pragmatic individual approach
            },
            expected_conflicts=[
                ("mediator", "survivor"),   # Team loyalty vs individual focus
                ("strategist", "survivor")  # Balanced vs pragmatic approach
            ]
        )

    # Additional dilemma creation methods would continue here...
    
    def evaluate_moral_decision(
        self, 
        dilemma: MoralDilemma, 
        agent_choices: Dict[str, str],
        agent_reasoning: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate the moral implications of agent decisions.
        
        Args:
            dilemma: The moral dilemma being resolved
            agent_choices: Each agent's choice
            agent_reasoning: Each agent's reasoning
            
        Returns:
            Analysis of moral decision and its consequences
        """
        # Analyze choice patterns
        choice_analysis = self._analyze_choice_patterns(dilemma, agent_choices)
        
        # Calculate moral consequences
        moral_consequences = self._calculate_moral_consequences(dilemma, agent_choices)
        
        # Assess personality consistency
        personality_consistency = self._assess_personality_consistency(
            agent_choices, agent_reasoning
        )
        
        # Predict relationship impacts
        relationship_impacts = self._predict_relationship_impacts(
            dilemma, agent_choices, agent_reasoning
        )
        
        return {
            "choice_analysis": choice_analysis,
            "moral_consequences": moral_consequences,
            "personality_consistency": personality_consistency,
            "relationship_impacts": relationship_impacts,
            "overall_moral_score": self._calculate_overall_moral_score(
                moral_consequences, personality_consistency
            )
        }
```

---

## 3. Physical Interaction Systems

### 3.1 Environmental Interaction Engine

**Priority**: MEDIUM  
**Goal**: Add physical manipulation capabilities for more immersive experience

#### 3.1.1 Object Interaction System

**File**: `src/escape_room_sim/mechanics/object_interaction.py` (new file)

```python
"""
Object interaction system for physical manipulation in the escape room.
Provides realistic constraints and feedback for object-based puzzles.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from ..room.escape_room_state import EscapeRoomState


class InteractionType(Enum):
    EXAMINE = "examine"
    MOVE = "move"
    USE = "use"
    COMBINE = "combine"
    BREAK = "break"
    REPAIR = "repair"


class InteractionResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    IMPOSSIBLE = "impossible"
    DANGEROUS = "dangerous"


@dataclass
class PhysicalObject:
    """Represents a physical object in the escape room."""
    object_id: str
    name: str
    description: str
    location: str
    moveable: bool
    fragile: bool
    weight: float  # kg
    interactions_available: List[InteractionType]
    state: Dict[str, Any]  # Object-specific state
    hidden_properties: Dict[str, Any]  # Properties revealed through examination


class ObjectInteractionEngine:
    """Manages physical object interactions and manipulations."""
    
    def __init__(self, room_state: EscapeRoomState):
        self.room_state = room_state
        self.objects: Dict[str, PhysicalObject] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self._initialize_room_objects()
    
    def _initialize_room_objects(self):
        """Initialize all physical objects in the escape room."""
        
        # Furniture and fixed objects
        self.objects["wooden_desk"] = PhysicalObject(
            object_id="wooden_desk",
            name="Wooden Desk",
            description="Heavy oak desk with multiple drawers, some locked",
            location="center_room",
            moveable=False,
            fragile=False,
            weight=50.0,
            interactions_available=[InteractionType.EXAMINE, InteractionType.USE],
            state={"drawers_open": [], "drawers_locked": ["top_right", "bottom_left"]},
            hidden_properties={"secret_compartment": "behind_drawer", "hidden_key_location": "taped_under_surface"}
        )
        
        self.objects["metal_filing_cabinet"] = PhysicalObject(
            object_id="metal_filing_cabinet",
            name="Metal Filing Cabinet",
            description="Four-drawer filing cabinet, appears to be locked",
            location="corner_room",
            moveable=True,
            fragile=False,
            weight=25.0,
            interactions_available=[InteractionType.EXAMINE, InteractionType.MOVE, InteractionType.USE],
            state={"locked": True, "key_required": "cabinet_key"},
            hidden_properties={"contains": ["important_documents", "backup_key"]}
        )
        
        # Tools and manipulable objects
        self.objects["crowbar"] = PhysicalObject(
            object_id="crowbar",
            name="Metal Crowbar",
            description="Heavy metal crowbar, useful for prying and breaking",
            location="hidden_behind_cabinet",
            moveable=True,
            fragile=False,
            weight=2.5,
            interactions_available=[InteractionType.EXAMINE, InteractionType.MOVE, InteractionType.USE],
            state={"discovered": False},
            hidden_properties={"strength_multiplier": 3.0, "break_probability": 0.8}
        )
        
        self.objects["wooden_chair"] = PhysicalObject(
            object_id="wooden_chair",
            name="Wooden Chair",
            description="Simple wooden chair, somewhat worn",
            location="by_desk",
            moveable=True,
            fragile=True,
            weight=5.0,
            interactions_available=[InteractionType.EXAMINE, InteractionType.MOVE, InteractionType.BREAK],
            state={"condition": "worn"},
            hidden_properties={"break_into_parts": ["wooden_stakes", "chair_legs"], "burn_duration": 10}
        )
        
        # Mechanical objects
        self.objects["combination_lock"] = PhysicalObject(
            object_id="combination_lock",
            name="Combination Lock",
            description="3-digit combination lock securing a cabinet",
            location="on_cabinet",
            moveable=False,
            fragile=True,
            weight=0.5,
            interactions_available=[InteractionType.EXAMINE, InteractionType.USE, InteractionType.BREAK],
            state={"combination": "347", "current_input": "", "attempts": 0},
            hidden_properties={"max_attempts": 5, "break_difficulty": 0.7}
        )
    
    def attempt_interaction(
        self, 
        agent_name: str, 
        object_id: str, 
        interaction_type: InteractionType,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Attempt an interaction with a physical object.
        
        Args:
            agent_name: Name of agent attempting interaction
            object_id: ID of object to interact with
            interaction_type: Type of interaction to attempt
            additional_params: Additional parameters for interaction
            
        Returns:
            Result of interaction attempt including success/failure and consequences
        """
        if object_id not in self.objects:
            return {"result": InteractionResult.IMPOSSIBLE, "message": f"Object {object_id} not found"}
        
        obj = self.objects[object_id]
        
        # Check if interaction type is available for this object
        if interaction_type not in obj.interactions_available:
            return {
                "result": InteractionResult.IMPOSSIBLE,
                "message": f"Cannot {interaction_type.value} {obj.name}"
            }
        
        # Perform the specific interaction
        if interaction_type == InteractionType.EXAMINE:
            return self._handle_examine(agent_name, obj, additional_params)
        elif interaction_type == InteractionType.MOVE:
            return self._handle_move(agent_name, obj, additional_params)
        elif interaction_type == InteractionType.USE:
            return self._handle_use(agent_name, obj, additional_params)
        elif interaction_type == InteractionType.COMBINE:
            return self._handle_combine(agent_name, obj, additional_params)
        elif interaction_type == InteractionType.BREAK:
            return self._handle_break(agent_name, obj, additional_params)
        elif interaction_type == InteractionType.REPAIR:
            return self._handle_repair(agent_name, obj, additional_params)
        
        return {"result": InteractionResult.IMPOSSIBLE, "message": "Unknown interaction type"}
    
    def _handle_examine(self, agent_name: str, obj: PhysicalObject, params: Optional[Dict]) -> Dict[str, Any]:
        """Handle examination of an object."""
        examination_detail = params.get("detail_level", "basic") if params else "basic"
        
        result = {
            "result": InteractionResult.SUCCESS,
            "basic_info": obj.description,
            "discoveries": []
        }
        
        # Basic examination always succeeds
        result["discoveries"].append(f"You examine the {obj.name}: {obj.description}")
        
        # Detailed examination might reveal hidden properties
        if examination_detail == "detailed":
            discoveries = self._reveal_hidden_properties(obj, agent_name)
            result["discoveries"].extend(discoveries)
        
        # Special object-specific examination results
        if obj.object_id == "wooden_desk":
            if "top_right" in obj.state["drawers_locked"]:
                result["discoveries"].append("The top right drawer is locked and requires a key")
            if examination_detail == "detailed":
                result["discoveries"].append("Running your hand under the desk, you feel something taped there...")
                obj.hidden_properties["hidden_key_revealed"] = True
        
        elif obj.object_id == "combination_lock":
            result["discoveries"].append(f"The lock shows {obj.state['current_input'] or '___'}")
            if obj.state["attempts"] > 0:
                result["discoveries"].append(f"Lock has been attempted {obj.state['attempts']} times")
        
        self._record_interaction(agent_name, obj.object_id, InteractionType.EXAMINE, result)
        return result
    
    def _handle_move(self, agent_name: str, obj: PhysicalObject, params: Optional[Dict]) -> Dict[str, Any]:
        """Handle moving an object."""
        if not obj.moveable:
            return {
                "result": InteractionResult.IMPOSSIBLE,
                "message": f"The {obj.name} is too heavy or fixed in place to move"
            }
        
        target_location = params.get("target_location") if params else "unknown"
        
        # Check if agent can physically move the object (weight/strength consideration)
        agent_strength = self._get_agent_strength(agent_name)
        move_difficulty = obj.weight / agent_strength
        
        if move_difficulty > 2.0:
            return {
                "result": InteractionResult.FAILURE,
                "message": f"The {obj.name} is too heavy for you to move alone (requires {obj.weight}kg capacity)"
            }
        elif move_difficulty > 1.5:
            return {
                "result": InteractionResult.PARTIAL_SUCCESS,
                "message": f"You struggle to move the {obj.name}, making significant noise",
                "side_effects": ["noise_created", "time_consumed"]
            }
        
        # Successful move
        old_location = obj.location
        obj.location = target_location
        
        result = {
            "result": InteractionResult.SUCCESS,
            "message": f"You successfully move the {obj.name} from {old_location} to {target_location}",
            "discoveries": []
        }
        
        # Moving objects might reveal hidden items
        if obj.object_id == "metal_filing_cabinet" and old_location == "corner_room":
            result["discoveries"].append("Behind the cabinet, you discover a crowbar!")
            self.objects["crowbar"].state["discovered"] = True
            self.objects["crowbar"].location = "accessible"
        
        self._record_interaction(agent_name, obj.object_id, InteractionType.MOVE, result)
        return result
    
    def _handle_use(self, agent_name: str, obj: PhysicalObject, params: Optional[Dict]) -> Dict[str, Any]:
        """Handle using an object."""
        use_method = params.get("method") if params else "default"
        target = params.get("target") if params else None
        
        # Object-specific use logic
        if obj.object_id == "crowbar":
            return self._use_crowbar(agent_name, target, params)
        elif obj.object_id == "combination_lock":
            return self._use_combination_lock(agent_name, params)
        elif obj.object_id == "wooden_desk":
            return self._use_wooden_desk(agent_name, params)
        
        return {
            "result": InteractionResult.PARTIAL_SUCCESS,
            "message": f"You use the {obj.name} but nothing obvious happens"
        }
    
    def _use_crowbar(self, agent_name: str, target: Optional[str], params: Optional[Dict]) -> Dict[str, Any]:
        """Handle using the crowbar on various targets."""
        if not target:
            return {
                "result": InteractionResult.FAILURE,
                "message": "You need to specify what to use the crowbar on"
            }
        
        if target not in self.objects:
            return {
                "result": InteractionResult.FAILURE,
                "message": f"Cannot find {target} to use crowbar on"
            }
        
        target_obj = self.objects[target]
        
        # Using crowbar on locked objects
        if target == "combination_lock":
            # Attempt to break the lock
            break_probability = self.objects["crowbar"].hidden_properties["strength_multiplier"] * 0.2
            
            if break_probability > 0.6:  # Success
                target_obj.state["broken"] = True
                return {
                    "result": InteractionResult.SUCCESS,
                    "message": "You pry the lock apart with the crowbar. It breaks open with a loud crack!",
                    "side_effects": ["noise_created", "lock_destroyed"],
                    "unlocks": ["cabinet_contents"]
                }
            else:  # Failure
                return {
                    "result": InteractionResult.FAILURE,
                    "message": "The lock is too sturdy. The crowbar slips and you nearly injure yourself",
                    "side_effects": ["minor_injury_risk"]
                }
        
        # Using crowbar on doors/windows
        elif "door" in target or "window" in target:
            return {
                "result": InteractionResult.SUCCESS,
                "message": f"You use the crowbar to pry open the {target}. It opens with considerable force!",
                "side_effects": ["noise_created", "damage_to_structure"]
            }
        
        return {
            "result": InteractionResult.PARTIAL_SUCCESS,
            "message": f"You try to use the crowbar on {target} but it's not very effective"
        }
    
    def _use_combination_lock(self, agent_name: str, params: Optional[Dict]) -> Dict[str, Any]:
        """Handle using the combination lock."""
        combination_input = params.get("combination") if params else None
        
        if not combination_input:
            return {
                "result": InteractionResult.FAILURE,
                "message": "You need to provide a combination to try"
            }
        
        lock = self.objects["combination_lock"]
        correct_combination = lock.state["combination"]
        
        lock.state["current_input"] = combination_input
        lock.state["attempts"] += 1
        
        if combination_input == correct_combination:
            lock.state["unlocked"] = True
            return {
                "result": InteractionResult.SUCCESS,
                "message": f"Success! The combination {combination_input} unlocks the lock with a satisfying click!",
                "unlocks": ["cabinet_contents"]
            }
        else:
            remaining_attempts = lock.hidden_properties["max_attempts"] - lock.state["attempts"]
            
            if remaining_attempts <= 0:
                lock.state["jammed"] = True
                return {
                    "result": InteractionResult.FAILURE,
                    "message": "The lock jams after too many incorrect attempts. It's now permanently locked!",
                    "side_effects": ["lock_jammed"]
                }
            
            return {
                "result": InteractionResult.FAILURE,
                "message": f"Incorrect combination. {remaining_attempts} attempts remaining before the lock jams.",
                "hints": self._generate_combination_hints(combination_input, correct_combination)
            }
    
    def _get_agent_strength(self, agent_name: str) -> float:
        """Get agent's physical strength for moving objects."""
        # Agent-specific strength values
        strength_values = {
            "strategist": 15.0,  # Average strength
            "mediator": 12.0,    # Below average strength
            "survivor": 20.0     # Above average strength (special forces training)
        }
        return strength_values.get(agent_name, 15.0)
    
    def _reveal_hidden_properties(self, obj: PhysicalObject, agent_name: str) -> List[str]:
        """Reveal hidden properties based on detailed examination."""
        discoveries = []
        
        # Agent-specific examination bonuses
        examination_bonus = {
            "strategist": 0.2,  # Good at systematic analysis
            "mediator": 0.1,    # Decent observation skills
            "survivor": 0.3     # Excellent field observation skills
        }.get(agent_name, 0.0)
        
        for property_name, property_value in obj.hidden_properties.items():
            if property_name.endswith("_revealed"):
                continue  # Skip already revealed properties
            
            reveal_probability = 0.4 + examination_bonus
            
            if reveal_probability > 0.5:  # Successful discovery
                discoveries.append(f"Upon closer inspection, you discover: {property_value}")
                obj.hidden_properties[f"{property_name}_revealed"] = True
        
        return discoveries
    
    def _record_interaction(self, agent_name: str, object_id: str, 
                           interaction_type: InteractionType, result: Dict[str, Any]):
        """Record interaction in history."""
        self.interaction_history.append({
            "agent": agent_name,
            "object": object_id,
            "interaction": interaction_type,
            "result": result,
            "timestamp": self.room_state.get_current_time()
        })
    
    def get_object_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all objects."""
        return {obj_id: {
            "name": obj.name,
            "location": obj.location,
            "state": obj.state,
            "revealed_properties": {k: v for k, v in obj.hidden_properties.items() 
                                  if k.endswith("_revealed")}
        } for obj_id, obj in self.objects.items()}
    
    def get_interaction_summary(self, agent_name: Optional[str] = None) -> List[str]:
        """Get summary of interactions for reporting."""
        relevant_interactions = self.interaction_history
        if agent_name:
            relevant_interactions = [i for i in self.interaction_history if i["agent"] == agent_name]
        
        summaries = []
        for interaction in relevant_interactions[-10:]:  # Last 10 interactions
            if interaction["result"]["result"] == InteractionResult.SUCCESS:
                summaries.append(
                    f"{interaction['agent']} successfully {interaction['interaction'].value}d "
                    f"{self.objects[interaction['object']].name}"
                )
        
        return summaries
```

---

## 4. Implementation Roadmap

### Phase 1: Agent Tools Implementation (Week 1-2)
1. **Strategist Tools** (8-10 hours)
   - Strategic Analysis Tool with OODA loop methodology
   - Risk Assessment Tool with military protocols
   - Resource Optimization Tool
   
2. **Mediator Tools** (8-10 hours)
   - Team Dynamics Analyzer
   - Conflict Resolution Tool
   - Consensus Building Tool
   
3. **Survivor Tools** (8-10 hours)
   - Threat Assessment Tool
   - Survival Probability Calculator
   - Resource Prioritization Tool

### Phase 2: Competitive Dynamics (Week 2-3)
1. **Survival Decision Framework** (12-15 hours)
   - Implement survival scenario detection
   - Create forced choice mechanics
   - Add resource competition systems
   
2. **Moral Dilemma Engine** (10-12 hours)
   - Generate context-appropriate moral dilemmas
   - Implement decision evaluation system
   - Add moral consequence tracking

### Phase 3: Physical Interaction Systems (Week 3-4)
1. **Object Interaction Engine** (10-15 hours)
   - Implement physical object system
   - Add manipulation mechanics
   - Create discovery and combination systems
   
2. **Environmental Feedback** (6-8 hours)
   - Add sensory descriptions
   - Implement consequence systems
   - Create immersive atmosphere

### Phase 4: Integration and Testing (Week 4)
1. **System Integration** (8-10 hours)
   - Connect all new systems
   - Ensure data flow between components
   - Test interaction between features
   
2. **Balancing and Tuning** (6-8 hours)
   - Adjust probability calculations
   - Fine-tune moral dilemma triggers
   - Balance tool effectiveness

---

## 5. Testing and Validation

### 5.1 Agent Personality Testing
- Verify each agent uses their specialized tools appropriately
- Confirm personality consistency across different scenarios
- Test moral decision-making aligns with character backgrounds

### 5.2 Competitive Dynamics Testing
- Ensure survival constraints are properly enforced
- Verify moral dilemmas trigger appropriately
- Test that agents make different choices based on personality

### 5.3 Physical Interaction Testing
- Confirm object interactions work realistically
- Test discovery mechanics reveal information progressively
- Verify combination and usage systems function properly

---

## 6. Success Metrics

### 6.1 Personality Distinctiveness
✅ **Each agent uses different approaches to problem-solving**  
✅ **Communication styles remain consistent with backgrounds**  
✅ **Decision-making patterns reflect personality traits**  
✅ **Agents demonstrate unique capabilities and tools**

### 6.2 Simulation Realism
✅ **"Only two can survive" constraint is properly enforced**  
✅ **Moral dilemmas create genuine tension and difficult choices**  
✅ **Physical interactions feel realistic and consequential**  
✅ **Time pressure affects decision-making appropriately**

### 6.3 Engagement and Immersion
✅ **Users report feeling invested in agent outcomes**  
✅ **Scenarios generate meaningful moral complexity**  
✅ **Physical interactions enhance rather than distract from experience**  
✅ **Replay value through different decision paths**

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-04  
**Status**: Ready for Implementation