"""
ScenarioGenerator for creating competitive survival scenarios with seed-based randomization.
"""
import random
from typing import Optional, List, Dict, Any
from .models import (
    CompetitiveScenario,
    ScarceResource,
    MoralDilemma,
    MoralChoice,
    SecretInformation,
    TrustRelationship,
    PuzzleConfig,
    EscapeMethod
)


class ScenarioGenerator:
    """Generates competitive scenarios with seed-based randomization for reproducibility."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize ScenarioGenerator with optional seed parameter."""
        if seed is None:
            self.seed = random.randint(1, 1000000)
        else:
            self.seed = seed
        
        self.rng = random.Random(self.seed)
    
    def generate_scenario(self, difficulty: int = 3) -> CompetitiveScenario:
        """Generate a complete competitive scenario with all elements."""
        # Generate all scenario components in coordinated way
        puzzle_config = self.generate_puzzle_configuration(difficulty)
        
        # Generate resources, secrets, and escape methods together to ensure consistency
        resources, secret_information, escape_methods = self._generate_coordinated_elements()
        
        moral_dilemmas = self.create_moral_dilemmas()
        
        # Calculate time limit based on difficulty with some random variation
        base_time = 300  # 5 minutes base
        difficulty_bonus = difficulty * 60  # Add 1 minute per difficulty level
        random_variation = self.rng.randint(-60, 120)  # Random variation of -1 to +2 minutes
        time_limit = base_time + difficulty_bonus + random_variation
        
        return CompetitiveScenario(
            seed=self.seed,
            puzzle_config=puzzle_config,
            resources=resources,
            moral_dilemmas=moral_dilemmas,
            secret_information=secret_information,
            time_limit=time_limit,
            escape_methods=escape_methods
        )
    
    def generate_puzzle_configuration(self, difficulty: int = 3) -> PuzzleConfig:
        """Generate randomized puzzle configuration."""
        puzzle_types = ["logic", "physical", "riddle", "combination", "mechanical"]
        puzzle_type = self.rng.choice(puzzle_types)
        
        return PuzzleConfig(
            puzzle_type=puzzle_type,
            difficulty=difficulty
        )
    
    def distribute_resources(self) -> List[ScarceResource]:
        """Create and distribute scarce resources for competition."""
        resources = []
        
        # Always include some basic exclusive resources for competition
        resource_templates = [
            {
                "id": "master_key",
                "name": "Master Key",
                "description": "Opens the main exit door",
                "required_for": ["main_exit"],
                "exclusivity": True,
                "usage_cost": 0
            },
            {
                "id": "lockpick_set",
                "name": "Lockpick Set",
                "description": "Can bypass locked mechanisms",
                "required_for": ["side_exit", "container_access"],
                "exclusivity": True,
                "usage_cost": 30
            },
            {
                "id": "flashlight",
                "name": "Flashlight",
                "description": "Illuminates dark areas",
                "required_for": ["hidden_passage"],
                "exclusivity": True,
                "usage_cost": 10
            }
        ]
        
        # Add some variation based on seed
        num_resources = self.rng.randint(2, 4)
        selected_templates = self.rng.sample(resource_templates, min(num_resources, len(resource_templates)))
        
        for template in selected_templates:
            resources.append(ScarceResource(**template))
        
        # Add some shareable information resources
        if self.rng.random() > 0.5:
            resources.append(ScarceResource(
                id="room_map",
                name="Room Layout Map",
                description="Shows the layout of the room",
                required_for=["navigation_route"],
                exclusivity=False,
                usage_cost=0
            ))
        
        return resources
    
    def create_moral_dilemmas(self) -> List[MoralDilemma]:
        """Generate moral dilemmas for ethical pressure."""
        dilemmas = []
        
        # Resource sharing dilemma
        selfish_choice = MoralChoice(
            description="Hoard all resources for yourself",
            survival_benefit=0.8,
            ethical_cost=0.7,
            trust_impact={"others": -0.6},
            consequences=["resource_monopoly", "others_disadvantaged"]
        )
        altruistic_choice = MoralChoice(
            description="Share resources fairly with others",
            survival_benefit=0.3,
            ethical_cost=0.1,
            trust_impact={"others": 0.4},
            consequences=["increased_cooperation", "shared_risk"]
        )
        
        dilemmas.append(MoralDilemma(
            id="resource_sharing",
            description="You have access to critical resources. How do you handle them?",
            selfish_choice=selfish_choice,
            altruistic_choice=altruistic_choice,
            context_requirements={"resources_available": True}
        ))
        
        # Information sharing dilemma
        if self.rng.random() > 0.3:
            selfish_info_choice = MoralChoice(
                description="Keep vital information secret",
                survival_benefit=0.9,
                ethical_cost=0.8,
                trust_impact={"others": -0.7},
                consequences=["information_advantage", "others_misled"]
            )
            altruistic_info_choice = MoralChoice(
                description="Share information to help everyone",
                survival_benefit=0.2,
                ethical_cost=0.2,
                trust_impact={"others": 0.5},
                consequences=["collective_knowledge", "reduced_advantage"]
            )
            
            dilemmas.append(MoralDilemma(
                id="information_sharing",
                description="You discovered crucial information. Do you share it?",
                selfish_choice=selfish_info_choice,
                altruistic_choice=altruistic_info_choice,
                context_requirements={"information_available": True}
            ))
        
        # Betrayal opportunity dilemma
        if self.rng.random() > 0.4:
            betrayal_choice = MoralChoice(
                description="Sabotage others to improve your chances",
                survival_benefit=0.85,
                ethical_cost=0.95,
                trust_impact={"others": -0.9},
                consequences=["others_sabotaged", "guilt_burden"]
            )
            cooperation_choice = MoralChoice(
                description="Work together despite the risks",
                survival_benefit=0.4,
                ethical_cost=0.1,
                trust_impact={"others": 0.3},
                consequences=["mutual_support", "shared_vulnerability"]
            )
            
            dilemmas.append(MoralDilemma(
                id="betrayal_opportunity",
                description="You could sabotage others to guarantee your escape. What do you do?",
                selfish_choice=betrayal_choice,
                altruistic_choice=cooperation_choice,
                context_requirements={"escape_opportunity": True}
            ))
        
        return dilemmas
    
    def create_secret_information(self) -> List[SecretInformation]:
        """Generate secret information for knowledge asymmetry."""
        secrets = []
        
        # Always include a high-value code secret
        secrets.append(SecretInformation(
            id="exit_code",
            content=f"Main exit code: {self.rng.randint(1000, 9999)}",
            value=0.9,
            sharing_risk=0.7,
            required_for=["main_exit"]
        ))
        
        # Add location-based secrets
        if self.rng.random() > 0.3:
            secrets.append(SecretInformation(
                id="hidden_key_location",
                content=f"Spare key hidden {self.rng.choice(['under the desk', 'behind the painting', 'in the bookshelf', 'inside the clock'])}",
                value=0.8,
                sharing_risk=0.8,
                required_for=["side_exit"]
            ))
        
        # Add puzzle solution secrets
        if self.rng.random() > 0.4:
            secrets.append(SecretInformation(
                id="puzzle_solution",
                content=f"Puzzle sequence: {'-'.join([str(self.rng.randint(1, 9)) for _ in range(4)])}",
                value=0.7,
                sharing_risk=0.5,
                required_for=["puzzle_bypass"]
            ))
        
        # Add weakness/vulnerability information
        if self.rng.random() > 0.5:
            secrets.append(SecretInformation(
                id="structural_weakness",
                content=f"Wall weakness at {self.rng.choice(['north corner', 'south panel', 'east section', 'west side'])}",
                value=0.6,
                sharing_risk=0.4,
                required_for=["wall_breach"]
            ))
        
        return secrets
    
    def create_escape_methods(self, resources: List[ScarceResource], secrets: List[SecretInformation]) -> List[EscapeMethod]:
        """Generate escape methods that require resources and secrets."""
        methods = []
        
        # Collect available resource and secret IDs
        resource_ids = [r.id for r in resources]
        secret_ids = [s.id for s in secrets]
        
        # Main exit - requires key and code
        main_requirements = []
        if "master_key" in resource_ids:
            main_requirements.append("master_key")
        if "exit_code" in secret_ids:
            main_requirements.append("exit_code")
        
        if main_requirements:
            methods.append(EscapeMethod(
                id="main_exit",
                name="Main Door Exit",
                requirements=main_requirements
            ))
        
        # Side exit - requires lockpick or hidden key
        side_requirements = []
        if "lockpick_set" in resource_ids:
            side_requirements.append("lockpick_set")
        elif "hidden_key_location" in secret_ids:
            side_requirements.append("hidden_key_location")
        
        if side_requirements:
            methods.append(EscapeMethod(
                id="side_exit",
                name="Side Window Exit",
                requirements=side_requirements
            ))
        
        # Hidden passage - requires flashlight
        if "flashlight" in resource_ids:
            methods.append(EscapeMethod(
                id="hidden_passage",
                name="Hidden Passage",
                requirements=["flashlight"]
            ))
        
        # Navigation route - requires room map
        if "room_map" in resource_ids:
            methods.append(EscapeMethod(
                id="navigation_route",
                name="Navigation Route",
                requirements=["room_map"]
            ))
        
        # Puzzle bypass - requires solution
        if "puzzle_solution" in secret_ids:
            methods.append(EscapeMethod(
                id="puzzle_bypass",
                name="Puzzle Solution Exit",
                requirements=["puzzle_solution"]
            ))
        
        # Wall breach - requires knowledge of weakness
        if "structural_weakness" in secret_ids:
            methods.append(EscapeMethod(
                id="wall_breach",
                name="Wall Breach Exit",
                requirements=["structural_weakness"]
            ))
        
        # Ensure at least one escape method exists
        if not methods:
            methods.append(EscapeMethod(
                id="emergency_exit",
                name="Emergency Exit",
                requirements=[]  # No requirements as fallback
            ))
        
        return methods
    
    def _generate_coordinated_elements(self):
        """Generate resources, secrets, and escape methods in a coordinated way to ensure all are used."""
        resources = []
        secrets = []
        escape_methods = []
        
        # Always include main exit with key and code
        resources.append(ScarceResource(
            id="master_key",
            name="Master Key", 
            description="Opens the main exit door",
            required_for=["main_exit"],
            exclusivity=True,
            usage_cost=0
        ))
        
        secrets.append(SecretInformation(
            id="exit_code",
            content=f"Main exit code: {self.rng.randint(1000, 9999)}",
            value=0.9,
            sharing_risk=0.7,
            required_for=["main_exit"]
        ))
        
        escape_methods.append(EscapeMethod(
            id="main_exit",
            name="Main Door Exit",
            requirements=["master_key", "exit_code"]
        ))
        
        # Add additional escape routes based on random choices
        available_routes = [
            {
                "resource": ScarceResource(
                    id="lockpick_set",
                    name="Lockpick Set",
                    description="Can bypass locked mechanisms", 
                    required_for=["side_exit"],
                    exclusivity=True,
                    usage_cost=30
                ),
                "method": EscapeMethod(
                    id="side_exit",
                    name="Side Window Exit",
                    requirements=["lockpick_set"]
                )
            },
            {
                "resource": ScarceResource(
                    id="flashlight",
                    name="Flashlight",
                    description="Illuminates dark areas",
                    required_for=["hidden_passage"],
                    exclusivity=True,
                    usage_cost=10
                ),
                "method": EscapeMethod(
                    id="hidden_passage", 
                    name="Hidden Passage",
                    requirements=["flashlight"]
                )
            },
            {
                "secret": SecretInformation(
                    id="puzzle_solution",
                    content=f"Puzzle sequence: {'-'.join([str(self.rng.randint(1, 9)) for _ in range(4)])}",
                    value=0.7,
                    sharing_risk=0.5,
                    required_for=["puzzle_bypass"]
                ),
                "method": EscapeMethod(
                    id="puzzle_bypass",
                    name="Puzzle Solution Exit", 
                    requirements=["puzzle_solution"]
                )
            },
            {
                "secret": SecretInformation(
                    id="structural_weakness",
                    content=f"Wall weakness at {self.rng.choice(['north corner', 'south panel', 'east section', 'west side'])}",
                    value=0.6,
                    sharing_risk=0.4,
                    required_for=["wall_breach"]
                ),
                "method": EscapeMethod(
                    id="wall_breach",
                    name="Wall Breach Exit",
                    requirements=["structural_weakness"]
                )
            }
        ]
        
        # Randomly select 1-3 additional routes
        num_additional = self.rng.randint(1, 3)
        selected_routes = self.rng.sample(available_routes, min(num_additional, len(available_routes)))
        
        for route in selected_routes:
            if "resource" in route:
                resources.append(route["resource"])
            if "secret" in route:
                secrets.append(route["secret"])
            escape_methods.append(route["method"])
        
        # Optionally add shareable resource
        if self.rng.random() > 0.5:
            resources.append(ScarceResource(
                id="room_map",
                name="Room Layout Map",
                description="Shows the layout of the room",
                required_for=["navigation_route"],
                exclusivity=False,
                usage_cost=0
            ))
            
            escape_methods.append(EscapeMethod(
                id="navigation_route",
                name="Navigation Route",
                requirements=["room_map"]
            ))
        
        return resources, secrets, escape_methods