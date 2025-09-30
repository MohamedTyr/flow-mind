"""
Routing optimizer that figures out where to go next in conversations
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class RoutingStrategy(Enum):
    DIRECT = "direct"  # Just go to the best match
    EXPLORATORY = "exploratory"  # Fish for more info
    RECOVERY = "recovery"  # Fix things when they go south
    ADVANCEMENT = "advancement"  # Push toward the goal
    MAINTENANCE = "maintenance"  # Keep things steady


@dataclass
class RoutingOption:
    target_node: str
    intent: str
    confidence: float
    benefit_score: float = 0.0
    risk_score: float = 0.0
    context_alignment: float = 0.0
    strategic_value: float = 0.0

    def calculate_total_score(self, weights: Dict[str, float] = None) -> float:
        # Default scoring weights - confidence and benefit matter most
        default_weights = {
            "confidence": 0.25,
            "benefit": 0.30,
            "context": 0.25,
            "strategic": 0.15,
            "risk": -0.05,  # Risk hurts the score
        }

        weights = weights or default_weights

        score = (
            self.confidence * weights.get("confidence", 0.25)
            + self.benefit_score * weights.get("benefit", 0.30)
            + self.context_alignment * weights.get("context", 0.25)
            + self.strategic_value * weights.get("strategic", 0.15)
            + self.risk_score * weights.get("risk", -0.05)
        )

        return max(0, min(1, score))


@dataclass
class RoutingDecision:
    should_route: bool
    target_node: Optional[str] = None
    strategy: RoutingStrategy = RoutingStrategy.DIRECT
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: List[RoutingOption] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "should_route": self.should_route,
            "target_node": self.target_node,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": [
                {"node": alt.target_node, "score": alt.calculate_total_score()}
                for alt in self.alternatives[:3]
            ],
        }


class RoutingOptimizer:
    # What kind of nodes we're dealing with
    NODE_TYPES = {
        "discovery": ["DISCOVERY", "QUALIFICATION", "INVESTIGATION"],
        "objection": ["OBJECTION", "CONCERN", "RESISTANCE", "SKEPTICAL"],
        "value": ["VALUE", "BENEFIT", "SOLUTION", "PROPOSITION"],
        "closing": ["CLOSING", "DEMO", "MEETING", "SCHEDULE", "COMMITMENT"],
        "recovery": ["RECOVERY", "APOLOGIZE", "CLARIFY", "REDIRECT"],
    }

    STAGE_GOALS = {
        "opening": "build_rapport",
        "discovery": "identify_pain_points",
        "value_proposition": "demonstrate_value",
        "objection_handling": "address_concerns",
        "closing": "secure_commitment",
    }

    def __init__(
        self,
        tree_manager=None,
        context_manager=None,
        analytics_tracker=None,
        debug_mode: bool = False,
    ):
        self.tree_manager = tree_manager
        self.context_manager = context_manager
        self.analytics_tracker = analytics_tracker
        self.debug_mode = debug_mode

        self.routing_history: List[Dict] = []
        self.node_visit_counts: Dict[str, int] = {}
        self.successful_paths: List[List[str]] = []

    def optimize_routing(
        self,
        current_node: str,
        classification_result: Dict,
        state: Any,
        user_input: str,
    ) -> RoutingDecision:
        """Main method - figures out where to go next"""

        context = (
            self.context_manager.get_context_summary() if self.context_manager else {}
        )

        options = self._get_routing_options(current_node, classification_result)

        if not options:
            return RoutingDecision(
                should_route=False, reasoning="No viable routing options"
            )

        # Score each option
        for option in options:
            self._calculate_option_benefits(option, current_node, context, state)

        strategy = self._select_routing_strategy(context, state)
        decision = self._make_routing_decision(options, strategy, context)
        self._track_routing(current_node, decision)

        if self.debug_mode:
            print(f"🚦 Routing Decision:")
            print(f"   Strategy: {strategy.value}")
            print(f"   Should Route: {decision.should_route}")
            if decision.should_route:
                print(f"   Target: {decision.target_node}")
                print(f"   Confidence: {decision.confidence:.2f}")

        return decision

    def _get_routing_options(
        self, current_node: str, classification_result: Dict
    ) -> List[RoutingOption]:
        """Pull out all the places we could potentially go"""

        options = []

        # Full matches are gold
        if classification_result.get("has_full_match"):
            for classification in classification_result.get("classifications", []):
                if classification.get("match_type") == "FULL":
                    options.append(
                        RoutingOption(
                            target_node=classification.get("target"),
                            intent=classification.get("intent"),
                            confidence=classification.get("confidence", 0),
                        )
                    )

        # Good partial matches are worth considering
        for classification in classification_result.get("classifications", []):
            if (
                classification.get("match_type") == "PARTIAL"
                and classification.get("confidence", 0) > 0.6
            ):
                options.append(
                    RoutingOption(
                        target_node=classification.get("target"),
                        intent=classification.get("intent"),
                        confidence=classification.get("confidence", 0)
                        * 0.8,  # Dock some points for being partial
                    )
                )

        # Global matches from other parts of the tree
        if "global_matches" in classification_result:
            for match in classification_result["global_matches"]:
                if match.get("confidence", 0) > 0.7:
                    options.append(
                        RoutingOption(
                            target_node=match.get("source_node"),
                            intent=match.get("intent"),
                            confidence=match.get("confidence", 0),
                        )
                    )

        # No duplicates
        seen = set()
        unique_options = []
        for option in options:
            key = (option.target_node, option.intent)
            if key not in seen:
                seen.add(key)
                unique_options.append(option)

        return unique_options

    def _calculate_option_benefits(
        self, option: RoutingOption, current_node: str, context: Dict, state: Any
    ):
        """Score how good each routing option looks"""

        progression_benefit = self._calculate_progression_benefit(
            current_node, option.target_node, context.get("decision_stage")
        )

        context_alignment = self._calculate_context_alignment(
            option.target_node, context
        )

        strategic_value = self._calculate_strategic_value(
            option.target_node, context.get("engagement_level"), state
        )

        risk_score = self._calculate_risk_score(
            current_node, option.target_node, context
        )

        option.benefit_score = progression_benefit
        option.context_alignment = context_alignment
        option.strategic_value = strategic_value
        option.risk_score = risk_score

    def _calculate_progression_benefit(
        self, current_node: str, target_node: str, decision_stage: Optional[str]
    ) -> float:
        """How much does this move help the conversation flow?"""

        benefit = 0.5

        current_type = self._get_node_type(current_node)
        target_type = self._get_node_type(target_node)

        # Some moves just make sense
        natural_progressions = {
            "discovery": ["value", "objection"],
            "objection": ["value", "recovery"],
            "value": ["closing", "objection"],
            "closing": ["closing"],  # Staying in closing is fine
        }

        if current_type in natural_progressions:
            if target_type in natural_progressions[current_type]:
                benefit += 0.3

        # Late stage? Time to close
        if decision_stage in ["evaluation", "decision"]:
            if target_type == "closing":
                benefit += 0.4

        # Don't go in circles
        recent_targets = [h.get("to") for h in self.routing_history[-3:]]
        if target_node in recent_targets:
            benefit -= 0.3

        # New places are interesting, but don't be a tourist
        visit_count = self.node_visit_counts.get(target_node, 0)
        if visit_count == 0:
            benefit += 0.1
        elif visit_count > 3:
            benefit -= 0.2

        return max(0, min(1, benefit))

    def _calculate_context_alignment(self, target_node: str, context: Dict) -> float:
        """Does this move make sense given what's happening?"""

        alignment = 0.5
        node_type = self._get_node_type(target_node)
        engagement = context.get("engagement_level", "neutral")

        # If they're pissed off, don't try to close
        if engagement in ["hostile", "resistant"]:
            if node_type in ["recovery", "objection"]:
                alignment += 0.3
            elif node_type == "closing":
                alignment -= 0.4

        # If they're interested, go for it
        elif engagement in ["interested", "engaged"]:
            if node_type in ["value", "closing"]:
                alignment += 0.3
            elif node_type == "recovery":
                alignment -= 0.2  # Don't be so careful

        # Got pain points? Talk value
        pain_points = context.get("pain_points", [])
        if pain_points and node_type == "value":
            alignment += 0.2

        # Match the decision stage
        stage = context.get("decision_stage", "unaware")
        stage_alignment = {
            "unaware": {"discovery": 0.3, "value": -0.2},
            "problem_aware": {"discovery": 0.2, "value": 0.2},
            "solution_aware": {"value": 0.3, "closing": 0.1},
            "evaluation": {"value": 0.2, "closing": 0.3},
            "decision": {"closing": 0.4, "objection": 0.2},
        }

        if stage in stage_alignment and node_type in stage_alignment[stage]:
            alignment += stage_alignment[stage][node_type]

        return max(0, min(1, alignment))

    def _calculate_strategic_value(
        self, target_node: str, engagement_level: Optional[str], state: Any
    ) -> float:
        """Is this a smart move strategically?"""

        value = 0.5

        # Been talking a while? Time to close
        if hasattr(state, "turn_count") and state.turn_count > 5:
            if any(word in target_node for word in ["CLOSING", "SCHEDULE", "MEETING"]):
                value += 0.3

        # Things going badly? Fix it
        if engagement_level in ["hostile", "resistant"]:
            if any(
                word in target_node for word in ["RECOVERY", "ACKNOWLEDGE", "EMPATHY"]
            ):
                value += 0.4

        # Early conversation? Learn stuff
        if hasattr(state, "turn_count") and state.turn_count < 3:
            if any(word in target_node for word in ["DISCOVERY", "QUALIFICATION"]):
                value += 0.3

        # Got some info already? Show value
        if hasattr(state, "collected_partials") and len(state.collected_partials) > 2:
            if any(word in target_node for word in ["VALUE", "BENEFIT"]):
                value += 0.2

        return max(0, min(1, value))

    def _calculate_risk_score(
        self, current_node: str, target_node: str, context: Dict
    ) -> float:
        """What could go wrong with this move?"""

        risk = 0.0

        # Don't be pushy when they're already annoyed
        if context.get("engagement_level") in ["hostile", "resistant"]:
            if any(word in target_node for word in ["CLOSING", "COMMITMENT"]):
                risk += 0.7

        # Don't go backwards when they're ready to decide
        if context.get("decision_stage") in ["evaluation", "decision"]:
            if "DISCOVERY" in target_node:
                risk += 0.3

        # Going in circles is bad
        recent_nodes = [h.get("to") for h in self.routing_history[-5:]]
        if target_node in recent_nodes:
            risk += 0.4

        # Some moves are just weird
        current_type = self._get_node_type(current_node)
        target_type = self._get_node_type(target_node)

        weird_moves = {
            "discovery": ["closing"],  # Way too early
            "objection": ["closing"],  # Fix the problem first
            "closing": ["discovery"],  # Why go backwards?
        }

        if current_type in weird_moves:
            if target_type in weird_moves[current_type]:
                risk += 0.5

        return max(0, min(1, risk))

    def _get_node_type(self, node_name: str) -> str:
        node_upper = node_name.upper()

        for node_type, keywords in self.NODE_TYPES.items():
            if any(keyword in node_upper for keyword in keywords):
                return node_type

        return "general"

    def _select_routing_strategy(self, context: Dict, state: Any) -> RoutingStrategy:
        """Pick the right approach for the situation"""

        engagement = context.get("engagement_level", "neutral")
        decision_stage = context.get("decision_stage", "unaware")

        # Things going south? Fix it
        if engagement in ["hostile", "resistant"]:
            return RoutingStrategy.RECOVERY

        # They're interested and ready? Go for it
        if engagement in ["interested", "engaged"] and decision_stage in [
            "evaluation",
            "decision",
        ]:
            return RoutingStrategy.ADVANCEMENT

        # Early conversation? Explore
        if hasattr(state, "turn_count") and state.turn_count < 3:
            return RoutingStrategy.EXPLORATORY

        # Things are good? Don't mess it up
        if engagement == "engaged" and context.get("engagement_trend") == "stable":
            return RoutingStrategy.MAINTENANCE

        return RoutingStrategy.DIRECT

    def _make_routing_decision(
        self, options: List[RoutingOption], strategy: RoutingStrategy, context: Dict
    ) -> RoutingDecision:
        """Final call - where do we actually go?"""

        # Different strategies care about different things
        strategy_weights = {
            RoutingStrategy.DIRECT: {
                "confidence": 0.40,
                "benefit": 0.30,
                "context": 0.20,
                "strategic": 0.10,
                "risk": 0.00,
            },
            RoutingStrategy.EXPLORATORY: {
                "confidence": 0.20,
                "benefit": 0.30,
                "context": 0.20,
                "strategic": 0.30,
                "risk": 0.00,
            },
            RoutingStrategy.RECOVERY: {
                "confidence": 0.15,
                "benefit": 0.20,
                "context": 0.40,
                "strategic": 0.35,
                "risk": -0.10,
            },
            RoutingStrategy.ADVANCEMENT: {
                "confidence": 0.25,
                "benefit": 0.35,
                "context": 0.15,
                "strategic": 0.30,
                "risk": -0.05,
            },
            RoutingStrategy.MAINTENANCE: {
                "confidence": 0.30,
                "benefit": 0.25,
                "context": 0.30,
                "strategic": 0.20,
                "risk": -0.05,
            },
        }

        weights = strategy_weights.get(
            strategy, strategy_weights[RoutingStrategy.DIRECT]
        )

        # Score all options
        for option in options:
            option.total_score = option.calculate_total_score(weights)

        sorted_options = sorted(options, key=lambda x: x.total_score, reverse=True)

        if not sorted_options:
            return RoutingDecision(
                should_route=False, strategy=strategy, reasoning="No options available"
            )

        best_option = sorted_options[0]
        threshold = self._get_routing_threshold(strategy, context)
        should_route = best_option.total_score > threshold

        reasoning = self._generate_routing_reasoning(
            best_option, should_route, strategy, context
        )

        return RoutingDecision(
            should_route=should_route,
            target_node=best_option.target_node if should_route else None,
            strategy=strategy,
            confidence=best_option.total_score,
            reasoning=reasoning,
            alternatives=sorted_options[1:4],
        )

    def _get_routing_threshold(self, strategy: RoutingStrategy, context: Dict) -> float:
        base_threshold = 0.5

        # Some strategies are more trigger-happy
        strategy_adjustments = {
            RoutingStrategy.EXPLORATORY: -0.1,  # More willing to jump around
            RoutingStrategy.RECOVERY: -0.15,  # Desperate to fix things
            RoutingStrategy.ADVANCEMENT: 0.0,
            RoutingStrategy.MAINTENANCE: 0.1,  # Don't rock the boat
            RoutingStrategy.DIRECT: 0.0,
        }

        threshold = base_threshold + strategy_adjustments.get(strategy, 0)

        # Bad vibes? Try something else
        if context.get("engagement_level") in ["hostile", "resistant"]:
            threshold -= 0.1
        elif context.get("engagement_level") == "engaged":
            threshold += 0.1  # Don't mess with success

        return max(0.2, min(0.8, threshold))

    def _generate_routing_reasoning(
        self,
        option: RoutingOption,
        should_route: bool,
        strategy: RoutingStrategy,
        context: Dict,
    ) -> str:
        """Explain why we're doing what we're doing"""

        if not should_route:
            return f"Staying put ({strategy.value}, score: {option.total_score:.2f})"

        reasons = []

        if option.confidence > 0.8:
            reasons.append("strong match")

        if option.benefit_score > 0.7:
            reasons.append("good flow")

        if option.context_alignment > 0.7:
            reasons.append("fits situation")

        if option.strategic_value > 0.7:
            reasons.append("smart move")

        if option.risk_score < 0.3:
            reasons.append("safe bet")

        engagement = context.get("engagement_level")
        if engagement:
            reasons.append(f"they're {engagement}")

        return (
            f"Going to {option.target_node} ({strategy.value}) - {', '.join(reasons)}"
        )

    def _track_routing(self, from_node: str, decision: RoutingDecision):
        """Keep track of where we've been"""

        self.routing_history.append(
            {
                "from": from_node,
                "to": decision.target_node,
                "strategy": decision.strategy.value,
                "confidence": decision.confidence,
                "timestamp": time.time(),
            }
        )

        if decision.target_node:
            self.node_visit_counts[decision.target_node] = (
                self.node_visit_counts.get(decision.target_node, 0) + 1
            )

        # Don't let history get crazy long
        if len(self.routing_history) > 50:
            self.routing_history = self.routing_history[-50:]

    def get_routing_analytics(self) -> Dict:
        """What patterns are we seeing?"""

        if not self.routing_history:
            return {}

        # Count strategy usage
        strategy_counts = {}
        for entry in self.routing_history:
            strategy = entry["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Common paths
        path_counts = {}
        for i in range(len(self.routing_history) - 1):
            path = (
                f"{self.routing_history[i]['from']} → {self.routing_history[i]['to']}"
            )
            path_counts[path] = path_counts.get(path, 0) + 1

        common_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_routings": len(self.routing_history),
            "strategy_distribution": strategy_counts,
            "most_visited_nodes": sorted(
                self.node_visit_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "common_paths": common_paths,
            "avg_confidence": sum(h["confidence"] for h in self.routing_history)
            / len(self.routing_history),
        }


if __name__ == "__main__":
    # Quick test
    optimizer = RoutingOptimizer(debug_mode=True)

    classification_result = {
        "has_full_match": True,
        "classifications": [
            {
                "intent": "OBJECTION_TIME",
                "match_type": "FULL",
                "confidence": 0.85,
                "target": "TIME_CONSTRAINT_HANDLING",
            },
            {
                "intent": "INTEREST_MODERATE",
                "match_type": "PARTIAL",
                "confidence": 0.65,
                "target": "VALUE_PROPOSITION",
            },
        ],
    }

    from context_manager import AdvancedContextManager

    context_mgr = AdvancedContextManager()
    context_mgr.engagement_level = context_mgr.EngagementLevel.INTERESTED

    class MockState:
        def __init__(self):
            self.turn_count = 4
            self.collected_partials = []

    state = MockState()

    decision = optimizer.optimize_routing(
        current_node="DISCOVERY",
        classification_result=classification_result,
        state=state,
        user_input="I don't have time for this right now",
    )

    print("\n🚦 Routing Decision:")
    import json

    print(json.dumps(decision.to_dict(), indent=2))

    # Try a few more
    for _ in range(3):
        decision = optimizer.optimize_routing(
            current_node="VALUE_PROPOSITION",
            classification_result=classification_result,
            state=state,
            user_input="Test input",
        )

    print("\n📊 Routing Analytics:")
    print(json.dumps(optimizer.get_routing_analytics(), indent=2))
