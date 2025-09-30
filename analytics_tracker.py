"""
Analytics tracker for conversation intelligence.
Keeps track of how well our sales conversations are going.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict


@dataclass
class ConversationMetrics:
    """What we track during conversations"""

    total_conversation_length: int = 0
    objection_count: int = 0
    positive_sentiment_indicators: int = 0
    negative_sentiment_indicators: int = 0
    decision_maker_confirmation: bool = False
    pain_point_identification_success: bool = False

    response_time_avg: float = 0.0
    engagement_score: float = 0.0
    clarification_requests: int = 0
    value_mentions: int = 0
    competitor_mentions: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SuccessIndicators:
    """Things that tell us we're winning"""

    meeting_scheduled: bool = False
    referral_obtained: bool = False
    content_delivery_requested: bool = False
    callback_scheduled: bool = False
    positive_engagement_maintained: bool = True

    pain_points_identified: List[str] = field(default_factory=list)
    objections_resolved: List[str] = field(default_factory=list)
    next_steps_agreed: bool = False
    budget_discussed: bool = False
    timeline_established: bool = False

    def calculate_success_score(self) -> float:
        """How well are we doing overall?"""
        weights = {
            "meeting_scheduled": 1.0,
            "referral_obtained": 0.8,
            "content_delivery_requested": 0.6,
            "callback_scheduled": 0.7,
            "positive_engagement_maintained": 0.3,
            "next_steps_agreed": 0.5,
            "budget_discussed": 0.4,
            "timeline_established": 0.4,
        }

        score = 0.0
        for field_name, weight in weights.items():
            if getattr(self, field_name, False):
                score += weight

        return min(1.0, score / sum(weights.values()))

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["success_score"] = self.calculate_success_score()
        return data


@dataclass
class OptimizationPoints:
    """Where we can get better"""

    objection_handling_effectiveness: float = 0.0
    value_proposition_resonance: float = 0.0
    timing_sensitivity_management: float = 0.0
    competitive_positioning_success: float = 0.0

    successful_pivots: List[Dict] = field(default_factory=list)
    missed_opportunities: List[Dict] = field(default_factory=list)
    effective_responses: List[Dict] = field(default_factory=list)
    conversation_momentum: List[float] = field(default_factory=list)

    def add_pivot(self, from_topic: str, to_topic: str, success: bool):
        """When we change topics mid-conversation"""
        self.successful_pivots.append(
            {
                "from": from_topic,
                "to": to_topic,
                "success": success,
                "timestamp": time.time(),
            }
        )

    def add_missed_opportunity(self, opportunity_type: str, context: str):
        """Track opportunities we missed"""
        self.missed_opportunities.append(
            {
                "type": opportunity_type,
                "context": context,
                "timestamp": time.time(),
            }
        )

    def calculate_optimization_score(self) -> float:
        """How good are we at handling different situations?"""
        scores = [
            self.objection_handling_effectiveness,
            self.value_proposition_resonance,
            self.timing_sensitivity_management,
            self.competitive_positioning_success,
        ]

        # Bonus points if we're getting better over time
        if len(self.conversation_momentum) > 2:
            recent_momentum = self.conversation_momentum[-3:]
            if all(
                recent_momentum[i] <= recent_momentum[i + 1]
                for i in range(len(recent_momentum) - 1)
            ):
                scores.append(0.8)

        return sum(scores) / len(scores) if scores else 0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["optimization_score"] = self.calculate_optimization_score()
        return data


class SentimentAnalyzer:
    """Analyze prospect sentiment and engagement"""

    POSITIVE_INDICATORS = [
        "yes",
        "sure",
        "absolutely",
        "definitely",
        "great",
        "excellent",
        "perfect",
        "sounds good",
        "interested",
        "tell me more",
        "go on",
        "makes sense",
        "I see",
        "right",
        "exactly",
        "correct",
        "agree",
        "love",
        "like",
        "helpful",
        "useful",
        "valuable",
        "important",
    ]

    NEGATIVE_INDICATORS = [
        "no",
        "not",
        "don't",
        "won't",
        "can't",
        "never",
        "hate",
        "dislike",
        "wrong",
        "bad",
        "terrible",
        "awful",
        "confused",
        "unclear",
        "complicated",
        "difficult",
        "expensive",
        "waste",
        "busy",
        "not interested",
        "go away",
        "stop",
        "enough",
    ]

    OBJECTION_INDICATORS = [
        "but",
        "however",
        "although",
        "concern",
        "worry",
        "problem",
        "issue",
        "challenge",
        "difficult",
        "expensive",
        "cost",
        "price",
        "competitor",
        "already have",
        "existing solution",
        "not sure",
    ]

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """Analyze the sentiment and intent behind user input"""

        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(
            1 for indicator in cls.POSITIVE_INDICATORS if indicator in text_lower
        )
        negative_count = sum(
            1 for indicator in cls.NEGATIVE_INDICATORS if indicator in text_lower
        )
        objection_count = sum(
            1 for indicator in cls.OBJECTION_INDICATORS if indicator in text_lower
        )

        total_indicators = positive_count + negative_count
        if total_indicators > 0:
            sentiment_score = (positive_count - negative_count) / total_indicators
        else:
            sentiment_score = 0.0

        if sentiment_score > 0.3:
            primary_sentiment = "positive"
        elif sentiment_score < -0.3:
            primary_sentiment = "negative"
        else:
            primary_sentiment = "neutral"

        is_question = "?" in text
        engagement_level = (
            "high" if len(words) > 20 else "medium" if len(words) > 5 else "low"
        )

        return {
            "sentiment_score": sentiment_score,
            "primary_sentiment": primary_sentiment,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "has_objection": objection_count > 0,
            "objection_strength": min(1.0, objection_count / 3),
            "is_question": is_question,
            "engagement_level": engagement_level,
            "word_count": len(words),
        }


class AnalyticsTracker:
    """Main analytics system that monitors conversation performance"""

    def __init__(self, config_manager=None, debug_mode: bool = False):
        self.config_manager = config_manager
        self.debug_mode = debug_mode

        self.metrics = ConversationMetrics()
        self.success_indicators = SuccessIndicators()
        self.optimization_points = OptimizationPoints()

        self.session_start = time.time()
        self.turn_timings = []
        self.sentiment_history = []
        self.intent_history = []
        self.routing_history = []

        self.patterns = defaultdict(int)
        self.topic_transitions = []

    def track_turn(
        self, user_input: str, bot_response: str, classification_result: Dict = None
    ):
        """Process what just happened in the conversation"""

        turn_start = time.time()
        self.metrics.total_conversation_length += 1

        sentiment = SentimentAnalyzer.analyze(user_input)
        self.sentiment_history.append(sentiment)

        if sentiment["primary_sentiment"] == "positive":
            self.metrics.positive_sentiment_indicators += sentiment[
                "positive_indicators"
            ]
        else:
            self.metrics.negative_sentiment_indicators += sentiment[
                "negative_indicators"
            ]

        if sentiment["has_objection"]:
            self.metrics.objection_count += 1

        self._update_engagement_score(sentiment)

        if classification_result:
            self.track_classification(classification_result)

        self._track_patterns(user_input)

        # Count when they mention value or competitors
        if any(
            word in user_input.lower()
            for word in ["value", "worth", "benefit", "roi", "return"]
        ):
            self.metrics.value_mentions += 1

        if any(
            word in user_input.lower()
            for word in [
                "competitor",
                "alternative",
                "other solution",
                "currently using",
            ]
        ):
            self.metrics.competitor_mentions += 1

        self.turn_timings.append(time.time() - turn_start)
        self.metrics.response_time_avg = sum(self.turn_timings) / len(self.turn_timings)

        if self.debug_mode:
            print(
                f"Turn {self.metrics.total_conversation_length}: {sentiment['primary_sentiment']} ({sentiment['sentiment_score']:.2f})"
            )
            print(f"   Engagement: {sentiment['engagement_level']}")

    def track_classification(self, classification_result: Dict):
        """What did we think they meant?"""

        if "primary_intent" in classification_result:
            self.intent_history.append(
                {
                    "intent": classification_result["primary_intent"],
                    "confidence": classification_result.get("confidence", 0),
                    "timestamp": time.time(),
                }
            )

        if classification_result.get("requires_clarification"):
            self.metrics.clarification_requests += 1

    def track_routing(self, from_node: str, to_node: str, reason: str):
        """Where did the conversation go?"""

        self.routing_history.append(
            {
                "from": from_node,
                "to": to_node,
                "reason": reason,
                "timestamp": time.time(),
            }
        )

        self.topic_transitions.append((from_node, to_node))

        if "objection" in reason.lower():
            self.optimization_points.objection_handling_effectiveness += 0.1

        if self.debug_mode:
            print(f"Routing: {from_node} → {to_node} ({reason})")

    def track_success_event(self, event_type: str, details: Dict = None):
        """Something good happened!"""

        success_field_mapping = {
            "meeting_scheduled": "meeting_scheduled",
            "referral_obtained": "referral_obtained",
            "content_requested": "content_delivery_requested",
            "callback_scheduled": "callback_scheduled",
            "next_steps": "next_steps_agreed",
            "budget_discussed": "budget_discussed",
            "timeline_established": "timeline_established",
        }

        if event_type in success_field_mapping:
            setattr(self.success_indicators, success_field_mapping[event_type], True)

            if self.debug_mode:
                print(f"Success: {event_type}")

        if event_type == "pain_point_identified" and details:
            self.success_indicators.pain_points_identified.append(
                details.get("pain_point", "")
            )
            self.metrics.pain_point_identification_success = True

        if event_type == "objection_resolved" and details:
            self.success_indicators.objections_resolved.append(
                details.get("objection", "")
            )

    def track_optimization_point(self, optimization_type: str, value: float):
        """How are we doing at specific things?"""

        optimization_mapping = {
            "objection_handling": "objection_handling_effectiveness",
            "value_proposition": "value_proposition_resonance",
            "timing": "timing_sensitivity_management",
            "competitive": "competitive_positioning_success",
        }

        if optimization_type in optimization_mapping:
            field = optimization_mapping[optimization_type]
            current = getattr(self.optimization_points, field)
            setattr(self.optimization_points, field, (current + value) / 2)

        current_score = self.get_current_performance_score()
        self.optimization_points.conversation_momentum.append(current_score)

    def _update_engagement_score(self, sentiment: Dict):
        """Update engagement score based on conversation signals"""

        factors = []
        factors.append((sentiment["sentiment_score"] + 1) / 2)

        length_score = min(1.0, sentiment["word_count"] / 30)
        factors.append(length_score)

        if sentiment["is_question"]:
            factors.append(0.8)

        # Consistent sentiment is usually good
        if len(self.sentiment_history) > 2:
            recent = self.sentiment_history[-3:]
            if all(
                s["primary_sentiment"] == recent[0]["primary_sentiment"] for s in recent
            ):
                factors.append(0.7)

        if factors:
            new_engagement = sum(factors) / len(factors)
            self.metrics.engagement_score = (
                self.metrics.engagement_score + new_engagement
            ) / 2

    def _track_patterns(self, user_input: str):
        """What kind of response was that?"""

        patterns = {
            "question": "?" in user_input,
            "short_response": len(user_input.split()) < 5,
            "long_response": len(user_input.split()) > 20,
            "agreement": any(
                word in user_input.lower()
                for word in ["yes", "agree", "right", "correct"]
            ),
            "disagreement": any(
                word in user_input.lower() for word in ["no", "disagree", "wrong"]
            ),
            "confusion": any(
                word in user_input.lower()
                for word in ["confused", "don't understand", "what do you mean"]
            ),
        }

        for pattern, detected in patterns.items():
            if detected:
                self.patterns[pattern] += 1

    def get_current_performance_score(self) -> float:
        """How are we doing overall?"""

        scores = []
        scores.append(self.metrics.engagement_score)

        if self.sentiment_history:
            recent_sentiment = sum(
                s["sentiment_score"] for s in self.sentiment_history[-5:]
            )
            scores.append(
                (recent_sentiment / min(5, len(self.sentiment_history)) + 1) / 2
            )

        scores.append(self.success_indicators.calculate_success_score())
        scores.append(self.optimization_points.calculate_optimization_score())

        # Fewer objections = better performance
        if self.metrics.total_conversation_length > 0:
            objection_rate = (
                self.metrics.objection_count / self.metrics.total_conversation_length
            )
            scores.append(1.0 - min(1.0, objection_rate * 2))

        return sum(scores) / len(scores) if scores else 0

    def get_analytics_summary(self) -> Dict:
        """Everything we know about this conversation"""

        duration = time.time() - self.session_start

        return {
            "session": {
                "duration_seconds": duration,
                "total_turns": self.metrics.total_conversation_length,
                "performance_score": self.get_current_performance_score(),
            },
            "metrics": self.metrics.to_dict(),
            "success_indicators": self.success_indicators.to_dict(),
            "optimization_points": self.optimization_points.to_dict(),
            "patterns": dict(self.patterns),
            "sentiment_trend": self._get_sentiment_trend(),
            "recommendations": self._generate_recommendations(),
        }

    def _get_sentiment_trend(self) -> str:
        """Analyze whether sentiment is improving or declining"""

        if len(self.sentiment_history) < 2:
            return "insufficient_data"

        mid = len(self.sentiment_history) // 2
        first_half = (
            sum(s["sentiment_score"] for s in self.sentiment_history[:mid]) / mid
        )
        second_half = sum(
            s["sentiment_score"] for s in self.sentiment_history[mid:]
        ) / len(self.sentiment_history[mid:])

        if second_half > first_half + 0.2:
            return "improving"
        elif second_half < first_half - 0.2:
            return "declining"
        else:
            return "stable"

    def _generate_recommendations(self) -> List[str]:
        """What should we do differently?"""

        recommendations = []

        if self.metrics.engagement_score < 0.3:
            recommendations.append(
                "Low engagement - consider more interactive questions"
            )

        if self.metrics.objection_count > self.metrics.total_conversation_length * 0.3:
            recommendations.append("High objection rate - focus on value demonstration")

        if not self.success_indicators.pain_points_identified:
            recommendations.append(
                "No pain points identified - deeper discovery needed"
            )

        if self.patterns.get("confusion", 0) > 2:
            recommendations.append("Multiple confusion indicators - simplify messaging")

        if (
            self.patterns.get("short_response", 0)
            > self.metrics.total_conversation_length * 0.5
        ):
            recommendations.append(
                "Predominantly short responses - ask open-ended questions"
            )

        return recommendations


if __name__ == "__main__":
    tracker = AnalyticsTracker(debug_mode=True)

    turns = [
        ("I'm not sure if this is right for us", "I understand your concern..."),
        ("We already have a solution in place", "That's great that you have..."),
        ("What makes yours different?", "Excellent question! Our solution..."),
        ("That sounds interesting", "I'm glad you find it valuable..."),
        ("Can we schedule a demo?", "Absolutely! I'd be happy to..."),
    ]

    for user_input, bot_response in turns:
        tracker.track_turn(user_input, bot_response)
        time.sleep(0.1)

    tracker.track_success_event(
        "pain_point_identified", {"pain_point": "Manual process taking too long"}
    )
    tracker.track_success_event("meeting_scheduled")

    tracker.track_routing("DISCOVERY", "OBJECTION_HANDLING", "User expressed concern")
    tracker.track_routing(
        "OBJECTION_HANDLING", "VALUE_PROPOSITION", "Objection addressed"
    )

    summary = tracker.get_analytics_summary()
    print("\nAnalytics Summary:")
    print(json.dumps(summary, indent=2))
