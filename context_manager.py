"""
Context manager that tracks what customers care about during conversations
Figures out their pain points, priorities, and where they are in the buying process
"""

import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict


class DecisionStage(Enum):
    """Where the customer is in their buying journey"""

    UNAWARE = "unaware"
    PROBLEM_AWARE = "problem_aware"
    SOLUTION_AWARE = "solution_aware"
    PRODUCT_AWARE = "product_aware"
    EVALUATION = "evaluation"
    DECISION = "decision"
    PURCHASE = "purchase"
    POST_PURCHASE = "post_purchase"


class EngagementLevel(Enum):
    """How engaged/interested the customer seems"""

    HOSTILE = "hostile"
    RESISTANT = "resistant"
    NEUTRAL = "neutral"
    INTERESTED = "interested"
    ENGAGED = "engaged"
    CHAMPION = "champion"


@dataclass
class PainPoint:
    """A problem the customer mentioned"""

    description: str
    severity: float  # 0-1 scale
    category: str  # operational, financial, strategic, etc.
    mentioned_count: int = 1
    first_mentioned: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    related_topics: List[str] = field(default_factory=list)

    def update_mention(self):
        """Customer brought this up again - must be important"""
        self.mentioned_count += 1
        self.last_mentioned = time.time()
        self.severity = min(1.0, self.severity + 0.1)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["recency_score"] = self._calculate_recency()
        return data

    def _calculate_recency(self) -> float:
        """How recently was this mentioned? Fades after 5 minutes"""
        time_since = time.time() - self.last_mentioned
        return max(0, 1.0 - (time_since / 300))


@dataclass
class Priority:
    """Something the customer wants to achieve"""

    goal: str
    importance: float  # 0-1 scale
    timeline: Optional[str] = None  # immediate, short-term, long-term
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EngagementPattern:
    """Tracks how engaged the customer is getting over time"""

    timestamps: List[float] = field(default_factory=list)
    response_lengths: List[int] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    sentiment_scores: List[float] = field(default_factory=list)
    question_ratio: float = 0.0  # How many questions vs statements

    def add_interaction(
        self,
        response_length: int,
        response_time: float,
        sentiment: float,
        is_question: bool,
    ):
        self.timestamps.append(time.time())
        self.response_lengths.append(response_length)
        self.response_times.append(response_time)
        self.sentiment_scores.append(sentiment)

        # Update question ratio
        total = len(self.timestamps)
        if is_question:
            self.question_ratio = ((self.question_ratio * (total - 1)) + 1) / total
        else:
            self.question_ratio = (self.question_ratio * (total - 1)) / total

    def get_trend(self) -> str:
        """Are they getting more or less engaged?"""
        if len(self.response_lengths) < 3:
            return "insufficient_data"

        # Compare recent responses to early ones
        recent = self.response_lengths[-3:]
        earlier = self.response_lengths[:3]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        if recent_avg > earlier_avg * 1.3:
            return "increasing"
        elif recent_avg < earlier_avg * 0.7:
            return "decreasing"
        else:
            return "stable"

    def get_engagement_score(self) -> float:
        """Overall engagement score from 0-1"""
        if not self.response_lengths:
            return 0.5

        scores = []

        # Longer responses = more engaged
        avg_length = sum(self.response_lengths) / len(self.response_lengths)
        length_score = min(1.0, avg_length / 50)
        scores.append(length_score)

        # Sentiment matters too
        if self.sentiment_scores:
            avg_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores)
            sentiment_score = (avg_sentiment + 1) / 2
            scores.append(sentiment_score)

        # Questions show they're thinking
        scores.append(min(1.0, self.question_ratio * 2))

        # Quick responses = engaged
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            time_score = max(0, 1.0 - (avg_time / 30))
            scores.append(time_score)

        return sum(scores) / len(scores)


class AdvancedContextManager:
    """Keeps track of what's happening in the conversation"""

    # Pain point categories
    PAIN_CATEGORIES = {
        "operational": [
            "slow",
            "manual",
            "inefficient",
            "time-consuming",
            "complex",
            "difficult",
        ],
        "financial": ["expensive", "cost", "budget", "roi", "investment", "price"],
        "strategic": [
            "competitive",
            "growth",
            "scale",
            "market",
            "innovation",
            "transform",
        ],
        "technical": [
            "integration",
            "compatibility",
            "security",
            "reliability",
            "performance",
        ],
        "organizational": [
            "team",
            "training",
            "adoption",
            "change",
            "culture",
            "process",
        ],
    }

    # Priority indicators
    PRIORITY_INDICATORS = {
        "immediate": ["urgent", "asap", "now", "immediately", "critical", "emergency"],
        "short_term": ["soon", "this quarter", "this month", "quickly", "near term"],
        "long_term": ["future", "next year", "eventually", "planning", "roadmap"],
    }

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

        self.pain_points: Dict[str, PainPoint] = {}
        self.priorities: List[Priority] = []
        self.engagement_pattern = EngagementPattern()
        self.decision_stage = DecisionStage.UNAWARE
        self.engagement_level = EngagementLevel.NEUTRAL

        self.mentioned_competitors: Set[str] = set()
        self.budget_mentioned: Optional[str] = None
        self.timeline_mentioned: Optional[str] = None
        self.decision_makers: List[str] = []
        self.key_requirements: List[str] = []

        self.topic_history: List[str] = []
        self.topic_interest_scores: Dict[str, float] = defaultdict(float)

        # Might use these later
        self.objection_patterns: List[Dict] = []
        self.agreement_patterns: List[Dict] = []
        self.confusion_points: List[Dict] = []

    def update_from_turn(
        self,
        user_input: str,
        classification_result: Dict = None,
        sentiment: Dict = None,
    ):
        """Process what the customer just said"""

        self._extract_pain_points(user_input)
        self._extract_priorities(user_input)

        if sentiment:
            self._update_engagement_pattern(user_input, sentiment)

        self._update_decision_stage(user_input, classification_result)
        self._extract_additional_context(user_input)

        if classification_result and "primary_intent" in classification_result:
            self._track_topic(classification_result["primary_intent"])

        if self.debug_mode:
            print(f"CONTEXT: Context updated:")
            print(f"   Pain points: {len(self.pain_points)}")
            print(f"   Decision stage: {self.decision_stage.value}")
            print(f"   Engagement: {self.engagement_level.value}")

    def _extract_pain_points(self, text: str):
        """Look for problems the customer mentions"""

        text_lower = text.lower()

        pain_indicators = [
            "problem",
            "issue",
            "challenge",
            "struggle",
            "difficult",
            "frustrat",
            "pain",
            "waste",
            "slow",
            "manual",
            "inefficient",
            "expensive",
            "complex",
            "broken",
            "failing",
            "can't",
            "unable",
        ]

        for indicator in pain_indicators:
            if indicator in text_lower:
                words = text_lower.split()
                if indicator in words:
                    idx = words.index(indicator)
                    start = max(0, idx - 5)
                    end = min(len(words), idx + 6)
                    context = " ".join(words[start:end])

                    category = self._categorize_pain_point(context)
                    severity = self._calculate_pain_severity(context)

                    pain_key = f"{category}:{indicator}"
                    if pain_key in self.pain_points:
                        self.pain_points[pain_key].update_mention()
                    else:
                        self.pain_points[pain_key] = PainPoint(
                            description=context,
                            severity=severity,
                            category=category,
                        )

                    if self.debug_mode:
                        print(
                            f"   PAIN_POINT: Pain point identified: {category} - {context[:50]}..."
                        )

                break

    def _categorize_pain_point(self, text: str) -> str:
        """Figure out what type of problem this is"""

        for category, keywords in self.PAIN_CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return category

        return "general"

    def _calculate_pain_severity(self, text: str) -> float:
        """How bad does this problem sound?"""

        high_intensity = [
            "very",
            "extremely",
            "incredibly",
            "totally",
            "completely",
            "absolutely",
        ]
        medium_intensity = ["quite", "pretty", "fairly", "rather", "somewhat"]

        severity = 0.5

        for word in high_intensity:
            if word in text:
                severity = min(1.0, severity + 0.3)

        for word in medium_intensity:
            if word in text:
                severity = min(1.0, severity + 0.1)

        # These words make it sound worse
        impact_words = ["losing", "wasting", "failing", "missing", "broken"]
        for word in impact_words:
            if word in text:
                severity = min(1.0, severity + 0.2)

        return severity

    def _extract_priorities(self, text: str):
        """Look for goals and things they want to accomplish"""

        text_lower = text.lower()

        goal_indicators = [
            "want to",
            "need to",
            "trying to",
            "goal is",
            "objective is",
            "looking for",
            "hoping to",
            "plan to",
            "aim to",
            "focus on",
        ]

        for indicator in goal_indicators:
            if indicator in text_lower:
                idx = text_lower.find(indicator)
                goal_text = text_lower[idx : idx + 100].split(".")[0]

                timeline = None
                for time_category, keywords in self.PRIORITY_INDICATORS.items():
                    if any(keyword in goal_text for keyword in keywords):
                        timeline = time_category
                        break

                importance = self._calculate_priority_importance(goal_text)

                priority = Priority(
                    goal=goal_text,
                    importance=importance,
                    timeline=timeline,
                )

                self.priorities.append(priority)

                if self.debug_mode:
                    print(
                        f"   PRIORITY: Priority identified: {goal_text[:50]}... ({timeline})"
                    )

                break

    def _calculate_priority_importance(self, text: str) -> float:
        """How important does this goal sound?"""

        importance = 0.5

        high_importance = ["critical", "essential", "must", "key", "vital", "crucial"]
        for word in high_importance:
            if word in text:
                importance = min(1.0, importance + 0.3)

        urgent_words = ["urgent", "asap", "immediately", "now"]
        for word in urgent_words:
            if word in text:
                importance = min(1.0, importance + 0.2)

        return importance

    def _update_engagement_pattern(self, user_input: str, sentiment: Dict):
        """Track how engaged they seem"""

        response_length = len(user_input.split())
        response_time = 1.0  # TODO: calculate from actual response time
        sentiment_score = sentiment.get("sentiment_score", 0)
        is_question = sentiment.get("is_question", False)

        self.engagement_pattern.add_interaction(
            response_length, response_time, sentiment_score, is_question
        )

        engagement_score = self.engagement_pattern.get_engagement_score()

        if engagement_score > 0.7:
            self.engagement_level = EngagementLevel.ENGAGED
        elif engagement_score > 0.5:
            self.engagement_level = EngagementLevel.INTERESTED
        elif engagement_score > 0.3:
            self.engagement_level = EngagementLevel.NEUTRAL
        else:
            self.engagement_level = EngagementLevel.RESISTANT

        # Really negative = hostile
        if (
            sentiment.get("primary_sentiment") == "negative"
            and sentiment.get("sentiment_score", 0) < -0.5
        ):
            self.engagement_level = EngagementLevel.HOSTILE

    def _update_decision_stage(self, user_input: str, classification_result: Dict):
        """Figure out where they are in the buying process"""

        text_lower = user_input.lower()

        stage_indicators = {
            DecisionStage.PROBLEM_AWARE: [
                "problem",
                "issue",
                "challenge",
                "struggling",
                "difficult",
            ],
            DecisionStage.SOLUTION_AWARE: [
                "solution",
                "options",
                "alternatives",
                "approaches",
                "methods",
            ],
            DecisionStage.PRODUCT_AWARE: [
                "your product",
                "your solution",
                "what you offer",
                "your service",
            ],
            DecisionStage.EVALUATION: [
                "compare",
                "vs",
                "versus",
                "difference",
                "better than",
                "advantages",
            ],
            DecisionStage.DECISION: [
                "decide",
                "decision",
                "choose",
                "select",
                "go with",
            ],
            DecisionStage.PURCHASE: [
                "buy",
                "purchase",
                "sign up",
                "subscribe",
                "get started",
            ],
        }

        for stage, keywords in stage_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                # Only move forward, never backward
                if stage.value > self.decision_stage.value:
                    self.decision_stage = stage
                    if self.debug_mode:
                        print(f"   Decision stage progressed to: {stage.value}")
                break

    def _extract_additional_context(self, text: str):
        """Pick up other useful context clues"""

        text_lower = text.lower()

        # Competitors mentioned?
        competitor_keywords = [
            "competitor",
            "alternative",
            "currently using",
            "switched from",
        ]
        for keyword in competitor_keywords:
            if keyword in text_lower:
                words = text_lower.split()
                if keyword in " ".join(words):
                    self.mentioned_competitors.add("competitor_mentioned")

        # Budget talk?
        budget_keywords = ["budget", "cost", "price", "invest", "spend", "$"]
        for keyword in budget_keywords:
            if keyword in text_lower:
                self.budget_mentioned = "budget_discussed"
                break

        # Timeline mentioned?
        timeline_keywords = ["when", "timeline", "timeframe", "by when", "deadline"]
        for keyword in timeline_keywords:
            if keyword in text_lower:
                self.timeline_mentioned = "timeline_discussed"
                break

        # Who makes decisions?
        decision_keywords = [
            "my boss",
            "our team",
            "management",
            "ceo",
            "cto",
            "decision maker",
        ]
        for keyword in decision_keywords:
            if keyword in text_lower:
                self.decision_makers.append(keyword)
                break

    def _track_topic(self, topic: str):
        """Keep track of what they're interested in"""

        self.topic_history.append(topic)
        self.topic_interest_scores[topic] += 1.0

        # Older topics become less important
        for t in self.topic_interest_scores:
            if t != topic:
                self.topic_interest_scores[t] *= 0.9

    def get_top_pain_points(self, n: int = 3) -> List[PainPoint]:
        if not self.pain_points:
            return []

        # Sort by severity * recency
        sorted_points = sorted(
            self.pain_points.values(),
            key=lambda p: p.severity * p._calculate_recency(),
            reverse=True,
        )

        return sorted_points[:n]

    def get_primary_priorities(self, n: int = 3) -> List[Priority]:
        if not self.priorities:
            return []

        sorted_priorities = sorted(
            self.priorities, key=lambda p: p.importance, reverse=True
        )

        return sorted_priorities[:n]

    def get_context_summary(self) -> Dict[str, Any]:
        """Everything we know about this conversation"""
        return {
            "decision_stage": self.decision_stage.value,
            "engagement_level": self.engagement_level.value,
            "engagement_score": self.engagement_pattern.get_engagement_score(),
            "engagement_trend": self.engagement_pattern.get_trend(),
            "pain_points": [p.to_dict() for p in self.get_top_pain_points()],
            "priorities": [p.to_dict() for p in self.get_primary_priorities()],
            "competitors_mentioned": list(self.mentioned_competitors),
            "budget_discussed": self.budget_mentioned is not None,
            "timeline_discussed": self.timeline_mentioned is not None,
            "decision_makers": self.decision_makers,
            "top_topics": self._get_top_topics(),
            "conversation_depth": len(self.topic_history),
        }

    def _get_top_topics(self, n: int = 5) -> List[tuple]:
        sorted_topics = sorted(
            self.topic_interest_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_topics[:n]

    def should_escalate(self) -> bool:
        """Should we hand this off to a human?"""

        # Hostile customers need human touch
        if self.engagement_level == EngagementLevel.HOSTILE:
            return True

        # Lots of serious problems = escalate
        high_severity_points = [
            p for p in self.pain_points.values() if p.severity > 0.8
        ]
        if len(high_severity_points) > 2:
            return True

        # Ready to buy? Get a human involved
        if self.decision_stage == DecisionStage.PURCHASE:
            return True

        return False

    def get_recommended_approach(self) -> str:
        """What conversation style should we use?"""

        if self.engagement_level in [
            EngagementLevel.HOSTILE,
            EngagementLevel.RESISTANT,
        ]:
            return "empathy_and_understanding"

        if self.decision_stage in [DecisionStage.UNAWARE, DecisionStage.PROBLEM_AWARE]:
            return "education_and_discovery"

        if self.decision_stage in [
            DecisionStage.SOLUTION_AWARE,
            DecisionStage.PRODUCT_AWARE,
        ]:
            return "value_demonstration"

        if self.decision_stage in [DecisionStage.EVALUATION, DecisionStage.DECISION]:
            return "comparison_and_proof"

        if len(self.pain_points) > 2:
            return "solution_focused"

        return "consultative"


# Quick test
if __name__ == "__main__":
    context_mgr = AdvancedContextManager(debug_mode=True)

    turns = [
        {
            "input": "We're struggling with manual lead qualification. It takes our team hours every day.",
            "sentiment": {"sentiment_score": -0.3, "is_question": False},
        },
        {
            "input": "We need to find a solution quickly, ideally this quarter. It's becoming critical.",
            "sentiment": {"sentiment_score": 0.2, "is_question": False},
        },
        {
            "input": "What makes your solution different from competitors?",
            "sentiment": {"sentiment_score": 0.0, "is_question": True},
        },
        {
            "input": "Our budget is limited, but ROI is important. When can we see results?",
            "sentiment": {"sentiment_score": 0.1, "is_question": True},
        },
    ]

    print("CONTEXT: Advanced Context Tracking Demo\n")

    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}: {turn['input'][:50]}...")
        context_mgr.update_from_turn(turn["input"], sentiment=turn["sentiment"])

    print("\n" + "=" * 50)
    print("SUMMARY: Context Summary:")
    import json

    summary = context_mgr.get_context_summary()
    print(json.dumps(summary, indent=2))

    print(
        f"\nRECOMMENDED: Recommended Approach: {context_mgr.get_recommended_approach()}"
    )
    print(f"WARNING: Should Escalate: {context_mgr.should_escalate()}")
