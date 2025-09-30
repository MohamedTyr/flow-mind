"""
Error handling and retry logic for AI calls.
Falls back to template responses when AI fails.
"""

import time
import random
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class RetryConfig:

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


class FallbackTemplates:
    """Backup responses when AI is down"""

    TEMPLATES = {
        "classification_failure": {
            "general": "I understand. Let me help you with that. Could you tell me more about what you're looking for?",
            "sales": "That's a great question. Let me understand your specific needs better. What's most important to you right now?",
            "support": "I see what you're asking about. Let me help you with that. Can you provide more details about the issue?",
            "survey": "Thank you for sharing that. Could you elaborate a bit more on your experience?",
        },
        "response_generation_failure": {
            "general": "I appreciate your patience. Let me address your question directly.",
            "sales": "That's exactly the kind of insight I wanted to share. Let me explain how we can help.",
            "support": "I understand the situation. Here's what we can do to resolve this.",
            "survey": "Thank you for that feedback. Your input is valuable to us.",
        },
        "routing_failure": {
            "general": "Let me make sure I understand correctly. Could you clarify what you'd like to know?",
            "sales": "That's an important point. Let me address that for you.",
            "support": "I want to make sure we solve this correctly. Tell me more about what you need.",
            "survey": "Your perspective is important. Please tell me more about your thoughts on this.",
        },
        "acknowledgment": {
            "confusion": "I want to make sure I understand you correctly. Could you rephrase that?",
            "technical": "Let me get the right information for you. One moment please.",
            "engagement": "That's interesting. Tell me more about that.",
            "clarification": "I want to give you the most accurate information. Can you specify what aspect you're most interested in?",
        },
    }

    @classmethod
    def get_fallback(
        cls, failure_type: str, domain: str = "general", context: Dict = None
    ) -> str:
        templates = cls.TEMPLATES.get(
            failure_type, cls.TEMPLATES["classification_failure"]
        )
        template = templates.get(domain, templates["general"])

        if context:
            if context.get("turn_count", 0) > 5:
                template = f"I appreciate your patience. {template}"
            if context.get("user_sentiment") == "frustrated":
                template = f"I understand this might be frustrating. {template}"
            if context.get("is_retry"):
                template = f"Thank you for waiting. {template}"

        return template

    @classmethod
    def get_contextual_response(
        cls, user_input: str, conversation_stage: str, domain: str
    ) -> str:

        stage_responses = {
            "opening": {
                "sales": "Thank you for taking the time to speak with me. I'd like to understand your current situation better.",
                "support": "I'm here to help. Let me understand what you're experiencing.",
                "survey": "Thank you for participating. Your feedback is important to us.",
            },
            "discovery": {
                "sales": "That's valuable information. Can you tell me more about your current process?",
                "support": "I see. Let me gather a bit more information to help you better.",
                "survey": "That's helpful context. What else can you share about your experience?",
            },
            "objection_handling": {
                "sales": "I completely understand your concern. Let me address that directly.",
                "support": "I understand the frustration. Let's work through this together.",
                "survey": "Your concerns are valid. Please tell me more about what led to this.",
            },
            "closing": {
                "sales": "Based on what you've shared, I think the next step would be to explore this further.",
                "support": "I believe we have a good path forward to resolve this.",
                "survey": "Thank you for your valuable feedback. Is there anything else you'd like to add?",
            },
        }

        stage_dict = stage_responses.get(
            conversation_stage, stage_responses["discovery"]
        )
        response = stage_dict.get(domain, stage_dict["sales"])

        if len(user_input) < 100:
            keywords = [
                "yes",
                "no",
                "maybe",
                "sure",
                "okay",
                "right",
                "exactly",
                "correct",
            ]
            if any(kw in user_input.lower() for kw in keywords):
                response = f"I understand. {response}"

        return response


class AIErrorHandler:
    """Retries failed AI calls and provides fallbacks"""

    def __init__(self, retry_config: RetryConfig = None, debug_mode: bool = False):
        self.retry_config = retry_config or RetryConfig()
        self.debug_mode = debug_mode
        self.error_history = []
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "fallback_uses": 0,
        }

    def with_retry(self, func, *args, **kwargs):
        last_exception = None

        for attempt in range(self.retry_config.max_retries):
            try:
                result = func(*args, **kwargs)

                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                    if self.debug_mode:
                        print(f"Retry {attempt} succeeded")

                return result

            except Exception as e:
                last_exception = e
                self.retry_stats["total_retries"] += 1

                self.error_history.append(
                    {
                        "timestamp": time.time(),
                        "attempt": attempt + 1,
                        "error": str(e),
                        "function": func.__name__,
                    }
                )

                if self.debug_mode:
                    print(f"Attempt {attempt + 1} failed: {e}")

                if attempt >= self.retry_config.max_retries - 1:
                    if self.debug_mode:
                        print(f"All {self.retry_config.max_retries} attempts failed")
                    raise last_exception

                delay = self._calculate_delay(attempt)

                if self.debug_mode:
                    print(f"Waiting {delay:.2f} seconds before retry...")

                time.sleep(delay)

    def _calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.retry_config.base_delay
            * (self.retry_config.exponential_base**attempt),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            delay *= 0.5 + random.random()

        return delay

    def get_fallback_response(
        self,
        failure_type: str,
        domain: str = "general",
        context: Dict = None,
        user_input: str = None,
        conversation_stage: str = None,
    ) -> str:
        self.retry_stats["fallback_uses"] += 1

        if user_input and conversation_stage and domain:
            response = FallbackTemplates.get_contextual_response(
                user_input, conversation_stage, domain
            )
        else:
            response = FallbackTemplates.get_fallback(failure_type, domain, context)

        if self.debug_mode:
            print(f"Using fallback response: {response[:50]}...")

        return response

    def wrap_ai_call(
        self,
        func,
        fallback_type: str,
        domain: str = "general",
        context: Dict = None,
        *args,
        **kwargs,
    ):
        try:
            return self.with_retry(func, *args, **kwargs)

        except Exception as e:
            if self.debug_mode:
                print(f"AI call failed after all retries: {e}")
                print(f"Using fallback strategy: {fallback_type}")

            fallback_context = context or {}
            fallback_context["is_retry"] = True
            fallback_context["error"] = str(e)

            if fallback_type == "classification":
                return {
                    "classifications": [],
                    "has_full_match": False,
                    "partial_count": 0,
                    "fallback_used": True,
                }
            elif fallback_type == "response":
                return self.get_fallback_response(
                    "response_generation_failure",
                    domain,
                    fallback_context,
                    kwargs.get("user_input"),
                    kwargs.get("conversation_stage"),
                )
            elif fallback_type == "routing":
                return {
                    "should_route": False,
                    "target_node": None,
                    "fallback_used": True,
                }
            else:
                return self.get_fallback_response(
                    "classification_failure",
                    domain,
                    fallback_context,
                )

    def get_stats(self) -> Dict:
        total_attempts = (
            self.retry_stats["total_retries"] + self.retry_stats["fallback_uses"]
        )
        fallback_rate = (
            self.retry_stats["fallback_uses"] / max(1, total_attempts)
            if total_attempts > 0
            else 0
        )

        return {
            "retry_stats": self.retry_stats,
            "error_count": len(self.error_history),
            "recent_errors": self.error_history[-5:] if self.error_history else [],
            "fallback_rate": fallback_rate,
        }
