"""
3-stage classification system for understanding user intent
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llm_bot_engine import (
    ClassificationType,
    QuestionType,
    IntentContext,
    PartialMatch,
    ClassificationResult,
    ConversationState,
)

from advanced_prompts import AdvancedPrompts, PromptSelector, PromptOptimizer


class MultiStageClassifier:
    """Handles intent classification through 3 stages"""

    def __init__(self, tree_manager, debug_mode: bool = False):
        self.tree_manager = tree_manager
        self.debug_mode = debug_mode
        self.client = None
        self.prompt_optimizer = PromptOptimizer()
        self.initialize_model()

    def initialize_model(self):
        if OpenAI:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("No API key found. Set OPENAI_API_KEY")
                self.client = None

    def classify(
        self, user_input: str, state: ConversationState
    ) -> ClassificationResult:
        """Run all 3 classification stages"""
        start_time = time.time()
        result = ClassificationResult(
            stage_completed=0,
            classification_type=ClassificationType.NO_MATCH,
            partial_matches=[],
            reasoning="",
        )

        try:
            if self.debug_mode:
                print("\nSTAGE 1: Analyzing current node...")

            stage1_result = self._classify_stage1_current_node(user_input, state)
            result.stage_completed = 1

            if stage1_result.get("has_full_match"):
                full_match = self._extract_full_match(stage1_result, state.current_node)
                if full_match:
                    result.classification_type = ClassificationType.FULL_MATCH
                    result.full_match = full_match
                    result.processing_time = time.time() - start_time
                    return result

            partials_stage1 = self._extract_partial_matches(
                stage1_result, state.current_node, stage=1
            )
            result.partial_matches.extend(partials_stage1)

            if self.debug_mode:
                print("\nSTAGE 2: Global search...")

            covered_aspects = [p.addresses_aspect for p in result.partial_matches]
            stage2_result = self._classify_stage2_global_search(
                user_input, covered_aspects, state
            )
            result.stage_completed = 2

            if stage2_result.get("found_better_full_match"):
                global_full_match = self._extract_global_full_match(stage2_result)
                if global_full_match:
                    result.classification_type = ClassificationType.FULL_MATCH
                    result.full_match = global_full_match
                    result.routing_decision = {
                        "target": stage2_result.get("recommended_routing")
                    }
                    result.processing_time = time.time() - start_time
                    return result

            partials_stage2 = self._extract_global_partials(stage2_result, stage=2)
            result.partial_matches.extend(partials_stage2)

            if self.debug_mode:
                print("\nSTAGE 3: Conceptual classification...")

            stage3_result = self._classify_stage3_conceptual(
                user_input, result.partial_matches, state
            )
            result.stage_completed = 3
            result.question_classification = stage3_result

            if result.partial_matches:
                result.classification_type = ClassificationType.PARTIAL_MATCHES
            else:
                result.classification_type = ClassificationType.NO_MATCH

            # Keep only top 5 matches
            if len(result.partial_matches) > 5:
                result.partial_matches = sorted(
                    result.partial_matches,
                    key=lambda x: x.relevance_score,
                    reverse=True,
                )[:5]

        except Exception as e:
            # Re-raise rate limit errors for retry
            if "429" in str(e) or "quota" in str(e).lower():
                raise
            if self.debug_mode:
                print(f"Classification error: {e}")
            result.reasoning = f"Classification failed: {str(e)}"

        result.processing_time = time.time() - start_time
        return result

    def _classify_stage1_current_node(
        self, user_input: str, state: ConversationState
    ) -> Dict:
        """Check intents in current conversation node"""

        current_intents = self.tree_manager.get_current_node_intents(state.current_node)
        if not current_intents:
            return {"has_full_match": False, "classifications": [], "partial_count": 0}

        intent_descriptions = []
        for intent in current_intents:
            preview = intent.prompt[:300] if intent.prompt else "No response available"
            intent_descriptions.append(
                f'- {intent.intent_name}: "{intent.prompt_summary}"\n'
                f'  Full Response Preview: "{preview}..."'
            )

        cognitive_config = PromptSelector.get_cognitive_depth(
            state.turn_count, "standard"
        )

        domain = self.tree_manager.config_manager.get_domain()
        prompt_config = self.tree_manager.config_manager.get_prompt_config()

        context_data = {
            "current_node": state.current_node,
            "user_input": user_input,
            "conversation_context": state.get_recent_context(),
            "available_intents": "\n".join(intent_descriptions),
            "context": state.get_recent_context(),
            "domain": domain,
            "domain_context": prompt_config.get("domain_context", "conversation"),
            "goals": prompt_config.get("conversation_goals", "advance conversation"),
            "salient_points": f"Turn {state.turn_count}, Node: {state.current_node}",
        }

        prompt = AdvancedPrompts.get_prompt(
            "intent_classification",
            **context_data,
            use_cognitive_framework=cognitive_config["use_cognitive_framework"],
        )

        optimization = self.prompt_optimizer.optimize_prompt(
            prompt, {"conversation_stage": state.current_node}
        )
        response = self._call_ai_model(
            optimization["prompt"],
            temperature=optimization["temperature"],
            max_tokens=2000,
            conversation_state=state,
        )

        try:
            # Clean markdown formatting
            clean_response = response
            if response.startswith("```json"):
                clean_response = response[7:]
            elif response.startswith("```"):
                clean_response = response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            result = json.loads(clean_response)
            if "has_full_match" in result and hasattr(
                self.prompt_optimizer, "add_performance_feedback"
            ):
                self.prompt_optimizer.add_performance_feedback(
                    "intent_classification", result["has_full_match"]
                )
            return result
        except json.JSONDecodeError:
            return {"has_full_match": False, "classifications": [], "partial_count": 0}

    def _classify_stage2_global_search(
        self, user_input: str, covered_aspects: List[str], state: ConversationState
    ) -> Dict:
        """Search all conversation nodes for matches"""

        all_intents = self.tree_manager.get_all_intents()

        # Group by conversation stage
        intents_by_stage = {}
        for intent in all_intents:
            stage = intent.conversation_stage or "general"
            if stage not in intents_by_stage:
                intents_by_stage[stage] = []
            intents_by_stage[stage].append(
                f"  • {intent.intent_name} ({intent.source_node}): {intent.prompt_summary}"
            )

        all_intents_text = []
        for stage, intents in intents_by_stage.items():
            all_intents_text.append(f"\n[{stage.upper()} STAGE]")
            all_intents_text.extend(intents[:20])  # Limit to avoid token overflow

        domain = (
            self.tree_manager.config_manager.get_domain()
            if hasattr(self.tree_manager, "config_manager")
            else "sales"
        )

        prompt = AdvancedPrompts.get_prompt(
            "global_search",
            domain=domain,
            user_input=user_input,
            current_node=state.current_node,
            covered_aspects=(", ".join(covered_aspects) if covered_aspects else "None"),
            all_intents="\n".join(all_intents_text),
        )

        response = self._call_ai_model(
            prompt, temperature=0.4, max_tokens=2000, conversation_state=state
        )

        try:
            clean_response = response
            if response.startswith("```json"):
                clean_response = response[7:]
            elif response.startswith("```"):
                clean_response = response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            result = json.loads(clean_response)
            if hasattr(self.prompt_optimizer, "add_performance_feedback"):
                self.prompt_optimizer.add_performance_feedback(
                    "global_search", result.get("found_better_full_match", False)
                )
            return result
        except json.JSONDecodeError:
            return {
                "found_better_full_match": False,
                "global_matches": [],
                "new_aspects_covered": [],
            }

    def _classify_stage3_conceptual(
        self,
        user_input: str,
        partial_matches: List[PartialMatch],
        state: ConversationState,
    ) -> Dict:
        """Final conceptual analysis and response planning"""

        partials_text = []
        for p in partial_matches[:5]:
            partials_text.append(
                f"- {p.intent} (Stage {p.extraction_stage}): Addresses '{p.addresses_aspect}' "
                f"(relevance: {p.relevance_score:.2f})"
            )

        current_node = self.tree_manager.get_node(state.current_node)
        conversation_stage = (
            current_node.get("conversation_stage", "unknown")
            if current_node
            else "unknown"
        )

        domain = (
            self.tree_manager.config_manager.get_domain()
            if hasattr(self.tree_manager, "config_manager")
            else "sales"
        )
        all_intents = self.tree_manager.get_all_intents()
        intents_by_stage = {}
        for intent in all_intents:
            stage = intent.conversation_stage or "general"
            if stage not in intents_by_stage:
                intents_by_stage[stage] = []
            intents_by_stage[stage].append(
                f"  • {intent.intent_name} ({intent.source_node}): {intent.prompt_summary}"
            )

        all_intents_text = []
        for stage, intents in intents_by_stage.items():
            all_intents_text.append(f"\n[{stage.upper()} STAGE]")
            all_intents_text.extend(intents[:20])

        prompt = AdvancedPrompts.get_prompt(
            "conceptual_classification",
            domain=domain,
            user_input=user_input,
            current_node=state.current_node,
            partial_matches=(
                "\n".join(partials_text)
                if partials_text
                else "No partial matches found"
            ),
            conversation_stage=conversation_stage,
            all_intents="\n".join(all_intents_text),
        )

        response = self._call_ai_model(
            prompt, temperature=0.2, max_tokens=1500, conversation_state=state
        )

        try:
            clean_response = response
            if response.startswith("```json"):
                clean_response = response[7:]
            elif response.startswith("```"):
                clean_response = response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            result = json.loads(clean_response)
            self.prompt_optimizer.add_performance_feedback(
                "conceptual_classification", result.get("confidence", 0) > 0.7
            )
            return result
        except:
            return {
                "question_type": "COMPANY_SPECIFIC",
                "identified_aspects": [],
                "covered_aspects": [],
                "gaps": [],
                "response_strategy": "GENERAL_EXPLANATION",
                "combination_plan": "",
                "confidence": 0.5,
            }

    def _call_ai_model(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: float = 5.0,
        conversation_state=None,
    ) -> str:

        if not self.client:
            return json.dumps(
                {
                    "classifications": [],
                    "has_full_match": False,
                    "partial_count": 0,
                    "fallback_used": True,
                }
            )

        try:

            messages = []

            if conversation_state and hasattr(
                conversation_state, "get_company_context_for_openai"
            ):
                company_context = conversation_state.get_company_context_for_openai()
                messages.append({"role": "system", "content": company_context})

            if conversation_state and hasattr(
                conversation_state, "get_openai_messages"
            ):
                history_messages = conversation_state.get_openai_messages(num_turns=3)
                messages.extend(history_messages)

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )

            return response.choices[0].message.content

        except Exception as e:
            # Re-raise rate limit errors for retry
            if "429" in str(e) or "quota" in str(e).lower():
                raise

            return json.dumps(
                {
                    "classifications": [],
                    "has_full_match": False,
                    "partial_count": 0,
                    "fallback_used": True,
                }
            )

    def _extract_full_match(
        self, stage1_result: Dict, current_node: str
    ) -> Optional[Dict]:

        classifications = stage1_result.get("classifications", [])
        for cls in classifications:
            confidence = cls.get("confidence", 0)
            if cls.get("match_type") == "FULL" and confidence > 0.7:
                intents = self.tree_manager.get_current_node_intents(current_node)
                for intent in intents:
                    if intent.intent_name == cls.get("intent_name"):
                        return {
                            "intent": intent.intent_name,
                            "target": intent.target_node,
                            "prompt": intent.prompt,
                            "confidence": confidence,
                            "reasoning": cls.get("reasoning", ""),
                        }
        return None

    def _extract_partial_matches(
        self, stage1_result: Dict, current_node: str, stage: int
    ) -> List[PartialMatch]:

        partials = []
        classifications = stage1_result.get("classifications", [])
        intents = self.tree_manager.get_current_node_intents(current_node)

        for cls in classifications:
            if cls.get("match_type") == "PARTIAL" and cls.get("confidence", 0) > 0.5:
                for intent in intents:
                    if intent.intent_name == cls.get("intent_name"):
                        partial = PartialMatch(
                            intent=intent.intent_name,
                            source_node=current_node,
                            target_node=intent.target_node,
                            prompt_summary=intent.prompt_summary,
                            relevance_score=cls.get("confidence", 0.6),
                            addresses_aspect=cls.get("addresses_aspect", "unknown"),
                            confidence=cls.get("confidence", 0.6),
                            extraction_stage=stage,
                        )
                        partials.append(partial)
                        break
        return partials

    def _extract_global_full_match(self, stage2_result: Dict) -> Optional[Dict]:

        matches = stage2_result.get("classifications", [])
        for match in matches:
            confidence = match.get("confidence", 0)
            if confidence > 0.7:
                intent_name = match.get("intent_name")
                if not intent_name:
                    continue

                intent_prompt = ""
                all_intents = self.tree_manager.get_all_intents()
                for intent in all_intents:
                    if intent.intent_name == intent_name:
                        intent_prompt = intent.prompt
                        break

                if intent_prompt:
                    return {
                        "intent": intent_name,
                        "target": match.get("source_node"),
                        "prompt": intent_prompt,
                        "confidence": confidence,
                        "reasoning": match.get("reasoning", ""),
                    }
        return None

    def _extract_global_partials(
        self, stage2_result: Dict, stage: int
    ) -> List[PartialMatch]:

        partials = []
        matches = stage2_result.get("global_matches", [])

        for match in matches:
            if (
                match.get("match_type") == "PARTIAL"
                and match.get("confidence", 0) > 0.5
            ):
                all_intents = self.tree_manager.get_all_intents()
                for intent in all_intents:
                    if intent.intent_name == match.get("intent_name"):
                        partial = PartialMatch(
                            intent=intent.intent_name,
                            source_node=intent.source_node,
                            target_node=intent.target_node,
                            prompt_summary=intent.prompt_summary,
                            relevance_score=match.get("confidence", 0.6),
                            addresses_aspect=match.get("addresses_aspect", "unknown"),
                            confidence=match.get("confidence", 0.6),
                            extraction_stage=stage,
                        )
                        partials.append(partial)
                        break
        return partials


class ResponseGenerator:
    """Generates responses from classification results"""

    def __init__(self, tree_manager, debug_mode: bool = False):
        self.tree_manager = tree_manager
        self.debug_mode = debug_mode
        self.client = None
        self.use_advanced_prompts = True
        self.prompt_optimizer = PromptOptimizer()
        self.initialize_model()

    def initialize_model(self):
        if OpenAI:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("No API key found. Set OPENAI_API_KEY")
                self.client = None

    def generate_response(
        self,
        classification: ClassificationResult,
        user_input: str,
        state: ConversationState,
        context_data: dict = None,
    ) -> str:

        if classification.has_full_match():
            return self._generate_full_match_response(classification, state)

        if classification.has_partials():
            return self._generate_partial_combination_response(
                classification, user_input, state
            )

        return self._generate_no_match_response(
            classification, user_input, state, context_data
        )

    def _generate_full_match_response(
        self, classification: ClassificationResult, state: ConversationState
    ) -> str:

        prompt_template = classification.full_match.get("prompt", "")

        if not prompt_template:
            target_node = classification.full_match.get("target")
            if target_node:
                node = self.tree_manager.get_node(target_node)
                if node:
                    prompt_template = node.get("prompt", "")

        if not prompt_template:
            return "I understand. Let me help you with that."

        personalized_content = self.tree_manager.personalize_prompt(
            prompt_template, state.client_name
        )

        return self._generate_advanced_response(
            classification=classification,
            user_input="",
            state=state,
            content_information=personalized_content,
            match_type="FULL_MATCH",
            strategy="DELIVER_PROVEN_CONTENT",
        )

    def _generate_advanced_response(
        self,
        classification: ClassificationResult,
        user_input: str,
        state: ConversationState,
        content_information: str,
        match_type: str,
        strategy: str,
        gaps: list = None,
    ) -> str:

        if not self.client:
            if match_type == "FULL_MATCH":
                return content_information
            else:
                return "I understand you have several questions. Let me address what I can help with right now."

        if gaps is None:
            gaps = []

        current_node = self.tree_manager.get_node(state.current_node)
        conversation_stage = (
            current_node.get("conversation_stage", "discovery")
            if current_node
            else "discovery"
        )

        domain = (
            self.tree_manager.config_manager.get_domain()
            if hasattr(self.tree_manager, "config_manager")
            else "sales"
        )

        prompt = AdvancedPrompts.get_prompt(
            "response_generation",
            domain=domain,
            representative_name=self.tree_manager.config_manager.get_rep_name(),
            company_name=self.tree_manager.config_manager.get_company_name(),
            user_input=user_input,
            conversation_context=state.get_recent_context(),
            content_information=content_information,
            match_type=match_type,
            strategy=strategy,
            gaps=", ".join(gaps) if gaps else "None",
            conversation_goals=self.tree_manager.config_manager.get_prompt_config().get(
                "conversation_goals", ["engage", "qualify", "advance"]
            ),
        )

        optimization = self.prompt_optimizer.optimize_prompt(
            prompt, {"conversation_stage": conversation_stage}
        )

        try:
            messages = []

            if hasattr(state, "get_company_context_for_openai"):
                company_context = state.get_company_context_for_openai()
                messages.append({"role": "user", "content": company_context})

            if hasattr(state, "get_openai_messages"):
                history_messages = state.get_openai_messages(num_turns=3)
                messages.extend(history_messages)

            messages.append({"role": "user", "content": optimization["prompt"]})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=optimization["max_tokens"],
                temperature=optimization["temperature"],
                messages=messages,
            )
            response_text = response.choices[0].message.content.strip()

            response_text = self._clean_response_output(response_text)

            # Keep responses concise (max 70 words)
            word_count = len(response_text.split())
            if word_count > 70:
                original_word_count = (
                    len(content_information.split()) if content_information else 0
                )
                if original_word_count <= 70:
                    words = response_text.split()
                    truncated = " ".join(words[:60])
                    if "." in truncated:
                        last_period = truncated.rfind(".")
                        if last_period > len(truncated) * 0.7:
                            response_text = truncated[: last_period + 1]
                    else:
                        response_text = truncated + "..."

            if match_type == "FULL_MATCH":
                if (
                    len(response_text) < 20
                    or "template" in response_text.lower()
                    or self._has_meta_commentary(response_text)
                ):
                    return content_information

            return response_text

        except Exception as e:
            # Re-raise rate limit errors for retry
            if "429" in str(e) or "quota" in str(e).lower():
                raise

            if match_type == "FULL_MATCH":
                return content_information
            else:
                return "I understand you have several questions. Let me address what I can help with right now."

    def _clean_response_output(self, response: str) -> str:
        """Clean up response to remove meta commentary and formatting issues"""
        if not response:
            return response

        # Remove common meta commentary patterns
        meta_patterns = [
            r"This response.*?:",
            r"The response.*?:",
            r"The key.*?:",
            r"Key elements.*?:",
            r"This version.*?:",
            r"Would you like.*?adjust.*?",
            r"Does this.*?resonate.*?",
            r"Want to.*?dive deeper.*?",
            r"Thoughts\?$",
            r"Sound fair\?$",
            r"Make sense\?$",
        ]

        import re

        cleaned = response
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Remove bullet points that explain the response
        lines = cleaned.split("\n")
        clean_lines = []
        skip_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip explanation sections
            if any(
                phrase in line.lower()
                for phrase in [
                    "this response maintains",
                    "the response maintains",
                    "key elements are",
                    "the key is",
                    "this version:",
                    "would you like me to",
                ]
            ):
                skip_section = True
                continue

            # Skip bullet point explanations
            if skip_section and (line.startswith("-") or line.startswith("•")):
                continue
            elif not line.startswith("-") and not line.startswith("•"):
                skip_section = False

            if not skip_section:
                clean_lines.append(line)

        # Join and clean up
        cleaned = " ".join(clean_lines)
        cleaned = re.sub(r"\s+", " ", cleaned)  # Remove extra whitespace
        return cleaned.strip()

    def _has_meta_commentary(self, response: str) -> bool:
        """Check if response contains meta commentary that should be removed"""
        meta_indicators = [
            "this response",
            "the response",
            "this version",
            "the key is",
            "key elements",
            "would you like me to",
            "does this",
            "want to dive deeper",
            "thoughts?",
            "make sense?",
            "sound fair?",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in meta_indicators)

    def _generate_partial_combination_response(
        self,
        classification: ClassificationResult,
        user_input: str,
        state: ConversationState,
    ) -> str:

        if not self.client:
            if classification.partial_matches:
                best_partial = classification.partial_matches[0]
                node = self.tree_manager.get_node(best_partial.target_node)
                if node and node.get("prompt"):
                    return self.tree_manager.personalize_prompt(
                        node.get("prompt", ""), state.client_name
                    )
                else:
                    return f"I can help with {best_partial.addresses_aspect}. Could you be more specific about what you'd like to know?"
            return "Could you clarify what specific aspect you're asking about?"

        partial_info = []
        for partial in classification.partial_matches[:5]:
            node = self.tree_manager.get_node(partial.target_node)
            if node:
                partial_info.append(
                    f"For '{partial.addresses_aspect}':\n{node.get('prompt', node.get('prompt_summary', ''))[:500]}"
                )

        strategy = "COMBINE_PARTIALS"
        gaps = []

        if classification.question_classification:
            strategy = classification.question_classification.get(
                "response_strategy", "COMBINE_PARTIALS"
            )
            gaps = classification.question_classification.get("gaps", [])

        return self._generate_advanced_response(
            classification=classification,
            user_input=user_input,
            state=state,
            content_information="\n\n".join(partial_info),
            match_type="PARTIAL_MATCHES",
            strategy=strategy,
            gaps=gaps,
        )

    def _generate_no_match_response(
        self,
        classification: ClassificationResult,
        user_input: str,
        state: ConversationState,
        context_data: dict = None,
    ) -> str:

        domain = (
            self.tree_manager.config_manager.get_domain()
            if hasattr(self.tree_manager, "config_manager")
            else "sales"
        )

        question_type = "UNKNOWN"
        if classification.question_classification:
            question_type = classification.question_classification.get(
                "question_type", "UNKNOWN"
            )

        formatted_context_data = "None available"
        consecutive_no_matches = 0
        if context_data:
            context_parts = []
            if context_data.get("pain_points"):
                pain_points = [
                    p.get("description", "")
                    for p in context_data.get("pain_points", [])
                ]
                context_parts.append(f"Pain Points: {', '.join(pain_points[:2])}")
            if context_data.get("priorities"):
                priorities = [
                    p.get("goal", "") for p in context_data.get("priorities", [])
                ]
                context_parts.append(f"Priorities: {', '.join(priorities[:2])}")
            if context_data.get("engagement_level"):
                context_parts.append(
                    f"Engagement: {context_data.get('engagement_level')}"
                )

            consecutive_no_matches = context_data.get("consecutive_no_matches", 0)

            if context_parts:
                formatted_context_data = " | ".join(context_parts)

        prompt = AdvancedPrompts.get_prompt(
            "no_match_response",
            domain=domain,
            representative_name=self.tree_manager.config_manager.get_rep_name(),
            user_input=user_input,
            question_type=question_type,
            conversation_context=state.get_recent_context(),
            current_node=state.current_node,
            context_data=formatted_context_data,
            consecutive_no_matches=consecutive_no_matches,
        )

        try:
            response = self._call_ai_model(
                prompt,
                temperature=0.6,
                max_tokens=200,
            )

            response = response.strip()
            if not response:
                return "I want to make sure I understand your question correctly. Could you tell me more about what you're looking for?"

            return response

        except Exception:
            return (
                "I want to make sure I understand your question correctly. "
                "Could you tell me more about what specific aspect you're asking about?"
            )

    def _call_ai_model(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 200,
        timeout: float = 5.0,
    ) -> str:

        if not self.client:
            return "I want to make sure I understand your question correctly. Could you tell me more about what you're looking for?"

        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1]

            return response_text

        except Exception:
            return "I want to make sure I understand your question correctly. Could you tell me more about what you're looking for?"


__all__ = ["MultiStageClassifier", "ResponseGenerator", "ClassificationPrompts"]
