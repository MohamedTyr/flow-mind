"""
Conversational bot engine with multi-stage classification.

Handles dynamic conversation flows based on JSON configuration files.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from config_manager import ConfigManager

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Run: pip install openai")
    sys.exit(1)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o"
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
DEFAULT_MAX_CONVERSATION_LENGTH = 15
DEFAULT_MAX_PARTIAL_MATCHES = 5


class ClassificationType(Enum):

    FULL_MATCH = "FULL_MATCH"
    PARTIAL_MATCHES = "PARTIAL_MATCHES"
    NO_MATCH = "NO_MATCH"


class QuestionType(Enum):

    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    COMPANY_SPECIFIC = "COMPANY_SPECIFIC"
    HIGH_STAKES = "HIGH_STAKES"
    OFF_TOPIC = "OFF_TOPIC"


@dataclass
class IntentContext:

    intent_name: str
    source_node: str
    target_node: str
    prompt: str
    prompt_summary: str
    conversation_stage: str
    primary_objective: str
    confidence_threshold: float = 0.7

    def to_dict(self) -> Dict:
        return {
            "intent_name": self.intent_name,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "prompt": self.prompt,
            "prompt_summary": self.prompt_summary,
            "conversation_stage": self.conversation_stage,
            "primary_objective": self.primary_objective,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class PartialMatch:

    intent: str
    source_node: str
    target_node: str
    prompt_summary: str
    relevance_score: float
    addresses_aspect: str
    confidence: float
    extraction_stage: int

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "prompt_summary": self.prompt_summary,
            "relevance_score": self.relevance_score,
            "addresses_aspect": self.addresses_aspect,
            "confidence": self.confidence,
            "extraction_stage": self.extraction_stage,
        }


@dataclass
class ClassificationResult:

    stage_completed: int
    classification_type: ClassificationType
    full_match: Optional[Dict] = None
    partial_matches: List[PartialMatch] = field(default_factory=list)
    question_classification: Optional[Dict] = None
    routing_decision: Optional[Dict] = None
    processing_time: float = 0.0
    token_count: int = 0
    reasoning: str = ""

    def has_full_match(self) -> bool:
        return (
            self.classification_type == ClassificationType.FULL_MATCH
            and self.full_match is not None
        )

    def has_partials(self) -> bool:
        return len(self.partial_matches) > 0

    def get_best_partial(self) -> Optional[PartialMatch]:
        if not self.partial_matches:
            return None
        return max(self.partial_matches, key=lambda x: x.relevance_score)


@dataclass
class ConversationState:

    current_node: str
    conversation_history: List[Dict]
    collected_partials: List[PartialMatch]
    classification_cache: Dict[str, ClassificationResult]
    user_context: Dict
    turn_count: int = 0
    start_time: float = field(default_factory=time.time)
    client_name: Optional[str] = None
    tree_manager: Optional["ConversationTreeManager"] = None
    consecutive_no_matches: int = 0

    def add_to_history(self, speaker: str, message: str):
        self.conversation_history.append(
            {
                "turn": self.turn_count,
                "speaker": speaker,
                "message": message,
                "timestamp": time.time(),
                "node": self.current_node,
            }
        )
        if speaker == "user":
            self.turn_count += 1

    def get_recent_context(self, num_turns: int = 3) -> str:
        recent = (
            self.conversation_history[-num_turns * 2 :]
            if len(self.conversation_history) > num_turns * 2
            else self.conversation_history
        )
        context = []
        for exchange in recent:
            context.append(f"{exchange['speaker'].upper()}: {exchange['message']}")
        return "\n".join(context)

    def get_openai_messages(self, num_turns: int = 5) -> List[Dict]:
        if not self.conversation_history:
            return []

        recent = (
            self.conversation_history[-num_turns * 2 :]
            if len(self.conversation_history) > num_turns * 2
            else self.conversation_history
        )

        messages = []
        for exchange in recent:
            role = "user" if exchange["speaker"] == "user" else "assistant"
            messages.append({"role": role, "content": exchange["message"]})

        return messages

    def get_company_context_for_openai(self) -> str:
        if not self.tree_manager:
            return ""

        company_facts = self.tree_manager.config_manager.get_company_facts()
        target_market = self.tree_manager.config_manager.get_target_market_data()
        advantages = self.tree_manager.config_manager.get_competitive_advantages()
        conversation_settings = (
            self.tree_manager.config_manager.get_conversation_settings()
        )

        context = f"""
COMPANY CONTEXT (Use this information when relevant in your responses):

COMPANY IDENTITY:
- Company: {self.tree_manager.config_manager.get_company_name()}
- Representative: {self.tree_manager.config_manager.get_rep_name()}
- Value Proposition: {self.tree_manager.config_manager.get_value_proposition()}

COMPANY FACTS:
- Founded: {company_facts.get('founded', 'N/A')}
- Years in Business: {company_facts.get('years_in_business', 'N/A')}
- Headquarters: {company_facts.get('headquarters', 'N/A')}
- Team Size: {company_facts.get('team_size', 'N/A')}
- Clients Served: {company_facts.get('clients_served', 'N/A')}
- Implementation Time: {company_facts.get('implementation_time', 'N/A')}
- Support: {company_facts.get('support_availability', 'N/A')}
- Lead Capacity: {company_facts.get('lead_capacity', 'N/A')}

TARGET MARKET:
- Industry Focus: {target_market.get('industry_focus', 'N/A')}
- Company Size: {target_market.get('target_company_size', 'N/A')}
- Deal Size: {target_market.get('avg_deal_size', 'N/A')}
- ROI Timeline: {target_market.get('roi_timeline', 'N/A')}

COMPETITIVE ADVANTAGES:
{chr(10).join(f'- {advantage}' for advantage in advantages)}

CONVERSATION STYLE:
- Tone: {conversation_settings.get('tone', 'professional')}
- Pace: {conversation_settings.get('pace', 'prospect_matched')}
- Objection Handling: {conversation_settings.get('objection_handling_style', 'collaborative')}
- Max Length: {conversation_settings.get('max_conversation_length', 15)} exchanges

Use this context naturally in your responses when relevant. Don't recite it, but incorporate details that strengthen your points and build credibility.
"""
        return context.strip()


class ConversationTreeManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.tree_data = config_manager.to_dict()
        self.nodes = config_manager.get_nodes()
        self.intent_index = {}
        self.all_intents = []
        self.company_config = config_manager.get_company_config()
        self.build_intent_index()

    def validate_and_report(self):
        if not self.nodes:
            print("Warning: No nodes found in conversation tree")
            return False

        print(f"Loaded conversation tree with {len(self.nodes)} nodes")
        print(f"Company: {self.company_config.get('company_name', 'Not specified')}")
        print(f"Domain: {self.config_manager.get_domain()}")
        return True

    def build_intent_index(self):
        self.intent_index = {}
        self.all_intents = []

        for node_name, node_data in self.nodes.items():
            if "paths" in node_data:
                for path in node_data["paths"]:
                    intent_name = path.get("intent", "")
                    target = path.get("target", "")

                    target_node = self.nodes.get(target, {})

                    intent_context = IntentContext(
                        intent_name=intent_name,
                        source_node=node_name,
                        target_node=target,
                        prompt=target_node.get("prompt", ""),
                        prompt_summary=target_node.get("prompt_summary", ""),
                        conversation_stage=target_node.get("conversation_stage", ""),
                        primary_objective=target_node.get("primary_objective", ""),
                        confidence_threshold=path.get("confidence_threshold", 0.7),
                    )

                    self.all_intents.append(intent_context)

                    if node_name not in self.intent_index:
                        self.intent_index[node_name] = []
                    self.intent_index[node_name].append(intent_context)

    def get_node(self, node_name: str) -> Optional[Dict]:
        return self.nodes.get(node_name)

    def get_start_node(self) -> str:
        return self.config_manager.get_start_node()

    def get_current_node_intents(self, node_name: str) -> List[IntentContext]:
        return self.intent_index.get(node_name, [])

    def get_all_intents(self) -> List[IntentContext]:
        return self.all_intents

    def personalize_prompt(self, prompt: str, client_name: Optional[str] = None) -> str:
        replacements = {
            "{client_name}": client_name or "there",
            "{company_name}": self.config_manager.get_company_name(),
            "{sales_rep_name}": self.config_manager.get_rep_name(),
            "{value_proposition}": self.config_manager.get_value_proposition(),
        }

        for key, value in self.company_config.items():
            if isinstance(value, str):
                replacements[f"{{{key}}}"] = value

        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        return prompt


class BotLogger:

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.start_time = time.time()

    def log_bot(self, message: str):
        print(f"\nBot: {message}")

    def log_user(self, message: str):
        print(f"\nUser: {message}")

    def log_system(self, message: str):
        if self.debug_mode:
            print(f"\nSystem: {message}")

    def log_classification(self, result: ClassificationResult):
        if not self.debug_mode:
            return

        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS")
        print("=" * 60)

        print(f"\nStage Completed: {result.stage_completed}")
        print(f"Classification Type: {result.classification_type.value}")

        if result.has_full_match():
            print(f"\nFull Match Found:")
            print(f"   Intent: {result.full_match.get('intent', 'Unknown')}")
            print(f"   Target: {result.full_match.get('target', 'Unknown')}")
            print(f"   Confidence: {result.full_match.get('confidence', 0):.2f}")

        if result.has_partials():
            print(f"\nPartial Matches ({len(result.partial_matches)}):")
            for partial in result.partial_matches:
                print(f"   • {partial.intent} (Stage {partial.extraction_stage})")
                print(f"     Relevance: {partial.relevance_score:.2f}")
                print(f"     Addresses: {partial.addresses_aspect}")

        if result.question_classification:
            print(
                f"\nQuestion Type: {result.question_classification.get('type', 'Unknown')}"
            )

        print(f"\nProcessing Time: {result.processing_time:.2f}s")

        print("=" * 60)

    def log_routing(self, from_node: str, to_node: str, reason: str = ""):
        if self.debug_mode:
            print(f"\nRouting: {from_node} -> {to_node}")
            if reason:
                print(f"   Reason: {reason}")


def _get_fallback_node(current_node: str, tree_manager) -> Optional[str]:

    fallback_progression = {
        "OPENING_WARM_INTRODUCTION": "DISCOVERY_INITIAL_QUALIFICATION",
        "DISCOVERY_INITIAL_QUALIFICATION": "VALUE_PROPOSITION_CLARIFICATION",
        "DISCOVERY_PAIN_POINT_EXPLORATION": "VALUE_PROPOSITION_CLARIFICATION",
        "TIME_CONSTRAINT_ACKNOWLEDGE": "VALUE_PROPOSITION_CLARIFICATION",
        "OBJECTION_BUDGET_CONSTRAINTS": "VALUE_PROPOSITION_ROI_FOCUS",
        "VALUE_PROPOSITION_CLARIFICATION": "CLOSING_SOFT_MEETING_REQUEST",
        "VALUE_PROPOSITION_ROI_FOCUS": "CLOSING_SOFT_MEETING_REQUEST",
        "DEFAULT": "DISCOVERY_INITIAL_QUALIFICATION",
    }

    fallback = fallback_progression.get(current_node)
    if not fallback:
        fallback = fallback_progression["DEFAULT"]

    if fallback and tree_manager.get_node(fallback):
        return fallback

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Conversational Bot Engine - Works with any company's decision tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python llm_bot_engine.py                           # Use default sales_flow.json
  python llm_bot_engine.py -j company_flow.json      # Use specific JSON file
  python llm_bot_engine.py -c "John" --debug         # Override client name with debug mode
  python llm_bot_engine.py --json-file custom.json   # Use custom configuration
        """,
    )
    parser.add_argument(
        "-j",
        "--json-file",
        type=str,
        default="sales_flow.json",
        help="Path to company's conversation flow JSON (default: sales_flow.json)",
    )
    parser.add_argument(
        "-c", "--client-name", type=str, help="Override client name for personalization"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed classification output",
    )
    args = parser.parse_args()

    # Initialize configuration
    try:
        config_manager = ConfigManager(args.json_file, args.client_name)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please ensure the file '{args.json_file}' exists.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON in '{args.json_file}'")
        print(f"Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)

    # Display dynamic header based on configuration
    metadata = config_manager.get_metadata()
    company_name = config_manager.get_company_name()
    domain = config_manager.get_domain()

    print("\n" + "=" * 60)
    print(f"DYNAMIC CONVERSATION ENGINE v2.1.0")
    print(f"Company: {company_name}")
    print(f"Domain: {domain.upper()}")
    print(f"Configuration: {args.json_file}")
    print("=" * 60)

    # Initialize logger
    logger = BotLogger(debug_mode=args.debug)

    # Load conversation tree
    logger.log_system("Initializing conversation system...")
    tree_manager = ConversationTreeManager(config_manager)

    if not tree_manager.validate_and_report():
        print("\nError: Invalid conversation tree structure")
        sys.exit(1)

    # Import and initialize all systems
    try:
        from classification_engine import MultiStageClassifier, ResponseGenerator
        from context_manager import AdvancedContextManager
        from routing_optimizer import RoutingOptimizer
        from analytics_tracker import AnalyticsTracker

        classifier = MultiStageClassifier(tree_manager, debug_mode=args.debug)
        response_generator = ResponseGenerator(tree_manager, debug_mode=args.debug)
        context_manager = AdvancedContextManager(debug_mode=args.debug)
        routing_optimizer = RoutingOptimizer(
            tree_manager, context_manager, debug_mode=args.debug
        )
        analytics_tracker = AnalyticsTracker(debug_mode=args.debug)
        logger.log_system("All systems initialized")
    except ImportError as e:
        logger.log_system(f"Warning: Systems not available: {e}")
        classifier = None
        response_generator = None
        context_manager = None
        routing_optimizer = None
        analytics_tracker = None

    # Initialize conversation state
    state = ConversationState(
        current_node=tree_manager.get_start_node(),
        conversation_history=[],
        collected_partials=[],
        classification_cache={},
        user_context={},
        client_name=config_manager.get_client_name(),
        tree_manager=tree_manager,
    )

    logger.log_system(f"Starting at node: {state.current_node}")

    # Display initial message
    initial_node = tree_manager.get_node(state.current_node)
    if initial_node:
        initial_prompt = tree_manager.personalize_prompt(
            initial_node.get("prompt", "Hello! How can I help you today?"),
            state.client_name,
        )
        logger.log_bot(initial_prompt)
        state.add_to_history("bot", initial_prompt)

    print("\n" + "=" * 60)
    print("CONVERSATION STARTED")
    print("Type 'quit' to end the conversation")
    print("=" * 60)

    # Main conversation loop
    max_length = config_manager.get_max_conversation_length()
    while state.turn_count < max_length:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                logger.log_bot("Thank you for your time. Have a great day!")
                break

            state.add_to_history("user", user_input)

            if context_manager:
                context_manager.update_from_turn(user_input, classification_result=None)

            if classifier and response_generator:
                logger.log_system("Classifying user input...")

                classification_result = classifier.classify(user_input, state)

                if context_manager:
                    context_manager.update_from_turn(
                        user_input,
                        classification_result=(
                            classification_result.to_dict()
                            if hasattr(classification_result, "to_dict")
                            else None
                        ),
                    )

                logger.log_classification(classification_result)
                if (
                    classification_result.classification_type
                    == ClassificationType.NO_MATCH
                ):
                    state.consecutive_no_matches += 1
                    logger.log_system(
                        f"No match found. Consecutive: {state.consecutive_no_matches}"
                    )

                    if analytics_tracker:
                        analytics_tracker.track_no_match(
                            user_input,
                            state.current_node,
                            state.consecutive_no_matches,
                            (
                                context_manager.get_context_summary()
                                if context_manager
                                else {}
                            ),
                        )

                    logger.log_system("Generating no-match response...")
                    context_data = (
                        context_manager.get_context_summary()
                        if context_manager
                        else None
                    )

                    if context_data is None:
                        context_data = {}
                    context_data["consecutive_no_matches"] = (
                        state.consecutive_no_matches
                    )

                    response = response_generator.generate_response(
                        classification_result, user_input, state, context_data
                    )

                    if state.consecutive_no_matches >= 2 and (
                        "specialist" in response.lower()
                        or "connect you" in response.lower()
                    ):
                        state.consecutive_no_matches = 0
                        if analytics_tracker:
                            analytics_tracker.track_escalation(
                                "consecutive_no_matches", state.current_node
                            )
                else:
                    state.consecutive_no_matches = 0
                    logger.log_system("Generating response...")
                    context_data = (
                        context_manager.get_context_summary()
                        if context_manager
                        else None
                    )
                    response = response_generator.generate_response(
                        classification_result, user_input, state, context_data
                    )

                if classification_result.has_full_match():
                    target_node = classification_result.full_match.get("target")
                    if target_node and target_node != state.current_node:
                        logger.log_routing(
                            state.current_node, target_node, "Full match found"
                        )
                        state.current_node = target_node
                elif (
                    classification_result.classification_type
                    == ClassificationType.NO_MATCH
                ):
                    routing_target = None
                    if routing_optimizer:
                        routing_decision = routing_optimizer.optimize_routing(
                            state.current_node,
                            {"has_full_match": False, "classifications": []},
                            state,
                            user_input,
                        )

                        if (
                            routing_decision.should_route
                            and routing_decision.target_node
                        ):
                            routing_target = routing_decision.target_node
                            logger.log_routing(
                                state.current_node,
                                routing_target,
                                f"No-match routing: {routing_decision.reasoning}",
                            )

                    if not routing_target:
                        fallback_target = _get_fallback_node(
                            state.current_node, tree_manager
                        )
                        if fallback_target and fallback_target != state.current_node:
                            routing_target = fallback_target
                            logger.log_routing(
                                state.current_node,
                                routing_target,
                                "Fallback progression",
                            )

                    if routing_target:
                        state.current_node = routing_target

                if classification_result.has_partials():
                    state.collected_partials.extend(
                        classification_result.partial_matches[:3]
                    )
                    if len(state.collected_partials) > 10:
                        state.collected_partials = state.collected_partials[-10:]
                logger.log_bot(response)
                state.add_to_history("bot", response)

            else:
                logger.log_bot("I understand. Let me help you with that.")
                state.add_to_history("bot", "I understand. Let me help you with that.")
            current_node = tree_manager.get_node(state.current_node)
            if current_node and current_node.get("type") == "END":
                endpoint = current_node.get("endpoint_name", "Unknown")
                success = current_node.get("success_metrics", {})

                logger.log_system(f"Reached end node: {endpoint}")
                logger.log_system(f"Outcome: {success.get('outcome', 'completed')}")

                if endpoint not in ["END_GRACEFUL_HOSTILE", "END_TECHNICAL_DIFFICULTY"]:
                    logger.log_bot("Thank you for the conversation. Have a great day!")
                break

        except KeyboardInterrupt:
            print("\n\nConversation interrupted by user.")
            break
        except Exception as e:
            logger.log_system(f"Error: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

            logger.log_bot(
                "I apologize for the confusion. Could you please repeat that?"
            )

    print("\n" + "=" * 60)
    print("CONVERSATION SUMMARY")
    print("=" * 60)
    print(f"Total Turns: {state.turn_count}")
    print(f"Duration: {time.time() - state.start_time:.1f} seconds")
    print(f"Final Node: {state.current_node}")

    if state.collected_partials and args.debug:
        print(f"Partial Matches Collected: {len(state.collected_partials)}")
        unique_aspects = set(p.addresses_aspect for p in state.collected_partials)
        print(f"Aspects Covered: {', '.join(unique_aspects)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
