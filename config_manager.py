"""
Configuration Manager for Dynamic Sales Bot Engine
Handles all configuration and ensures complete separation of company-specific and general logic
"""

from typing import Dict, Any, Optional, List
import json
import os


class ConfigManager:
    """Centralized configuration management for dynamic bot operation"""

    def __init__(self, json_path: str, override_client_name: Optional[str] = None):
        """
        Initialize configuration from JSON file

        Args:
            json_path: Path to company's conversation flow JSON
            override_client_name: Optional override for client name
        """
        self.json_path = json_path
        self.override_client_name = override_client_name
        self.config = {}
        self.load_configuration()

    def load_configuration(self):
        """Load and validate configuration from JSON"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Configuration file not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Validate required fields
        if "nodes" not in self.config:
            raise ValueError("JSON must contain 'nodes' field with conversation tree")

        # Set defaults for optional fields
        self._set_defaults()

    def _set_defaults(self):
        """Set default values for optional configuration fields"""

        # System defaults (non-company specific)
        if "system_config" not in self.config:
            self.config["system_config"] = {}

        system_defaults = {
            "default_start_node": self._detect_start_node(),
            "fallback_company_name": "our company",
            "fallback_rep_name": "your representative",
            "max_conversation_length": 15,
            "timeout_seconds": 5.0,
            "prompt_style": "professional",
        }

        for key, value in system_defaults.items():
            if key not in self.config["system_config"]:
                self.config["system_config"][key] = value

        # Company config defaults
        if "company_config" not in self.config:
            self.config["company_config"] = {}

        if "company_name" not in self.config["company_config"]:
            self.config["company_config"]["company_name"] = self.config[
                "system_config"
            ]["fallback_company_name"]

        if "sales_rep_name" not in self.config["company_config"]:
            # Try common field names
            rep_name = (
                self.config["company_config"].get("rep_name")
                or self.config["company_config"].get("agent_name")
                or self.config["company_config"].get("representative_name")
                or self.config["system_config"]["fallback_rep_name"]
            )
            self.config["company_config"]["sales_rep_name"] = rep_name

        # Metadata defaults
        if "metadata" not in self.config:
            self.config["metadata"] = {}

        metadata_defaults = {
            "domain": "sales",  # sales, support, survey
            "conversation_type": "general",
            "version": "1.0.0",
        }

        for key, value in metadata_defaults.items():
            if key not in self.config["metadata"]:
                self.config["metadata"][key] = value

        # Prompt configuration
        if "prompt_config" not in self.config:
            self.config["prompt_config"] = {}

        prompt_defaults = {
            "domain_context": self._infer_domain_context(),
            "industry": "general business",
            "conversation_goals": ["engage", "qualify", "advance"],
            "language_style": "professional",
            "formality_level": "medium",
        }

        for key, value in prompt_defaults.items():
            if key not in self.config["prompt_config"]:
                self.config["prompt_config"][key] = value

    def _detect_start_node(self) -> str:
        """Detect the start node from JSON structure"""
        # Check for explicit start_node field
        if "start_node" in self.config:
            return self.config["start_node"]

        # Check for common start node names
        nodes = self.config.get("nodes", {})
        common_starts = [
            "START",
            "OPENING",
            "OPENING_WARM_INTRODUCTION",
            "INTRODUCTION",
            "GREETING",
            "BEGIN",
            "INITIAL",
        ]

        for node_name in common_starts:
            if node_name in nodes:
                return node_name

        # Return first node if exists
        if nodes:
            return next(iter(nodes))

        return "START"

    def _infer_domain_context(self) -> str:
        """Infer domain context from configuration"""
        domain = self.config.get("metadata", {}).get("domain", "sales")
        conversation_type = self.config.get("metadata", {}).get("conversation_type", "")

        domain_contexts = {
            "sales": "B2B sales and lead qualification",
            "support": "customer service and issue resolution",
            "survey": "information gathering and feedback collection",
            "onboarding": "new customer setup and training",
            "retention": "customer retention and satisfaction",
        }

        if conversation_type:
            return conversation_type.replace("_", " ").title()

        return domain_contexts.get(domain, "general business conversation")

    def get_company_name(self) -> str:
        """Get company name from config"""
        return self.config["company_config"]["company_name"]

    def get_rep_name(self) -> str:
        """Get representative name from config"""
        return self.config["company_config"]["sales_rep_name"]

    def get_client_name(self) -> Optional[str]:
        """Get client name (with override support)"""
        return self.override_client_name

    def get_start_node(self) -> str:
        """Get starting node name"""
        return self.config.get(
            "start_node", self.config["system_config"]["default_start_node"]
        )

    def get_nodes(self) -> Dict:
        """Get conversation nodes"""
        return self.config.get("nodes", {})

    def get_company_config(self) -> Dict:
        """Get full company configuration"""
        return self.config.get("company_config", {})

    def get_prompt_config(self) -> Dict:
        """Get prompt configuration for dynamic prompt generation"""
        return self.config.get("prompt_config", {})

    def get_metadata(self) -> Dict:
        """Get metadata about the conversation flow"""
        return self.config.get("metadata", {})

    def get_system_config(self) -> Dict:
        """Get system configuration"""
        return self.config.get("system_config", {})

    def get_domain(self) -> str:
        """Get conversation domain (sales, support, etc.)"""
        return self.config["metadata"]["domain"]

    def get_max_conversation_length(self) -> int:
        """Get maximum conversation length"""
        return self.config["system_config"].get("max_conversation_length", 15)

    def get_timeout_seconds(self) -> float:
        """Get API timeout in seconds"""
        return self.config["system_config"].get("timeout_seconds", 5.0)

    def get_value_proposition(self) -> str:
        """Get company value proposition"""
        return self.config["company_config"].get(
            "value_proposition",
            f"help businesses with {self.get_prompt_config()['domain_context']}",
        )

    def get_company_facts(self) -> Dict:
        """Get company facts"""
        return self.config["company_config"].get("company_facts", {})

    def get_target_market_data(self) -> Dict:
        """Get target market data"""
        return {
            "industry_focus": self.config["company_config"].get("industry_focus", ""),
            "target_company_size": self.config["company_config"].get(
                "target_company_size", ""
            ),
            "avg_deal_size": self.config["company_config"].get("avg_deal_size", ""),
            "roi_timeline": self.config["company_config"].get("roi_timeline", ""),
        }

    def get_competitive_advantages(self) -> List[str]:
        """Get competitive advantages"""
        return self.config["company_config"].get("competitive_advantages", [])

    def get_conversation_settings(self) -> Dict:
        """Get conversation settings"""
        return self.config.get("conversation_settings", {})

    def to_dict(self) -> Dict:
        """Return full configuration as dictionary"""
        return self.config

    def validate_structure(self) -> bool:
        """Validate that JSON has required structure for bot operation"""
        required = ["nodes"]
        for field in required:
            if field not in self.config:
                return False

        # Validate nodes have required fields
        for node_name, node_data in self.config["nodes"].items():
            if not isinstance(node_data, dict):
                return False

            # Check for either prompt or type (for END nodes)
            if "type" not in node_data and "prompt" not in node_data:
                print(f"Warning: Node {node_name} missing prompt or type")

        return True

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration with overrides applied"""
        config_copy = self.config.copy()

        # Apply client name override if provided
        if self.override_client_name:
            config_copy["runtime"] = {"client_name": self.override_client_name}

        return config_copy
