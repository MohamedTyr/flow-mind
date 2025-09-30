# Bare-Metal Conversational AI Engine

An experimental from-scratch implementation of a multi-stage intent classification and routing system. No RAG, no LangChain, no LlamaIndex - just custom logic for understanding how these systems work under the hood.
This project exists because I wanted to build a conversational AI where every step is visible and debuggable. Instead of chaining together framework abstractions, everything from intent classification to response generation is implemented manually. This makes it easy to trace exactly what the system is doing at each step, but it also means the architecture has deliberate trade-offs.

For example, the system makes separate LLM calls for classification, routing analysis, and response generation. This is slow and inefficient - a real production system would collapse these into a single prompt with function calling or use async processing. But keeping them separate makes the logic clearer and easier to modify. Same goes for the routing optimizer and context manager: they're more verbose than they need to be because I wanted each decision point to be explicit.

By implementing everything from the multi-stage intent classification to the context-aware routing manually, this project explores the first principles of building a robust, config-driven conversational engine. The result is a fully functional B2B sales bot that navigates a complex decision tree.

## System Architecture

The system enforces a strict separation of **conversational control flow** (Python engine) and **conversational content** (JSON configuration). The engine defines how dialogue is executed; the JSON defines what content is authorized and when it is emitted. Switching domains (sales, support, onboarding) is a configuration-only change; engine code is unaffected.

### Core Components

- **`llm_bot_engine.py`**: Orchestrator. Manages configuration, `ConversationState`, and the primary turn loop.
- **`classification_engine.py`**: Three-stage intent classifier and dynamic response generator.
- **`context_manager.py`**: Short-term context store. Tracks pain points, priorities, sentiment, and decision stage.
- **`routing_optimizer.py`**: Strategic router. Weighs classified intent against conversational context to select next node.
- **`analytics_tracker.py`**: Telemetry. Logs conversation metrics for evaluation.
- **`advanced_prompts.py`**: Domain-aware prompt templates guiding reasoning and response generation.
- **`config_manager.py`**: Loader and validator for the JSON decision tree.
- **`sales_flow.json`**: Conversation graph: nodes, prompts, transitions, and constraints.

---

## How It Works

A user input traverses a deterministic, multi-stage processing pipeline.

_[Mermaid Diagram: User Input → Classification → Context Update → Routing → Response Generation → Bot Output]_

#### **Step 1: Ingestion & State Update**

The `llm_bot_engine` records the input in history and updates `ConversationState` (current node plus metadata).

#### **Step 2: Multi-Stage Intent Classification**

The `MultiStageClassifier` runs a three-stage funnel to determine intent with speed and accuracy:

1.  **Stage 1 (Local Analysis):** Attempt match against the constrained `paths` of the current node only. If a high-confidence full match is found, stop.
2.  **Stage 2 (Global Search):** If no local match, expand to compare against all intents in the decision tree. Supports out-of-band topic shifts (e.g., pricing in the introduction).
3.  **Stage 3 (Conceptual Fallback):** If still unresolved, perform conceptual categorization of the query (question, confusion, off‑topic, etc.) to select a response strategy.

#### **Step 3: Contextual Analysis**

In parallel, the `AdvancedContextManager` extracts implicit signals and updates context:

- **Pain Points:** Keywords indicating inefficiency, constraints, or high costs.
- **Priorities:** Language revealing goals, timelines, and critical needs.
- **Engagement Level:** Sentiment to determine whether hostile, interested, or neutral.
- **Decision Stage:** Signals placing the user on the buyer's journey (e.g., problem‑aware vs. solution‑aware).

#### **Step 4: Strategic Routing**

Classifier output and the context summary feed the `RoutingOptimizer`. Candidate paths are scored for: progression benefit, context alignment, and strategic value. Example considerations:

- Resist moving to closing nodes when user stance is negative despite an apparent match.
- Prioritize value‑proposition nodes when multiple pain points surface over low‑value clarifications.

#### **Step 5: Dynamic Response Generation**

After a target node is selected, the `ResponseGenerator` composes the final reply. The generator does not fabricate facts; it elevates delivery only. It operates in three modes:

1.  **Full Match Generation:** Retrieve the node `prompt` and inject it into a template in `advanced_prompts.py` (e.g., `RESPONSE_GENERATION_ADVANCED`). The template directs the LLM to deliver the core message with clarity and persuasive structure while adhering strictly to source facts.

2.  **Partial Match Combination:** When input spans multiple topics, gather `prompts` from all partially matched nodes and synthesize a single coherent response that addresses each facet without contradiction.

3.  **No‑Match (Fallback) Generation:** When no intent matches, use a fallback template that acknowledges uncertainty, avoids speculation, and redirects using `AdvancedContextManager` signals to a relevant business topic.

#### **Step 6: Output & Analytics**

Emit the response. The `AnalyticsTracker` records the outcome and updates engagement, objection handling, and progression metrics.

---

## The Blueprint: The JSON Decision Tree

Engine logic is generic. The knowledge and flow are defined by a JSON file (`sales_flow.json`), which is the single source of truth for bot behavior.

_[Mermaid Diagram: Node graph with edges labeled by intents.]_

### Key Concepts:

- **`nodes`**: Conversation states (e.g., `OPENING_WARM_INTRODUCTION`, `OBJECTION_BUDGET_CONSTRAINTS`) with script and exits.
- **`prompt`**: Factually authorized message for the node. Supports placeholders like `{client_name}` for personalization.
- **`paths`**: Transition entries consisting of:
  - `intent`: Classified trigger (e.g., `AFFIRMATIVE_INTEREST`).
  - `target`: Next node name.

### Customizing for Any Use Case

Deep customization is configuration-only; the engine remains unchanged.

- **Add a New Objection:** Create a new node with a handler prompt. Add a `path` from relevant discovery nodes, define a new `intent` (e.g., `OBJECTION_ALREADY_HAVE_SOLUTION`), and set its `target`.
- **Change Persona:** Edit `company_config` (company name, rep name, value proposition).
- **Adapt for Customer Support:** Replace sales nodes with troubleshooting steps (`GATHER_ISSUE_DETAILS`), information delivery (`EXPLAIN_RESET_PROCEDURE`), and end states (`TICKET_RESOLVED`, `ESCALATE_TO_HUMAN`). Classification and routing logic are unchanged.

The JSON configuration keeps the bot's logic transparent, inspectable, and domain‑portable.

## Technical Stack

- **Language:** Python 3.9+
- **LLM Provider:** OpenAI API (`gpt-4o-mini`, `gpt-4o`)
- **Core Logic:** Custom-built classification, routing, and context management.
- **Tooling:** Dataclasses, enums, and type hints for maintainable, type-safe code.
- **Configuration:** JSON-driven decision trees.

## Setup

1.  **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

2.  **Set your OpenAI API key:**

    ```sh
    # For bash/zsh
    export OPENAI_API_KEY="sk-..."

    # For PowerShell
    $env:OPENAI_API_KEY="sk-..."
    ```

3.  **Run the engine:**
    ```sh
    python llm_bot_engine.py
    ```

### CLI Options

The engine supports several runtime flags for testing and customization:

| Flag                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| `-j`, `--json-file`   | Path to the JSON configuration file.               |
| `-c`, `--client-name` | Override the client's name for personalization.    |
| `-d`, `--debug`       | Enable verbose debug output for all system stages. |

**Example:**

```sh
python llm_bot_engine.py -j support_flow.json -c "Maria" --debug
```
