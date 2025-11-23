# AI Term Explainer Agent

## Overview
A tool-augmented language model agent that explains AI/ML terms at three difficulty levels (beginner, intermediate, expert) using Wikipedia as a knowledge source. Built using the `smolagents` library for CS 678 Assignment 3.

This agent demonstrates:
- **Custom tool implementation** with Wikipedia API integration
- **Multi-step reasoning** capabilities (up to 5 steps)
- **Human-in-the-loop** interactive feedback mechanism
- **Adaptive explanations** at three complexity levels

## Features

### 1. Custom Tools
- **`fetch_wikipedia_definition`**: Fetches Wikipedia definitions for AI/ML terms and truncates them to manageable length
- **`get_term_context`**: Retrieves related terms and categories for better contextual understanding

### 2. Multi-Step Reasoning
- Agent uses up to 5 steps to gather information and synthesize explanations
- First step: Fetches Wikipedia definition
- Subsequent steps: Processes information and adapts to different difficulty levels
- Final step: Generates comprehensive explanations

### 3. Three Difficulty Levels
- **Beginner**: Simple analogies, no technical jargon, easy to understand
- **Intermediate**: Technical details with basic mathematical concepts
- **Expert**: Deep technical explanations with mathematical formulations and implementation details

### 4. Human-in-the-Loop
- Interactive mode for user queries
- Feedback mechanism to improve explanations
- Follow-up question support for clarifications

## How to Run

### Prerequisites
- **Python 3.10 or higher** (Python 3.11+ recommended)
- At least one API key for LLM access (see API Setup below)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**For zsh users** (if you get "no matches found" error):
```bash
pip install "smolagents[toolkit,litellm]" wikipedia-api wikipedia requests python-dotenv arize-phoenix openinference-instrumentation-litellm
```

**For Python 3.11+ users**:
```bash
python3.11 -m pip install -r requirements.txt
```

### Step 2: Set Up API Keys

Create a `.env` file in the project directory with at least one API key:

```bash
# Option 1: OpenRouter (Recommended - FREE tier: $5 credits)
OPENROUTER_API_KEY=sk-or-v1-your_key_here

# Option 2: Groq (100% FREE, no credit card required)
GROQ_API_KEY=your_groq_api_key_here

# Option 3: Together AI (FREE tier available)
TOGETHER_API_KEY=your_together_api_key_here

# Option 4: Hugging Face (FREE with token)
HF_TOKEN=your_huggingface_token_here

# Option 5: OpenAI (requires paid account)
OPENAI_API_KEY=your_openai_key_here
```

**Get API Keys:**
- OpenRouter: https://openrouter.ai/keys
- Groq: https://console.groq.com/keys
- Together AI: https://api.together.xyz
- Hugging Face: https://huggingface.co/settings/tokens

The code automatically uses the first available API key in this priority order: OpenRouter → Together AI → Groq → Hugging Face → OpenAI

### Step 3: Run the Agent

**Interactive Mode** (Recommended):
```bash
python main.py
# or
python3.11 main.py
```

This starts an interactive session where you can:
1. Enter AI/ML terms to get explanations
2. Provide feedback on explanations
3. Ask follow-up questions
4. Type 'quit' to exit

**Test Mode** (Automated testing):
```bash
python test_agent.py
# or
python3.11 test_agent.py
```

This runs automated tests on 5 predefined AI/ML terms:
- cross-entropy loss
- attention mechanism
- gradient descent
- transformer architecture
- backpropagation

**Programmatic Usage**:
```python
from main import create_ai_term_explainer_agent, explain_term

# Create agent
agent = create_ai_term_explainer_agent()

# Explain a term at all difficulty levels
result = explain_term(agent, "cross-entropy loss", difficulty="all")
print(result)

# Explain at specific level
result = explain_term(agent, "neural network", difficulty="beginner")
print(result)
```

## Project Architecture

### Agent Type
- **CodeAgent** from `smolagents` library
- Generates Python code snippets to call tools dynamically
- Allows flexible multi-step reasoning

### Model Support
- **Primary**: Qwen 2.5 7B via OpenRouter (fast, capable)
- **Alternatives**: Together AI, Groq, Hugging Face, OpenAI
- Automatic fallback to next available provider

### Tools
- **2 Custom Tools** defined with `@tool` decorator
- **Max Steps**: 5 (configurable for multi-step reasoning)

## Example Usage Session

```
$ python main.py
Initializing AI Term Explainer Agent...
Agent ready!

================================================================================
AI Term Explainer - Interactive Mode
Enter an AI/ML term to get explanations at 3 difficulty levels.
Type 'quit' to exit.

Enter an AI/ML term: cross-entropy loss

================================================================================
Explaining: cross-entropy loss
================================================================================

BEGINNER LEVEL:
Cross-entropy loss is a way to measure how well a machine learning model is 
guessing the right answers. Imagine you're trying to guess a number between 1 
and 10, and the actual number is 7. If your guesses are very wrong, like 1 or 
10, the loss would be high. But if you guessed close, like 8, the loss would 
be low.

INTERMEDIATE LEVEL:
Cross-entropy loss is a loss function commonly used in classification tasks 
within machine learning. It measures the difference between the predicted 
probability distribution and the true distribution. Specifically, if we denote 
the true probabilities as p_i and the predicted probabilities as q_i, the 
cross-entropy loss is calculated as -Σ p_i log(q_i).

EXPERT LEVEL:
The cross-entropy loss for a multi-class classification problem is defined as:
L = -Σ(y_i * log(ŷ_i))
where y_i is the true label distribution and ŷ_i is the predicted distribution.
This function is differentiable and convex, making it suitable for gradient-based
optimization methods. It's closely related to Kullback-Leibler divergence and
information entropy.

Was this explanation helpful? (yes/no/skip): yes

Enter an AI/ML term: quit
Goodbye!
```

## Project Structure

```
HW3/
├── main.py                 # Main agent implementation (CodeAgent setup, explain_term, interactive_mode)
├── custom_tools.py         # Custom tool definitions (fetch_wikipedia_definition, get_term_context)
├── requirements.txt        # Python dependencies
├── test_agent.py           # Automated test script
├── README.md               # This documentation file
├── .gitignore             # Git ignore file (excludes .env)
└── .env                   # Environment variables (create this, not in repo)
```

## Key Implementation Details

### Custom Tools (`custom_tools.py`)
- **`fetch_wikipedia_definition(term: str) -> str`**
  - Uses `wikipediaapi` library to fetch page summaries
  - Truncates text to 500 characters for context management
  - Handles cases where pages don't exist with fallback search
  - Decorated with `@tool` from `smolagents`

- **`get_term_context(term: str) -> str`**
  - Retrieves Wikipedia categories and related terms
  - Provides context for better multi-step reasoning
  - Helps agent understand relationships between concepts

### Agent Configuration (`main.py`)
- **`create_ai_term_explainer_agent()`**
  - Initializes `CodeAgent` with custom tools
  - Configures LLM model (supports multiple providers)
  - Sets `max_steps=5` for multi-step reasoning

- **`explain_term(agent, term, difficulty)`**
  - Main function for generating explanations
  - Supports "beginner", "intermediate", "expert", or "all" levels
  - Uses agent's multi-step reasoning capabilities

- **`interactive_mode(agent)`**
  - Implements human-in-the-loop functionality
  - Prompts for user input and feedback
  - Supports follow-up questions

## Testing

Run automated tests:
```bash
python test_agent.py
```

Tests 5 AI/ML terms:
- cross-entropy loss
- attention mechanism
- gradient descent
- transformer architecture
- backpropagation

## Technical Notes

- **Wikipedia Integration**: Uses `wikipediaapi` and `wikipedia` libraries for robust page retrieval
- **Text Truncation**: Definitions limited to 500 characters to manage LLM context windows
- **Multi-Step Reasoning**: Agent can use up to 5 steps, typically:
  1. Fetch Wikipedia definition
  2. Process and adapt for beginner level
  3. Expand for intermediate level
  4. Deepen for expert level
  5. Synthesize final answer
- **Error Handling**: Tools include try-except blocks for graceful failure handling
- **API Key Security**: All API keys stored in `.env` file (excluded from git)
- **Phoenix Observability**: Optional - Phoenix UI automatically starts at http://127.0.0.1:6006 for tracing and observability. Traces show LLM calls, tool usage, and performance metrics.

## Limitations

- Wikipedia dependency: Only uses Wikipedia as knowledge source
- Text truncation: Definitions limited to 500 characters
- No caching: Repeated queries fetch same data
- Rate limits: Subject to API rate limits (Groq, OpenRouter, etc.)
- Model quality: Depends on underlying LLM capabilities

## Troubleshooting

**Issue**: "No API key found" error
- **Solution**: Create `.env` file with at least one API key (see Step 2 above)

**Issue**: "smolagents not found" error
- **Solution**: Ensure Python 3.10+ is installed and dependencies are installed

**Issue**: "zsh: no matches found" error
- **Solution**: Use quotes around package name: `pip install "smolagents[toolkit,litellm]"`

**Issue**: Wikipedia page not found
- **Solution**: Try alternative term names or check spelling

