"""
AI Term Explainer Agent
Main implementation of the tool-augmented agent for explaining AI/ML terms.
"""

from smolagents import CodeAgent, LiteLLMModel
from custom_tools import fetch_wikipedia_definition, get_term_context
import os
from dotenv import load_dotenv

# Phoenix integration for observability
try:
    import phoenix as px
    from phoenix.otel import register
    import webbrowser
    import time
    
    # Launch Phoenix app
    print("\n" + "="*80)
    print("🚀 Starting Phoenix for observability...")
    print("="*80)
    
    # Use localhost to avoid IPv6 binding issues
    session = px.launch_app(host="127.0.0.1", port=6006)
    
    # Get Phoenix URL
    phoenix_url = "http://127.0.0.1:6006"
    if hasattr(session, 'url'):
        phoenix_url = session.url
    elif hasattr(session, 'endpoint'):
        phoenix_url = session.endpoint
    
    # Wait for Phoenix to fully initialize
    time.sleep(2)
    
    # Register Phoenix tracer provider to capture traces
    print("\n📡 Registering Phoenix tracer provider...")
    tracer_provider = register(
        project_name="ai_term_explainer",
        endpoint=f"{phoenix_url}/v1/traces",
    )
    print("✅ Tracer provider registered!")
    
    # Configure LiteLLM to use OpenTelemetry tracing
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        # Instrument LiteLLM to send traces to Phoenix
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        print("✅ LiteLLM instrumented with OpenTelemetry!")
    except ImportError:
        # Fallback: use environment variables
        import litellm
        os.environ["OTEL_SERVICE_NAME"] = "ai_term_explainer"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{phoenix_url}/v1/traces"
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
        print("✅ OpenTelemetry environment configured (using env vars)")
    
    print(f"\n✅ Phoenix is running!")
    print(f"📊 Phoenix UI URL: {phoenix_url}")
    print(f"\n💡 To view traces:")
    print(f"   1. Open your web browser")
    print(f"   2. Go to: {phoenix_url}")
    print(f"   3. Traces will appear as you use the agent\n")
    
    # Try to open browser automatically
    try:
        webbrowser.open(phoenix_url)
        print(f"🌐 Attempted to open browser automatically")
        print(f"   If it didn't open, manually visit: {phoenix_url}\n")
    except Exception as browser_error:
        print(f"⚠️  Could not open browser automatically")
        print(f"   Please manually open: {phoenix_url}\n")
    
    PHOENIX_ENABLED = True
    PHOENIX_SESSION = session
except Exception as e:
    print(f"⚠️  Phoenix not available: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing without Phoenix observability...\n")
    PHOENIX_ENABLED = False
    PHOENIX_SESSION = None
    tracer_provider = None

load_dotenv()


def create_ai_term_explainer_agent():
    """
    Creates the AI Term Explainer agent with custom tools.
    
    Returns:
        CodeAgent instance configured with custom tools
    """
    # Create tools list (using standalone functions)
    tools = [
        fetch_wikipedia_definition,
        get_term_context,
    ]
    
    # Initialize model - Try free APIs first
    # Option 1: Qwen 2.5 via OpenRouter (FREE tier: $5 credits)
    if os.getenv("OPENROUTER_API_KEY"):
        # Configure LiteLLM to use Phoenix tracing if available
        model_kwargs = {
            "model": "openrouter/qwen/qwen-2.5-7b-instruct",  # Qwen 2.5 7B - fast and capable
            "api_key": os.getenv("OPENROUTER_API_KEY")
        }
        if PHOENIX_ENABLED:
            # Enable tracing for LiteLLM
            model_kwargs["callbacks"] = []  # LiteLLM will use OpenTelemetry automatically
        model = LiteLLMModel(**model_kwargs)
    # Option 2: Qwen 2.5 via Together AI (FREE tier available)
    elif os.getenv("TOGETHER_API_KEY"):
        model = LiteLLMModel(
            model="Qwen/Qwen2.5-7B-Instruct",  # Qwen 2.5 7B on Together AI
            api_key=os.getenv("TOGETHER_API_KEY")
        )
    # Option 3: Groq (FREE, fast, generous limits)
    elif os.getenv("GROQ_API_KEY"):
        model = LiteLLMModel(
            model="groq/llama-3.1-8b-instant",  # Updated: llama-3.1-70b was decommissioned, using 8b-instant
            api_key=os.getenv("GROQ_API_KEY")
        )
    # Option 4: Hugging Face Inference API (FREE with token)
    elif os.getenv("HF_TOKEN"):
        from smolagents import InferenceClientModel
        model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
    # Option 5: OpenAI (fallback if other keys not available)
    elif os.getenv("OPENAI_API_KEY"):
        model = LiteLLMModel(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(
            "No API key found! Please set one of: OPENROUTER_API_KEY, TOGETHER_API_KEY, "
            "GROQ_API_KEY, HF_TOKEN, or OPENAI_API_KEY in your .env file"
        )
    
    # Create agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=5,  # Allow multi-step reasoning
    )
    
    return agent


def explain_term(agent, term: str, difficulty: str = "all"):
    """
    Main function to explain an AI term at specified difficulty levels.
    
    Args:
        agent: The CodeAgent instance
        term: The AI/ML term to explain
        difficulty: "beginner", "intermediate", "expert", or "all"
    
    Returns:
        Explanation string
    """
    if difficulty == "all":
        prompt = f"""
        Explain the AI/ML term "{term}" at three difficulty levels:
        
        1. BEGINNER level: Explain it as if the reader has no prior knowledge of AI/ML.
           Use simple analogies and avoid technical jargon.
        
        2. INTERMEDIATE level: Explain it for someone with basic AI/ML knowledge.
           Include technical details and mathematical concepts where relevant.
        
        3. EXPERT level: Provide a deep, technical explanation suitable for researchers.
           Include mathematical formulations, implementation details, and connections to related concepts.
        
        First, use the fetch_wikipedia_definition tool to get the base definition.
        Then, use your knowledge to expand and adapt it to each difficulty level.
        
        CRITICAL INSTRUCTIONS:
        - Create all THREE explanations (beginner, intermediate, expert)
        - Format them as a single combined response with clear section headers:
          
          BEGINNER LEVEL:
          [your beginner explanation here]
          
          INTERMEDIATE LEVEL:
          [your intermediate explanation here]
          
          EXPERT LEVEL:
          [your expert explanation here]
        
        - Call final_answer() EXACTLY ONCE with the complete formatted response containing all three levels
        - Do NOT call final_answer() multiple times - combine everything into one answer
        """
    else:
        level_instructions = {
            "beginner": "Simple analogies, no jargon, explain as if to a complete beginner",
            "intermediate": "Technical details, basic math, assume basic AI/ML knowledge",
            "expert": "Deep technical, mathematical formulations, implementation details"
        }
        
        prompt = f"""
        Explain the AI/ML term "{term}" at the {difficulty.upper()} level.
        
        First, use the fetch_wikipedia_definition tool to get the base definition.
        Then, adapt it to the {difficulty} level: {level_instructions.get(difficulty, "")}
        
        Provide a clear, structured explanation.
        """
    
    result = agent.run(prompt)
    return result


def interactive_mode(agent):
    """
    Interactive mode where users can query terms and provide feedback.
    Implements human-in-the-loop functionality.
    """
    print("=" * 80)
    print("AI Term Explainer - Interactive Mode")
    print("Enter an AI/ML term to get explanations at 3 difficulty levels.")
    print("Type 'quit' to exit.\n")
    
    while True:
        term = input("\nEnter an AI/ML term: ").strip()
        
        if term.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not term:
            continue
        
        print(f"\n{'='*80}")
        print(f"Explaining: {term}")
        print('='*80)
        
        try:
            result = explain_term(agent, term, difficulty="all")
            print(f"\n{result}\n")
            
            # Human-in-the-loop: Ask for feedback
            feedback = input("Was this explanation helpful? (yes/no/skip): ").strip().lower()
            if feedback == 'no':
                clarification = input("What would you like to know more about? ")
                if clarification:
                    follow_up = f"Provide more details about: {clarification} in the context of {term}"
                    result = agent.run(follow_up)
                    print(f"\n{result}\n")
        
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different term.")


if __name__ == "__main__":
    try:
        # Create agent
        print("Initializing AI Term Explainer Agent...")
        agent = create_ai_term_explainer_agent()
        print("Agent ready!\n")
        
        # Run interactive mode
        interactive_mode(agent)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if PHOENIX_ENABLED:
            try:
                px.close_app()
            except:
                pass
    except Exception as e:
        print(f"\nError: {e}")
        if PHOENIX_ENABLED:
            try:
                px.close_app()
            except:
                pass
        raise

