"""
Test script for the AI Term Explainer agent.
Tests the agent with various AI/ML terms.
"""

from main import create_ai_term_explainer_agent, explain_term


def test_agent():
    """Test the agent with various AI/ML terms."""
    print("Initializing agent...")
    agent = create_ai_term_explainer_agent()
    print("Agent ready!\n")
    
    test_terms = [
        "cross-entropy loss",
        "attention mechanism",
        "gradient descent",
        "transformer architecture",
        "backpropagation"
    ]
    
    for i, term in enumerate(test_terms, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_terms)}: {term}")
        print('='*80)
        
        try:
            result = explain_term(agent, term, difficulty="all")
            print(result)
            print("\n" + "-"*80)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("-"*80)


if __name__ == "__main__":
    test_agent()

