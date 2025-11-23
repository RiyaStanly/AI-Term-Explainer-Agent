"""
Custom tools for the AI Term Explainer agent.
Implements DefinitionExpanderTool with Wikipedia API integration.
"""

from smolagents import tool
import wikipediaapi
from typing import Optional

# Global Wikipedia API instance
_wiki = wikipediaapi.Wikipedia(
    user_agent='AI-Term-Explainer/1.0',
    language='en'
)

# Default max length for truncation
_MAX_LENGTH = 500


@tool
def fetch_wikipedia_definition(term: str) -> str:
    """
    Fetches the Wikipedia definition for a given AI/ML term.
    Truncates the text to a manageable length for the LLM.
    
    Args:
        term: The AI/ML term to look up (e.g., "cross-entropy loss")
    
    Returns:
        Truncated Wikipedia definition text
    """
    try:
        # Search for the page
        page = _wiki.page(term)
        
        if not page.exists():
            # Try alternative search using Wikipedia search
            import wikipedia
            try:
                search_results = wikipedia.search(term, results=1)
                if search_results:
                    page = _wiki.page(search_results[0])
                else:
                    return f"Could not find Wikipedia page for '{term}'. Please try a different term."
            except Exception:
                return f"Could not find Wikipedia page for '{term}'. Please try a different term."
        
        # Get the summary (first section)
        summary = page.summary
        
        # Truncate to max_length while preserving word boundaries
        if len(summary) > _MAX_LENGTH:
            truncated = summary[:_MAX_LENGTH].rsplit(' ', 1)[0]
            summary = truncated + "..."
        
        return f"Wikipedia definition for '{term}':\n{summary}"
        
    except Exception as e:
        return f"Error fetching definition: {str(e)}"


@tool
def get_term_context(term: str) -> str:
    """
    Gets additional context about a term (related terms, categories).
    Useful for multi-step reasoning.
    
    Args:
        term: The AI/ML term to get context for
    
    Returns:
        Context information about the term
    """
    try:
        page = _wiki.page(term)
        if not page.exists():
            # Try search using Wikipedia search
            import wikipedia
            try:
                search_results = wikipedia.search(term, results=1)
                if search_results:
                    page = _wiki.page(search_results[0])
                else:
                    return f"No context found for '{term}'"
            except Exception:
                return f"No context found for '{term}'"
        
        # Get categories
        categories = list(page.categories.keys())[:5]
        categories_str = ", ".join([cat.split(':')[-1] for cat in categories])
        
        # Get links (related terms)
        links = list(page.links.keys())[:10]
        related_terms = ", ".join(links[:5])
        
        return f"Context for '{term}':\nCategories: {categories_str}\nRelated terms: {related_terms}"
        
    except Exception as e:
        return f"Error fetching context: {str(e)}"


# For backward compatibility, provide a class wrapper
class DefinitionExpanderTool:
    """
    Wrapper class for backward compatibility.
    The tools are now standalone functions.
    """
    
    def __init__(self, max_length: int = 500):
        """
        Initialize the tool (for backward compatibility).
        
        Args:
            max_length: Maximum length of text to return (for truncation)
        """
        global _MAX_LENGTH
        _MAX_LENGTH = max_length
    
    @property
    def fetch_wikipedia_definition(self):
        """Return the fetch_wikipedia_definition tool function."""
        return fetch_wikipedia_definition
    
    @property
    def get_term_context(self):
        """Return the get_term_context tool function."""
        return get_term_context

