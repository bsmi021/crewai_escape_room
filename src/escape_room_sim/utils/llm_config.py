"""
LLM Configuration utilities for Google Gemini integration.

This module provides utilities for configuring and creating Google Gemini LLM instances
for use with CrewAI agents in the escape room simulation.
"""

import os
from typing import Optional, Any
from crewai import LLM


def create_gemini_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 0.9,
    api_key: Optional[str] = None,
    **kwargs: Any
) -> LLM:
    """
    Create a Google Gemini LLM instance for CrewAI agents.
    
    Args:
        model_name: Gemini model name (e.g., 'gemini/gemini-2.5-flash-lite')
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        top_p: Top-p sampling parameter
        api_key: Google API key (if not in environment)
        **kwargs: Additional parameters for LLM
        
    Returns:
        Configured CrewAI LLM instance
        
    Raises:
        ValueError: If API key is not provided or found in environment
    """
    
    # Get API key from parameter or environment
    GEMINI_API_KEY = api_key or os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError(
            "Google API key not found. Please set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Get model name from parameter or environment and ensure proper format
    model = model_name or os.getenv("MODEL", "gemini/gemini-2.5-flash-lite")
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"
    
    # Create CrewAI LLM instance
    return LLM(
        provider="google",  # Explicitly specify the provider
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=GEMINI_API_KEY,
        max_tokens=max_tokens,
        **kwargs
    )


def get_default_gemini_llm() -> LLM:
    """
    Get the default Gemini LLM configuration for escape room simulation.
    
    Returns:
        LLM configured with appropriate settings for agent conversations
    """
    return create_gemini_llm(
        temperature=0.8,  # Slightly creative for personality expression
        top_p=0.9,        # Good balance of coherence and creativity
        max_tokens=4096   # Sufficient for detailed responses
    )


def get_strategic_gemini_llm() -> LLM:
    """
    Get Gemini LLM configured for strategic analysis (Strategist agent).
    
    Returns:
        LLM optimized for analytical and logical reasoning
    """
    return create_gemini_llm(
        temperature=0.5,  # Lower temperature for more focused analysis
        top_p=0.8,        # Balanced but focused sampling
        max_tokens=3072   # Sufficient for detailed strategic analysis
    )


def get_diplomatic_gemini_llm() -> LLM:
    """
    Get Gemini LLM configured for diplomatic communication (Mediator agent).
    
    Returns:
        LLM optimized for empathetic and collaborative responses
    """
    return create_gemini_llm(
        temperature=0.9,  # Higher temperature for empathetic responses
        top_p=0.95,       # More diverse sampling for social nuance
        max_tokens=3072   # Sufficient for relationship-focused responses
    )


def get_pragmatic_gemini_llm() -> LLM:
    """
    Get Gemini LLM configured for pragmatic decision-making (Survivor agent).
    
    Returns:
        LLM optimized for practical and action-oriented responses
    """
    return create_gemini_llm(
        temperature=0.6,  # Moderate temperature for practical creativity
        top_p=0.85,       # Focused but adaptable sampling
        max_tokens=2048   # Concise but comprehensive responses
    )


def validate_gemini_configuration() -> tuple[bool, str]:
    """
    Validate that Gemini LLM configuration is properly set up.
    
    Returns:
        Tuple of (is_valid, message) indicating configuration status
    """
    try:
        # Check if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False, "GEMINI_API_KEY environment variable not set"
        
        # Check if model is specified
        model = os.getenv("MODEL")
        if not model:
            return False, "MODEL environment variable not set"
        
    
        # Try to create a test LLM instance
        test_llm = create_gemini_llm()
        
        return True, f"Google Gemini configuration valid - Model: {model}"
        
    except ImportError:
        return False, "langchain-google-genai package not installed"
    except Exception as e:
        return False, f"Configuration error: {str(e)}"


class GeminiModelConfig:
    """Configuration constants for different Gemini models."""
    
    # Available Gemini models
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    
    # Recommended configurations for different agent types
    STRATEGIST_CONFIG = {
        "temperature": 0.5,
        "top_p": 0.8,
        "max_tokens": 3072
    }
    
    MEDIATOR_CONFIG = {
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 3072
    }
    
    SURVIVOR_CONFIG = {
        "temperature": 0.6,
        "top_p": 0.85,
        "max_tokens": 2048
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a specific Gemini model."""
        model_info = {
            cls.GEMINI_2_5_FLASH_LITE: {
                "name": "Gemini 2.5 Flash Lite",
                "description": "Lightweight, fast model for quick responses",
                "max_tokens": 8192,
                "best_for": "Quick interactions, cost-efficient"
            },
            cls.GEMINI_2_5_FLASH: {
                "name": "Gemini 2.5 Flash",
                "description": "Balanced performance and speed",
                "max_tokens": 32768,
                "best_for": "General-purpose agent conversations"
            },
            cls.GEMINI_1_5_PRO: {
                "name": "Gemini 1.5 Pro",
                "description": "High-quality reasoning and analysis",
                "max_tokens": 2097152,
                "best_for": "Complex strategic analysis"
            },
            cls.GEMINI_1_5_FLASH: {
                "name": "Gemini 1.5 Flash",
                "description": "Fast model with good performance",
                "max_tokens": 1048576,
                "best_for": "Real-time conversations"
            }
        }
        
        return model_info.get(model_name, {
            "name": model_name,
            "description": "Custom model configuration",
            "max_tokens": 4096,
            "best_for": "General use"
        })