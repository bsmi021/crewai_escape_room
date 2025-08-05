"""
Optimized LLM Client with Circuit Breaker and Performance Monitoring

Provides async LLM integration with circuit breaker pattern, timeout handling,
retry mechanisms, and performance optimization.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import hashlib
import json

from .circuit_breaker import LLMCircuitBreaker
from .exceptions import (
    RateLimitExceededError, 
    AuthenticationError, 
    ServiceUnavailableError,
    ModelNotFoundError,
    InvalidResponseError
)
from .fallback import FallbackDecisionGenerator


class OptimizedLLMClient:
    """
    Optimized LLM client with circuit breaker, caching, and performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the optimized LLM client
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 500)
        self.timeout = config.get("timeout", 10.0)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.batch_timeout = config.get("batch_timeout", 30.0)
        
        # Caching configuration
        self.enable_caching = config.get("enable_caching", False)
        self.cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
        self._response_cache: Dict[str, tuple] = {}  # (response, timestamp)
        
        # Circuit breaker configuration
        cb_config = config.get("circuit_breaker", {})
        self.circuit_breaker = LLMCircuitBreaker(
            failure_threshold=cb_config.get("failure_threshold", 5),
            recovery_timeout=cb_config.get("recovery_timeout", 30.0),
            timeout=cb_config.get("timeout", self.timeout),
            half_open_max_calls=cb_config.get("half_open_max_calls", 2)
        )
        
        # Performance metrics
        self._performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_response_time": 0.0,
            "last_reset": datetime.now()
        }
        
        # Fallback generator
        self.fallback_generator = FallbackDecisionGenerator()
        
        # Initialize LLM client (will be set up based on available APIs)
        self._llm_client = None
        self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on available APIs"""
        try:
            # Try OpenAI first
            import openai
            import os
            
            # Check if API key is available
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._llm_client = openai.AsyncOpenAI()
                self._client_type = "openai"
            else:
                # No API key available - use mock client
                self._llm_client = None
                self._client_type = "mock"
        except (ImportError, Exception):
            # Could add other providers here (Anthropic, etc.)
            self._llm_client = None
            self._client_type = "mock"
    
    async def generate_decision(self, prompt: str, agent_id: str) -> str:
        """
        Generate a single decision using LLM
        
        Args:
            prompt: The prompt to send to the LLM
            agent_id: ID of the agent making the request
            
        Returns:
            LLM response string
        """
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(prompt, agent_id)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self._performance_metrics["cache_hits"] += 1
                return cached_response
            self._performance_metrics["cache_misses"] += 1
        
        # Record call attempt
        self._performance_metrics["total_calls"] += 1
        start_time = time.time()
        
        try:
            # Use circuit breaker to make the call
            response = await self.circuit_breaker.call(self._make_llm_call, prompt, agent_id)
            
            # Record success
            response_time = time.time() - start_time
            self._performance_metrics["successful_calls"] += 1
            self._performance_metrics["total_response_time"] += response_time
            
            # Cache response if enabled
            if self.enable_caching:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            # Record failure
            self._performance_metrics["failed_calls"] += 1
            
            # Convert to appropriate exception type
            error_message = str(e).lower()
            if "rate limit" in error_message:
                raise RateLimitExceededError(str(e))
            elif "authentication" in error_message or "api key" in error_message:
                raise AuthenticationError(str(e))
            elif "service unavailable" in error_message:
                raise ServiceUnavailableError(str(e))
            elif "model" in error_message and "not found" in error_message:
                raise ModelNotFoundError(str(e), self.model)
            else:
                # For any other error, re-raise as is
                raise e
    
    async def generate_decisions_batch(self, prompts: Dict[str, str]) -> Dict[str, str]:
        """
        Generate decisions for multiple agents concurrently
        
        Args:
            prompts: Dictionary mapping agent_id to prompt
            
        Returns:
            Dictionary mapping agent_id to response
        """
        tasks = []
        for agent_id, prompt in prompts.items():
            task = self.generate_decision(prompt, agent_id)
            tasks.append((agent_id, task))
        
        # Execute all tasks concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                ),
                timeout=self.batch_timeout
            )
            
            # Process results
            responses = {}
            for i, result in enumerate(results):
                agent_id = tasks[i][0]
                if isinstance(result, Exception):
                    # Handle individual failures
                    responses[agent_id] = f"Error: {str(result)}"
                else:
                    responses[agent_id] = result
            
            return responses
            
        except asyncio.TimeoutError:
            # Handle batch timeout
            responses = {}
            for agent_id, _ in tasks:
                responses[agent_id] = "Error: Batch timeout exceeded"
            return responses
    
    async def _make_llm_call(self, prompt: str, agent_id: str) -> str:
        """
        Make the actual LLM API call with retry logic
        
        Args:
            prompt: The prompt to send
            agent_id: Agent ID for context
            
        Returns:
            LLM response string
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                if self._client_type == "openai" and self._llm_client:
                    response = await self._make_openai_call(prompt, agent_id)
                else:
                    # Mock response for testing
                    response = await self._make_mock_call(prompt, agent_id)
                
                return response
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on authentication or model errors
                error_message = str(e).lower()
                if ("authentication" in error_message or 
                    "api key" in error_message or
                    "model" in error_message):
                    raise e
                
                # Wait before retry
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    async def _make_openai_call(self, prompt: str, agent_id: str) -> str:
        """Make OpenAI API call"""
        try:
            response = await self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a {agent_id} in an escape room scenario."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise InvalidResponseError("Empty response from OpenAI")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _make_mock_call(self, prompt: str, agent_id: str) -> str:
        """Make mock LLM call for testing"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on agent type
        agent_type = agent_id.lower()
        
        if "strategist" in agent_type:
            responses = [
                "I will analyze the current situation systematically",
                "Let me assess the risks and plan our next move",
                "Based on my analysis, we should examine the puzzle mechanism"
            ]
        elif "mediator" in agent_type:
            responses = [
                "I'll coordinate with the team to ensure we work together",
                "Let me facilitate communication between all agents",
                "We need to align our strategies for maximum effectiveness"
            ]
        elif "survivor" in agent_type:
            responses = [
                "I'll search for tools that can help us escape",
                "My priority is finding the most direct escape route",
                "I need to gather resources for our survival"
            ]
        else:
            responses = [
                "I will observe the situation and plan accordingly",
                "Let me examine the environment for clues",
                "I'll take action based on current conditions"
            ]
        
        # Add some randomness to prompt context
        import random
        response = random.choice(responses)
        
        # Add context from prompt
        if "puzzle" in prompt.lower():
            response += " focusing on the puzzle elements"
        elif "team" in prompt.lower() or "coordinate" in prompt.lower():
            response += " while maintaining team coordination"
        elif "time" in prompt.lower():
            response += " considering our time constraints"
        
        return response
    
    def _generate_cache_key(self, prompt: str, agent_id: str) -> str:
        """Generate cache key for prompt and agent combination"""
        combined = f"{agent_id}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if still valid"""
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return response
            else:
                # Remove expired cache entry
                del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with timestamp"""
        self._response_cache[cache_key] = (response, datetime.now())
        
        # Clean up old cache entries periodically
        if len(self._response_cache) > 1000:  # Arbitrary limit
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = []
        
        for key, (_, timestamp) in self._response_cache.items():
            if now - timestamp >= timedelta(seconds=self.cache_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._response_cache[key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self._performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics["total_calls"] > 0:
            metrics["success_rate"] = metrics["successful_calls"] / metrics["total_calls"]
            metrics["failure_rate"] = metrics["failed_calls"] / metrics["total_calls"]
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
        
        if metrics["successful_calls"] > 0:
            metrics["average_response_time"] = metrics["total_response_time"] / metrics["successful_calls"]
        else:
            metrics["average_response_time"] = 0.0
        
        total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        if total_cache_requests > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_requests
        else:
            metrics["cache_hit_rate"] = 0.0
        
        # Add circuit breaker state
        metrics["circuit_breaker"] = self.circuit_breaker.get_state_info()
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self._performance_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_response_time": 0.0,
            "last_reset": datetime.now()
        }
        self.circuit_breaker.reset()