"""
Unit tests for LLM Circuit Breaker and Optimization

Tests implement TDD methodology for LLM integration with circuit breaker pattern,
timeout handling, and performance optimization.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import concurrent.futures
import time


class TestLLMCircuitBreaker:
    """Test circuit breaker pattern for LLM failures"""
    
    @pytest.fixture
    def circuit_breaker_config(self):
        """Standard circuit breaker configuration"""
        return {
            "failure_threshold": 3,
            "recovery_timeout": 10.0,
            "timeout": 5.0,
            "half_open_max_calls": 2
        }
    
    def test_circuit_breaker_initialization(self, circuit_breaker_config):
        """Test circuit breaker initializes in closed state"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker, CircuitState
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10.0
        assert cb.timeout == 5.0
        assert not cb.is_open
        assert not cb.is_half_open
    
    def test_circuit_breaker_closes_to_open(self, circuit_breaker_config):
        """Test circuit breaker opens after threshold failures"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker, CircuitState
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Record failures up to threshold
        for i in range(3):
            cb.record_failure()
            if i < 2:
                assert cb.state == CircuitState.CLOSED
        
        # Should open after threshold failures
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert cb.failure_count == 3
    
    def test_circuit_breaker_open_to_half_open(self, circuit_breaker_config):
        """Test circuit breaker transitions to half-open after recovery timeout"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker, CircuitState
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Force open state
        cb.failure_count = 3
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.now() - timedelta(seconds=15)  # Past recovery timeout
        
        # Check if should attempt recovery
        assert cb.should_attempt_call()
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_half_open
    
    def test_circuit_breaker_half_open_success(self, circuit_breaker_config):
        """Test circuit breaker closes after successful half-open calls"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker, CircuitState
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Set to half-open state
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 0
        
        # Record successful calls
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Still half-open
        
        cb.record_success()
        assert cb.state == CircuitState.CLOSED  # Should close after max_calls successes
        assert cb.failure_count == 0
    
    def test_circuit_breaker_half_open_failure(self, circuit_breaker_config):
        """Test circuit breaker reopens immediately on half-open failure"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker, CircuitState
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Set to half-open state
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 1
        
        # Record failure - should immediately reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_call_wrapper(self, circuit_breaker_config):
        """Test circuit breaker call wrapper functionality"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Test successful call
        async def successful_llm_call():
            await asyncio.sleep(0.1)
            return "successful response"
        
        result = await cb.call(successful_llm_call)
        assert result == "successful response"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_handling(self, circuit_breaker_config):
        """Test circuit breaker handles LLM timeouts"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker
        
        # Set short timeout for testing
        circuit_breaker_config["timeout"] = 0.1
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Create slow LLM call
        async def slow_llm_call():
            await asyncio.sleep(1.0)  # Takes longer than timeout
            return "should not reach here"
        
        # Should raise TimeoutError and record failure
        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow_llm_call)
        
        assert cb.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls_when_open(self, circuit_breaker_config):
        """Test circuit breaker prevents calls when open"""
        from src.escape_room_sim.hybrid.llm.circuit_breaker import LLMCircuitBreaker
        from src.escape_room_sim.hybrid.llm.exceptions import CircuitOpenError
        
        cb = LLMCircuitBreaker(**circuit_breaker_config)
        
        # Force open state
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.now()
        
        async def llm_call():
            return "should not be called"
        
        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await cb.call(llm_call)


class TestLLMOptimization:
    """Test LLM optimization features"""
    
    @pytest.fixture
    def llm_config(self):
        """Standard LLM configuration"""
        return {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 500,
            "timeout": 10.0,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "batch_timeout": 30.0
        }
    
    @pytest.mark.asyncio
    async def test_llm_client_initialization(self, llm_config):
        """Test LLM client initialization with configuration"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        client = OptimizedLLMClient(llm_config)
        
        assert client.model == "gpt-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 500
        assert client.timeout == 10.0
        assert client.circuit_breaker is not None
    
    @pytest.mark.asyncio
    async def test_single_llm_call_success(self, llm_config):
        """Test successful single LLM call"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Strategic analysis complete"
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            client = OptimizedLLMClient(llm_config)
            
            prompt = "Analyze the current situation"
            response = await client.generate_decision(prompt, "strategist")
            
            assert response == "Strategic analysis complete"
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_llm_calls(self, llm_config):
        """Test batch LLM call optimization"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Mock multiple responses
            responses = []
            for i, content in enumerate(["analyze", "coordinate", "search"]):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = content
                responses.append(mock_response)
            
            mock_client.chat.completions.create = AsyncMock(side_effect=responses)
            
            client = OptimizedLLMClient(llm_config)
            
            prompts = {
                "strategist": "Analyze the situation",
                "mediator": "Coordinate team efforts", 
                "survivor": "Search for tools"
            }
            
            start_time = time.time()
            results = await client.generate_decisions_batch(prompts)
            end_time = time.time()
            
            # Should complete faster than sequential calls
            assert (end_time - start_time) < 1.0
            
            assert len(results) == 3
            assert results["strategist"] == "analyze"
            assert results["mediator"] == "coordinate" 
            assert results["survivor"] == "search"
    
    @pytest.mark.asyncio
    async def test_llm_retry_mechanism(self, llm_config):
        """Test LLM retry mechanism on transient failures"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # First two calls fail, third succeeds
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    Exception("Rate limit exceeded"),
                    Exception("Service temporarily unavailable"),
                    Mock(choices=[Mock(message=Mock(content="Success on retry"))])
                ]
            )
            
            client = OptimizedLLMClient(llm_config)
            
            response = await client.generate_decision("Test prompt", "test_agent")
            
            assert response == "Success on retry"
            # Should have made 3 calls (2 failures + 1 success)
            assert mock_client.chat.completions.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_llm_response_caching(self, llm_config):
        """Test LLM response caching for identical prompts"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        llm_config["enable_caching"] = True
        llm_config["cache_ttl"] = 300  # 5 minutes
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Cached response"
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            client = OptimizedLLMClient(llm_config)
            
            prompt = "Analyze the room layout"
            
            # First call
            response1 = await client.generate_decision(prompt, "strategist")
            assert response1 == "Cached response"
            
            # Second identical call should use cache
            response2 = await client.generate_decision(prompt, "strategist")
            assert response2 == "Cached response"
            
            # Should only have made one actual LLM call
            assert mock_client.chat.completions.create.call_count == 1
    
    @pytest.mark.asyncio
    async def test_llm_performance_metrics(self, llm_config):
        """Test LLM performance metrics collection"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Performance test response"
            
            # Simulate some delay
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.1)
                return mock_response
            
            mock_client.chat.completions.create = AsyncMock(side_effect=delayed_response)
            
            client = OptimizedLLMClient(llm_config)
            
            await client.generate_decision("Test prompt", "test_agent")
            
            metrics = client.get_performance_metrics()
            
            assert "total_calls" in metrics
            assert "successful_calls" in metrics
            assert "failed_calls" in metrics
            assert "average_response_time" in metrics
            assert "cache_hit_rate" in metrics
            
            assert metrics["total_calls"] == 1
            assert metrics["successful_calls"] == 1
            assert metrics["average_response_time"] > 0


class TestLLMExceptionHandling:
    """Test comprehensive LLM exception handling"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test handling of API rate limits"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        from src.escape_room_sim.hybrid.llm.exceptions import RateLimitExceededError
        
        config = {"model": "gpt-4", "timeout": 10.0, "retry_attempts": 2}
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Simulate rate limit error
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Rate limit exceeded")
            )
            
            client = OptimizedLLMClient(config)
            
            with pytest.raises(RateLimitExceededError):
                await client.generate_decision("Test prompt", "test_agent")
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test handling of authentication errors"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        from src.escape_room_sim.hybrid.llm.exceptions import AuthenticationError
        
        config = {"model": "gpt-4", "timeout": 10.0}
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Simulate auth error
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Invalid API key")
            )
            
            client = OptimizedLLMClient(config)
            
            with pytest.raises(AuthenticationError):
                await client.generate_decision("Test prompt", "test_agent")
    
    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self):
        """Test handling of service unavailable errors"""
        from src.escape_room_sim.hybrid.llm.client import OptimizedLLMClient
        from src.escape_room_sim.hybrid.llm.exceptions import ServiceUnavailableError
        
        config = {"model": "gpt-4", "timeout": 10.0, "retry_attempts": 1}
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Simulate service unavailable
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Service temporarily unavailable")
            )
            
            client = OptimizedLLMClient(config)
            
            with pytest.raises(ServiceUnavailableError):
                await client.generate_decision("Test prompt", "test_agent")


class TestLLMFallbackSystem:
    """Test LLM fallback system when primary LLM fails"""
    
    @pytest.mark.asyncio
    async def test_fallback_decision_generation(self):
        """Test fallback decision generation when LLM is unavailable"""
        from src.escape_room_sim.hybrid.llm.fallback import FallbackDecisionGenerator
        from src.escape_room_sim.hybrid.core_architecture import PerceptionData
        
        fallback = FallbackDecisionGenerator()
        
        perception = PerceptionData(
            agent_id="strategist",
            timestamp=datetime.now(),
            spatial_data={"current_position": (2, 3)},
            environmental_state={"lighting": "dim"},
            nearby_agents=["mediator"],
            available_actions=["move", "examine", "analyze", "communicate"],
            resources={"energy": 0.8},
            constraints={"action_points": 3}
        )
        
        decision = fallback.generate_fallback_decision(perception)
        
        assert decision.agent_id == "strategist"
        assert decision.chosen_action in perception.available_actions
        assert decision.confidence_level < 0.6  # Lower confidence for fallbacks
        assert decision.reasoning.startswith("Fallback decision")
        assert len(decision.fallback_actions) > 0
    
    @pytest.mark.asyncio
    async def test_rule_based_action_selection(self):
        """Test rule-based action selection for different agent types"""
        from src.escape_room_sim.hybrid.llm.fallback import FallbackDecisionGenerator
        from src.escape_room_sim.hybrid.core_architecture import PerceptionData
        
        fallback = FallbackDecisionGenerator()
        
        # Test strategist fallback
        strategist_perception = PerceptionData(
            agent_id="strategist",
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=["move", "examine", "analyze"],
            resources={},
            constraints={}
        )
        
        strategist_decision = fallback.generate_fallback_decision(strategist_perception)
        # Strategist should prefer analytical actions
        assert strategist_decision.chosen_action in ["analyze", "examine"]
        
        # Test survivor fallback
        survivor_perception = PerceptionData(
            agent_id="survivor",
            timestamp=datetime.now(),
            spatial_data={},
            environmental_state={},
            nearby_agents=[],
            available_actions=["move", "use_tool", "survive"],
            resources={},
            constraints={}
        )
        
        survivor_decision = fallback.generate_fallback_decision(survivor_perception)
        # Survivor should prefer survival actions
        assert survivor_decision.chosen_action in ["survive", "use_tool", "move"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])