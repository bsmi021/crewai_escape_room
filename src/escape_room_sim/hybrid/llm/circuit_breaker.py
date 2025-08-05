"""
Circuit Breaker Pattern for LLM Integration

Implements circuit breaker pattern to handle LLM failures gracefully
and prevent cascading failures in the decision engine.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
from enum import Enum

from .exceptions import CircuitOpenError, TimeoutError


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class LLMCircuitBreaker:
    """
    Circuit breaker for LLM API calls
    
    Implements the circuit breaker pattern to prevent cascading failures
    when LLM services are unreliable or unavailable.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        timeout: float = 10.0,
        half_open_max_calls: int = 2
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            timeout: Timeout for individual LLM calls
            half_open_max_calls: Max calls to test in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
    @property
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open"""
        return self.state == CircuitState.HALF_OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        return self.state == CircuitState.CLOSED
    
    def should_attempt_call(self) -> bool:
        """
        Determine if a call should be attempted
        
        Returns:
            True if call should be attempted, False if circuit is open
        """
        if self.is_closed:
            return True
        
        if self.is_half_open:
            return self.half_open_calls < self.half_open_max_calls
        
        if self.is_open:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                self._transition_to_half_open()
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record a successful call"""
        if self.is_half_open:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self._transition_to_closed()
        elif self.is_closed:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.is_half_open:
            # Failure in half-open state immediately reopens circuit
            self._transition_to_open()
        elif self.is_closed and self.failure_count >= self.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_closed(self) -> None:
        """Transition to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
    
    def _transition_to_open(self) -> None:
        """Transition to open state"""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
    
    def _transition_to_half_open(self) -> None:
        """Transition to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker
        
        Args:
            func: Async function to call
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            CircuitOpenError: If circuit is open
            TimeoutError: If call times out
            Exception: Any exception from the function call
        """
        if not self.should_attempt_call():
            raise CircuitOpenError(
                f"Circuit breaker is open (failures: {self.failure_count})",
                self.failure_count
            )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            self.record_success()
            return result
            
        except asyncio.TimeoutError:
            self.record_failure()
            raise TimeoutError(f"LLM call timed out after {self.timeout}s", self.timeout)
            
        except Exception as e:
            self.record_failure()
            raise e
    
    def get_state_info(self) -> dict:
        """Get current state information"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "half_open_calls": self.half_open_calls,
            "is_open": self.is_open,
            "is_half_open": self.is_half_open,
            "is_closed": self.is_closed
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0