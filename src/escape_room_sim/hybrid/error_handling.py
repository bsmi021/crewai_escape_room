"""
Mesa-CrewAI Hybrid Error Handling and Fault Tolerance System

This module implements comprehensive error handling strategies for the hybrid
architecture, ensuring graceful degradation and recovery from failures in
either Mesa or CrewAI components.

Key Features:
- Circuit breaker pattern for LLM failures
- Graceful degradation strategies
- Automatic retry with exponential backoff  
- Error classification and recovery routing
- Comprehensive logging and monitoring
"""

from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import traceback
import asyncio
import logging
import time
import random
from contextlib import asynccontextmanager


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"              # Non-critical, system continues normally
    MEDIUM = "medium"        # Some degradation, fallback mechanisms activated
    HIGH = "high"           # Significant impact, major fallbacks required
    CRITICAL = "critical"   # System stability threatened, emergency protocols


class ErrorCategory(Enum):
    """Categories of errors that can occur"""
    LLM_FAILURE = "llm_failure"                    # LLM API failures
    MESA_FAILURE = "mesa_failure"                  # Mesa simulation errors
    CREWAI_FAILURE = "crewai_failure"             # CrewAI framework errors
    STATE_SYNC_FAILURE = "state_sync_failure"     # State synchronization errors
    NETWORK_FAILURE = "network_failure"           # Network connectivity issues
    VALIDATION_FAILURE = "validation_failure"     # Data validation errors
    RESOURCE_EXHAUSTION = "resource_exhaustion"   # Memory/CPU/disk issues
    TIMEOUT_FAILURE = "timeout_failure"           # Operation timeouts
    CONFIGURATION_ERROR = "configuration_error"   # Invalid configuration
    UNKNOWN_ERROR = "unknown_error"               # Unclassified errors


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"             # Use fallback mechanism
    DEGRADE = "degrade"               # Graceful degradation
    RESET = "reset"                   # Reset component state
    ESCALATE = "escalate"             # Escalate to higher level
    ABORT = "abort"                   # Abort operation safely


@dataclass
class ErrorContext:
    """Context information for an error"""
    error_id: str
    timestamp: datetime
    component: str
    operation: str
    error_type: Type[Exception]
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    fallback_data: Any = None


class IErrorHandler(ABC):
    """Interface for error handling strategies"""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the error"""
        pass
    
    @abstractmethod
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle the error and attempt recovery"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get handler priority (lower = higher priority)"""
        pass


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5           # Failures before opening
    timeout_duration: int = 60           # Seconds before trying again
    success_threshold: int = 3           # Successes before closing
    monitoring_window: int = 300         # Window for failure tracking


class CircuitBreaker:
    """
    Circuit breaker implementation for LLM and external service calls
    
    Architecture Decision: Prevent cascade failures
    - Fails fast when service is down
    - Provides fallback mechanisms
    - Automatically recovers when service is restored
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Monitoring
        self.failure_history: List[datetime] = []
        self.call_count = 0
        self.success_rate = 1.0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        self.call_count += 1
        
        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self) -> None:
        """Record successful call"""
        self.success_count += 1
        self.last_success_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        
        self._update_success_rate()
    
    def _record_failure(self) -> None:
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.failure_history.append(datetime.now())
        
        # Clean old failures outside monitoring window
        cutoff_time = datetime.now() - timedelta(seconds=self.config.monitoring_window)
        self.failure_history = [f for f in self.failure_history if f > cutoff_time]
        
        # Check if should open circuit
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
        
        self._update_success_rate()
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_duration
    
    def _update_success_rate(self) -> None:
        """Update success rate calculation"""
        if self.call_count == 0:
            self.success_rate = 1.0
        else:
            recent_failures = len(self.failure_history)
            recent_calls = min(self.call_count, 100)  # Consider last 100 calls
            self.success_rate = max(0.0, 1.0 - (recent_failures / recent_calls))
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "call_count": self.call_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None
        }


class RetryConfig:
    """Configuration for retry mechanisms"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay += random.uniform(0, delay * 0.1)
        
        return delay


class RetryableOperation:
    """Wrapper for operations that can be retried"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts:
                    delay = self.config.get_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Max attempts reached
                    break
        
        # If we get here, all attempts failed
        raise MaxRetriesExceededError(f"Max retries ({self.config.max_attempts}) exceeded") from last_exception


class HybridErrorManager:
    """
    Central error management system for the hybrid architecture
    
    Architecture Decision: Centralized error handling with specialized handlers
    - Single point for error classification and routing
    - Pluggable error handlers for different error types
    - Comprehensive monitoring and reporting
    - Graceful degradation strategies
    """
    
    def __init__(self):
        self.error_handlers: List[IErrorHandler] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.max_error_history = 1000
        
        # Error statistics
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_counts: Dict[RecoveryStrategy, int] = {}
        
        # Degradation state
        self.degradation_level = 0  # 0 = normal, 1-3 = increasing degradation
        self.fallback_mode = False
        
        # Logging
        self.logger = logging.getLogger("hybrid_error_manager")
        self._setup_logging()
    
    def register_error_handler(self, handler: IErrorHandler) -> None:
        """Register an error handler"""
        self.error_handlers.append(handler)
        # Sort by priority
        self.error_handlers.sort(key=lambda h: h.get_priority())
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> RecoveryResult:
        """Handle an error with appropriate recovery strategy"""
        
        # Create error context
        error_context = self._create_error_context(error, context or {})
        
        # Log error
        self._log_error(error_context)
        
        # Update statistics
        self._update_error_statistics(error_context)
        
        # Store in history
        self._store_error_in_history(error_context)
        
        # Find appropriate handler
        handler = self._find_error_handler(error_context)
        
        if handler:
            try:
                recovery_result = await handler.handle_error(error_context)
                self._update_recovery_statistics(recovery_result)
                
                if recovery_result.success:
                    self.logger.info(f"Successfully recovered from error {error_context.error_id} using {recovery_result.strategy_used.value}")
                else:
                    self.logger.warning(f"Recovery attempt failed for error {error_context.error_id}")
                
                return recovery_result
                
            except Exception as recovery_error:
                self.logger.error(f"Error handler failed: {recovery_error}")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.ESCALATE,
                    message=f"Error handler failed: {recovery_error}"
                )
        else:
            # No handler found
            self.logger.error(f"No handler found for error {error_context.error_id}")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                message="No appropriate error handler found"
            )
    
    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create error context from exception and context data"""
        error_id = f"error_{int(time.time() * 1000000)}"
        
        # Classify error
        category = self._classify_error(error)
        severity = self._assess_severity(error, category)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            component=context.get("component", "unknown"),
            operation=context.get("operation", "unknown"),
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            context_data=context
        )
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category"""
        error_type = type(error)
        
        # Network-related errors
        if "network" in str(error).lower() or "connection" in str(error).lower():
            return ErrorCategory.NETWORK_FAILURE
        
        # Timeout errors
        if "timeout" in str(error).lower() or isinstance(error, asyncio.TimeoutError):
            return ErrorCategory.TIMEOUT_FAILURE
        
        # Memory/resource errors
        if isinstance(error, MemoryError) or "memory" in str(error).lower():
            return ErrorCategory.RESOURCE_EXHAUSTION
        
        # Validation errors
        if "validation" in str(error).lower() or isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION_FAILURE
        
        # Configuration errors
        if "config" in str(error).lower() or "configuration" in str(error).lower():
            return ErrorCategory.CONFIGURATION_ERROR
        
        # LLM/API errors
        if any(keyword in str(error).lower() for keyword in ["api", "llm", "openai", "anthropic", "gemini"]):
            return ErrorCategory.LLM_FAILURE
        
        # Mesa errors
        if "mesa" in str(error).lower():
            return ErrorCategory.MESA_FAILURE
        
        # CrewAI errors
        if "crew" in str(error).lower() or "crewai" in str(error).lower():
            return ErrorCategory.CREWAI_FAILURE
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity"""
        # Critical errors that threaten system stability
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors that significantly impact functionality
        if category in [ErrorCategory.MESA_FAILURE, ErrorCategory.STATE_SYNC_FAILURE]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that cause some degradation
        if category in [ErrorCategory.LLM_FAILURE, ErrorCategory.CREWAI_FAILURE]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that are recoverable
        return ErrorSeverity.LOW
    
    def _find_error_handler(self, error_context: ErrorContext) -> Optional[IErrorHandler]:
        """Find appropriate error handler"""
        for handler in self.error_handlers:
            if handler.can_handle(error_context):
                return handler
        return None
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level"""
        log_message = f"Error {error_context.error_id}: {error_context.error_message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_error_statistics(self, error_context: ErrorContext) -> None:
        """Update error statistics"""
        category = error_context.category
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        
        # Update degradation level based on error frequency
        self._update_degradation_level()
    
    def _update_recovery_statistics(self, recovery_result: RecoveryResult) -> None:
        """Update recovery statistics"""
        strategy = recovery_result.strategy_used
        self.recovery_counts[strategy] = self.recovery_counts.get(strategy, 0) + 1
    
    def _store_error_in_history(self, error_context: ErrorContext) -> None:
        """Store error in history with size limit"""
        self.error_history.append(error_context)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _update_degradation_level(self) -> None:
        """Update system degradation level based on error patterns"""
        recent_errors = [e for e in self.error_history 
                        if (datetime.now() - e.timestamp).total_seconds() < 300]  # Last 5 minutes
        
        error_rate = len(recent_errors) / 5.0  # Errors per minute
        
        if error_rate > 10:
            self.degradation_level = 3  # Severe degradation
        elif error_rate > 5:
            self.degradation_level = 2  # Moderate degradation
        elif error_rate > 1:
            self.degradation_level = 1  # Light degradation
        else:
            self.degradation_level = 0  # Normal operation
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        return {
            "degradation_level": self.degradation_level,
            "fallback_mode": self.fallback_mode,
            "recent_error_count": len([e for e in self.error_history 
                                     if (datetime.now() - e.timestamp).total_seconds() < 300]),
            "error_categories": dict(self.error_counts),
            "recovery_strategies": dict(self.recovery_counts),
            "circuit_breakers": {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
        }


# Specific Error Handler Implementations

class LLMErrorHandler(IErrorHandler):
    """Handler for LLM-specific errors"""
    
    def __init__(self, circuit_breaker: CircuitBreaker, fallback_responses: Dict[str, str] = None):
        self.circuit_breaker = circuit_breaker
        self.fallback_responses = fallback_responses or {}
        self.retry_config = RetryConfig(max_attempts=3, base_delay=2.0)
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.LLM_FAILURE
    
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        operation = error_context.operation
        
        # Try fallback response first
        if operation in self.fallback_responses:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                message="Used fallback response",
                fallback_data=self.fallback_responses[operation]
            )
        
        # Try degraded response generation
        if error_context.context_data.get("agent_personality"):
            fallback_response = self._generate_degraded_response(error_context)
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.DEGRADE,
                message="Generated degraded response",
                fallback_data=fallback_response
            )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            message="No fallback available for LLM failure"
        )
    
    def get_priority(self) -> int:
        return 1  # High priority for LLM errors
    
    def _generate_degraded_response(self, error_context: ErrorContext) -> str:
        """Generate a simple rule-based response when LLM fails"""
        personality = error_context.context_data.get("agent_personality", "")
        
        if "strategist" in personality.lower():
            return "I need to analyze the situation more carefully. Let me gather more information before making a decision."
        elif "mediator" in personality.lower():
            return "Let's work together on this. I suggest we communicate and share our perspectives."
        elif "survivor" in personality.lower():
            return "I need to focus on immediate survival priorities. What resources do we have available?"
        else:
            return "I need more time to process this situation. Let me reconsider my options."


class MesaErrorHandler(IErrorHandler):
    """Handler for Mesa simulation errors"""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.MESA_FAILURE
    
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        # Mesa errors are often critical - try to reset the simulation state
        if "position" in error_context.error_message.lower():
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RESET,
                message="Reset agent positions to safe state"
            )
        elif "bounds" in error_context.error_message.lower():
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RESET,
                message="Reset agents to within bounds"
            )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            message="Mesa error requires manual intervention"
        )
    
    def get_priority(self) -> int:
        return 2  # High priority for Mesa errors


class StateSyncErrorHandler(IErrorHandler):
    """Handler for state synchronization errors"""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.STATE_SYNC_FAILURE
    
    async def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        # State sync errors can often be resolved by resetting state
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RESET,
            message="Reset state synchronization"
        )
    
    def get_priority(self) -> int:
        return 3


# Custom Exception Classes

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class MaxRetriesExceededError(Exception):
    """Raised when maximum retry attempts are exceeded"""
    pass


class GracefulDegradationError(Exception):
    """Raised when system enters graceful degradation mode"""
    pass


# Context Managers for Error Handling

@asynccontextmanager
async def error_handling_context(error_manager: HybridErrorManager, 
                                component: str, operation: str):
    """Context manager for automatic error handling"""
    try:
        yield
    except Exception as e:
        context = {"component": component, "operation": operation}
        recovery_result = await error_manager.handle_error(e, context)
        
        if not recovery_result.success:
            raise e  # Re-raise if recovery failed


@asynccontextmanager  
async def circuit_breaker_context(circuit_breaker: CircuitBreaker):
    """Context manager for circuit breaker protection"""
    try:
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker {circuit_breaker.name} is open")
        yield
        circuit_breaker._record_success()
    except Exception as e:
        circuit_breaker._record_failure()
        raise


# Utility functions for error handling setup

def setup_default_error_handling(error_manager: HybridErrorManager) -> None:
    """Setup default error handlers"""
    
    # Create circuit breakers
    llm_circuit_breaker = error_manager.get_circuit_breaker("llm", CircuitBreakerConfig(
        failure_threshold=5,
        timeout_duration=60,
        success_threshold=3
    ))
    
    # Register handlers
    error_manager.register_error_handler(LLMErrorHandler(llm_circuit_breaker))
    error_manager.register_error_handler(MesaErrorHandler())
    error_manager.register_error_handler(StateSyncErrorHandler())


def create_fallback_responses() -> Dict[str, str]:
    """Create fallback responses for common operations"""
    return {
        "analyze_situation": "I'm analyzing the current situation based on available information.",
        "make_decision": "I need to make a decision based on the current circumstances.",
        "communicate": "I want to communicate with my team members.",
        "take_action": "I'm taking action based on my assessment of the situation."
    }