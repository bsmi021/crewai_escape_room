"""
LLM Exception Classes for Circuit Breaker Pattern

Defines custom exceptions for LLM integration and circuit breaker handling.
"""


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class CircuitOpenError(LLMError):
    """Raised when circuit breaker is open and calls are not allowed"""
    def __init__(self, message: str = "Circuit breaker is open", failure_count: int = 0):
        super().__init__(message)
        self.failure_count = failure_count


class RateLimitExceededError(LLMError):
    """Raised when API rate limits are exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Raised when API authentication fails"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class ServiceUnavailableError(LLMError):
    """Raised when LLM service is temporarily unavailable"""
    def __init__(self, message: str = "Service temporarily unavailable", retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(LLMError):
    """Raised when LLM calls timeout"""
    def __init__(self, message: str = "LLM call timed out", timeout_duration: float = None):
        super().__init__(message)
        self.timeout_duration = timeout_duration


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available"""
    def __init__(self, message: str = "Model not found", model_name: str = None):
        super().__init__(message)
        self.model_name = model_name


class InvalidResponseError(LLMError):
    """Raised when LLM response is invalid or malformed"""
    def __init__(self, message: str = "Invalid LLM response", response: str = None):
        super().__init__(message)
        self.response = response