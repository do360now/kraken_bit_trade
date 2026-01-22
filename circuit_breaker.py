"""
Thread-safe circuit breaker implementation for fault tolerance.
"""
import time
import threading
from typing import Callable, Any, Dict, Tuple
from functools import wraps
from core.constants import CircuitState
from core.exceptions import APIError


class CircuitBreakerException(APIError):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Recovery attempt, limited requests allowed
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 300,
        expected_exception: tuple = (Exception,),
        half_open_attempts: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Tuple of exceptions to count as failures
            half_open_attempts: Number of successful calls needed to close from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.half_open_attempts = half_open_attempts
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
        # Thread safety
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result from func
        
        Raises:
            CircuitBreakerException: If circuit is open
        """
        with self._lock:
            current_state = self._check_state()
            
            if current_state == CircuitState.OPEN:
                raise CircuitBreakerException(
                    f"Circuit breaker OPEN for {func.__name__}. "
                    f"Will retry after {self.recovery_timeout}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure(func.__name__)
            raise e
    
    def _check_state(self) -> CircuitState:
        """Check and update circuit state based on time"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                from logger_config import logger
                logger.info(
                    f"Circuit breaker moved to HALF_OPEN",
                    failure_count=self.failure_count,
                    recovery_timeout=self.recovery_timeout
                )
        
        return self.state
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.half_open_attempts:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    
                    from logger_config import logger
                    logger.info(
                        "Circuit breaker moved to CLOSED after successful recovery",
                        required_successes=self.half_open_attempts
                    )
            
            elif self.state == CircuitState.CLOSED:
                # Gradually reduce failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, func_name: str):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure during recovery, reopen circuit
                self.state = CircuitState.OPEN
                from logger_config import logger
                logger.error(
                    f"Circuit breaker REOPENED for {func_name} after failure in HALF_OPEN state",
                    failure_count=self.failure_count
                )
            
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                from logger_config import logger
                logger.error(
                    f"Circuit breaker OPENED for {func_name}",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold
                )
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time else None
            }


# Thread-safe global registry of circuit breakers
_breakers: Dict[Tuple[str, str], CircuitBreaker] = {}
_breaker_lock = threading.Lock()


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 300,
    expected_exception: tuple = (Exception,)
):
    """
    Decorator to add circuit breaker protection to a function.
    
    Args:
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before attempting recovery
        expected_exception: Tuple of exceptions to count as failures
    
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def api_call():
            # Make API call
            pass
    """
    def decorator(func):
        # Use module + qualname for unique identification
        key = (func.__module__, func.__qualname__)
        
        with _breaker_lock:
            if key not in _breakers:
                _breakers[key] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception
                )
            breaker = _breakers[key]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to wrapper for manual control
        wrapper.breaker = breaker
        
        return wrapper
    return decorator


def get_breaker_status(func: Callable) -> Dict[str, Any]:
    """
    Get status of circuit breaker for a function.
    
    Args:
        func: Function decorated with @circuit_breaker
    
    Returns:
        Status dict or None if not found
    """
    if hasattr(func, 'breaker'):
        return func.breaker.get_status()
    return None


def reset_breaker(func: Callable):
    """
    Manually reset circuit breaker for a function.
    
    Args:
        func: Function decorated with @circuit_breaker
    """
    if hasattr(func, 'breaker'):
        func.breaker.reset()


def get_all_breaker_statuses() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    with _breaker_lock:
        return {
            f"{module}.{qualname}": breaker.get_status()
            for (module, qualname), breaker in _breakers.items()
        }