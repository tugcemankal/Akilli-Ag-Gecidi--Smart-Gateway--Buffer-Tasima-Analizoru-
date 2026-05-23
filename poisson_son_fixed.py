from __future__ import annotations

import math
from typing import Optional


class PacketLossModelError(Exception):
    """Base exception for Smart Gateway packet loss model."""


class InvalidLambdaError(PacketLossModelError):
    """Raised when lambda is invalid."""


class InvalidCapacityError(PacketLossModelError):
    """Raised when capacity K is invalid."""


class OverflowProtectedPoissonModel:
    """
    Smart Gateway packet arrival model (per 1 ms time window) - OVERFLOW PROTECTED.
    
    **Queueing Theory Context:**
    - M/M/1/K queue approximation with Poisson arrivals
    - Tail drop when arrivals exceed capacity K
    - Packet loss probability = P(X > K)
    
    **Overflow Protection:**
    - Logarithmic calculation for large values: log(P) = -λ + x·log(λ) - log(x!)
    - Uses math.lgamma(x+1) instead of math.factorial(x)
    - Automatic method selection based on value magnitude
    - Try-except blocks for all critical calculations
    """

    # Safe calculation limits
    MAX_FACTORIAL_ARG = 170  # math.factorial(171) overflows
    MAX_LAMBDA_DIRECT = 100.0  # exp(-λ) underflow threshold
    MAX_X_DIRECT = 100  # Direct calculation safe limit

    def __init__(self, lambda_per_ms: float, capacity_k: int) -> None:
        self.lambda_per_ms = self._validate_lambda(lambda_per_ms)
        self.capacity_k = self._validate_capacity(capacity_k)

    def poisson_probability(self, x: int) -> float:
        """
        P(X = x) for Poisson(λ) - OVERFLOW PROTECTED.
        
        Formula: P(X=x) = (e^-λ · λ^x) / x!
        
        **Protection Strategy:**
        1. Small values (x ≤ 170, λ ≤ 100): Direct calculation
        2. Large values: Logarithmic calculation
           log(P) = -λ + x·log(λ) - log(x!)
           P = exp(log(P))
        
        Args:
            x: Number of events (packets)
            
        Returns:
            Probability value [0, 1]
            
        Raises:
            PacketLossModelError: If calculation fails
        """
        x = self._validate_x(x)
        lam = self.lambda_per_ms
        
        # Special case: λ = 0
        if lam == 0.0:
            return 1.0 if x == 0 else 0.0
        
        try:
            # Strategy selection based on value magnitude
            if (x <= self.MAX_X_DIRECT and 
                lam <= self.MAX_LAMBDA_DIRECT and
                x <= self.MAX_FACTORIAL_ARG):
                # Safe zone: Direct calculation
                return self._calculate_direct(lam, x)
            else:
                # Risk zone: Logarithmic calculation
                return self._calculate_logarithmic(lam, x)
                
        except (OverflowError, ValueError, FloatingPointError) as e:
            # Fallback to logarithmic if direct fails
            try:
                return self._calculate_logarithmic(lam, x)
            except Exception as log_e:
                raise PacketLossModelError(
                    f"Poisson calculation failed for λ={lam}, x={x}: {e}"
                ) from log_e

    def _calculate_direct(self, lam: float, x: int) -> float:
        """Direct Poisson calculation for small values."""
        # Optimize order to minimize intermediate overflow
        # P = exp(-λ) · (λ^x / x!)
        exp_term = math.exp(-lam)
        lambda_pow = lam ** x
        factorial = math.factorial(x)
        
        return (exp_term * lambda_pow) / factorial

    def _calculate_logarithmic(self, lam: float, x: int) -> float:
        """
        Logarithmic Poisson calculation for large values.
        
        Formula:
            log(P) = -λ + x·log(λ) - log(x!)
            log(x!) = lgamma(x+1)
            P = exp(log(P))
        """
        # Calculate in log space
        log_lambda = math.log(lam)
        log_factorial = math.lgamma(x + 1)  # log(x!)
        
        log_prob = -lam + (x * log_lambda) - log_factorial
        
        # Handle extreme underflow (log_prob < -1000 means P ≈ 0)
        if log_prob < -1000:
            return 0.0
        
        return math.exp(log_prob)

    def packet_loss_probability(self) -> float:
        """
        Packet loss event probability: P(X > K) = 1 - Σ_{x=0..K} P(X=x)
        
        **Optimized for large K:**
        - Uses recurrence relation for numerical stability:
          P(X=x+1) = P(X=x) · λ / (x+1)
        - Starts from P(X=0) = exp(-λ)
        - Avoids repeated expensive calculations
        
        Returns:
            Packet loss probability [0, 1]
        """
        lam = self.lambda_per_ms
        K = self.capacity_k
        
        # Special cases
        if lam == 0.0:
            return 0.0  # No arrivals = no loss
        if K < 0:
            return 1.0  # Negative capacity = all dropped
        
        try:
            # Use recurrence relation for better stability
            # P(X=0) = exp(-λ)
            # P(X=x+1) = P(X=x) · λ / (x+1)
            
            cumulative = 0.0
            px = math.exp(-lam)  # P(X=0)
            
            for x in range(K + 1):
                cumulative += px
                # Next term using recurrence
                if x < K:  # Don't calculate beyond K
                    px = px * lam / (x + 1)
            
            # Clamp to [0, 1] to handle floating point errors
            loss_prob = max(0.0, min(1.0, 1.0 - cumulative))
            return loss_prob
            
        except (OverflowError, ValueError) as e:
            # Fallback: Use individual probability calculations
            try:
                cumulative = sum(
                    self.poisson_probability(x) 
                    for x in range(K + 1)
                )
                return max(0.0, min(1.0, 1.0 - cumulative))
            except Exception as fallback_e:
                raise PacketLossModelError(
                    f"Packet loss calculation failed for λ={lam}, K={K}"
                ) from fallback_e

    def expected_dropped_packets_per_ms(self) -> float:
        """
        Expected dropped packets per ms: E[(X-K)^+]
        
        **Queueing Theory Formula:**
        E[(X-K)^+] = Σ_{x=K+1..∞} (x-K) · P(X=x)
        
        **Implementation:**
        - Uses recurrence relation for P(X=x)
        - Truncates when contribution becomes negligible (< eps)
        - Maximum iterations: 200,000
        
        Returns:
            Expected number of dropped packets per ms
        """
        lam = self.lambda_per_ms
        K = self.capacity_k
        
        if K < 0:
            # All packets dropped
            return lam
        if lam == 0.0:
            return 0.0
        
        try:
            # Start from x = K+1 with recurrence
            x = K + 1
            
            # Initial probability P(X=K+1) using logarithmic method
            px = self.poisson_probability(x)
            
            dropped_expectation = 0.0
            eps = 1e-12
            max_terms = 200000
            terms = 0
            
            while terms < max_terms:
                contrib = (x - K) * px
                dropped_expectation += contrib
                
                # Convergence check
                if (contrib < eps and px < eps and 
                    x > lam + 10 * math.sqrt(lam + 1.0)):
                    break
                
                # Recurrence: P(X=x+1) = P(X=x) · λ / (x+1)
                x += 1
                px = px * lam / x
                terms += 1
            
            return max(0.0, dropped_expectation)
            
        except (OverflowError, ValueError, FloatingPointError) as e:
            # Fallback approximation using normal distribution
            # For large λ: Poisson ≈ N(λ, λ)
            try:
                # E[(X-K)^+] ≈ E[X] - K + E[(K-X)^+] (for large λ)
                # Simplified: max(0, λ - K) for very large values
                return max(0.0, lam - K)
            except Exception:
                raise PacketLossModelError(
                    f"Expected dropped calculation failed for λ={lam}, K={K}"
                ) from e

    def summary(self) -> dict:
        """Return model outputs as a backend-friendly dictionary."""
        try:
            return {
                "scenario": (
                    "Smart Gateway receives asynchronous telemetry packets; "
                    "tail drop occurs when arrivals in 1 ms exceed capacity K."
                ),
                "lambda_per_ms": self.lambda_per_ms,
                "capacity_k": self.capacity_k,
                "packet_loss_event_probability": self.packet_loss_probability(),
                "expected_dropped_packets_per_ms": self.expected_dropped_packets_per_ms(),
                "calculation_status": "success",
            }
        except PacketLossModelError as e:
            return {
                "scenario": "Smart Gateway packet loss model",
                "lambda_per_ms": self.lambda_per_ms,
                "capacity_k": self.capacity_k,
                "packet_loss_event_probability": None,
                "expected_dropped_packets_per_ms": None,
                "calculation_status": "error",
                "error_message": str(e),
            }

    @staticmethod
    def _validate_lambda(value: float) -> float:
        """Validate lambda parameter."""
        try:
            lam = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidLambdaError("Lambda must be numeric.") from exc
        
        if lam < 0:
            raise InvalidLambdaError("Lambda cannot be negative.")
        if not math.isfinite(lam):
            raise InvalidLambdaError("Lambda must be finite.")
        if lam > 1e6:  # Practical upper limit
            raise InvalidLambdaError("Lambda exceeds practical limit (1e6).")
        
        return lam

    @staticmethod
    def _validate_capacity(value: int) -> int:
        """Validate capacity K parameter."""
        if not isinstance(value, int):
            raise InvalidCapacityError("Capacity K must be an integer.")
        if value < 0:
            raise InvalidCapacityError("Capacity K cannot be negative.")
        if value > 10000000:  # Practical upper limit
            raise InvalidCapacityError("Capacity K exceeds practical limit.")
        return value

    @staticmethod
    def _validate_x(value: int) -> int:
        """Validate event count x."""
        if not isinstance(value, int):
            raise PacketLossModelError("x must be an integer.")
        if value < 0:
            raise PacketLossModelError("x cannot be negative.")
        if value > 10000000:  # Practical limit
            raise PacketLossModelError("x exceeds calculation limit.")
        return value


# Convenience function for quick calculations
def calculate_packet_loss(lambda_per_ms: float, capacity_k: int) -> dict:
    """
    Quick packet loss calculation for router queue analysis.
    
    Args:
        lambda_per_ms: Average packet arrival rate per ms
        capacity_k: Maximum packets processable per ms
        
    Returns:
        Dictionary with loss probability and expected drops
    """
    try:
        model = OverflowProtectedPoissonModel(lambda_per_ms, capacity_k)
        return {
            "lambda_per_ms": lambda_per_ms,
            "capacity_k": capacity_k,
            "packet_loss_probability": model.packet_loss_probability(),
            "expected_dropped_per_ms": model.expected_dropped_packets_per_ms(),
            "status": "success",
        }
    except PacketLossModelError as e:
        return {
            "lambda_per_ms": lambda_per_ms,
            "capacity_k": capacity_k,
            "error": str(e),
            "status": "error",
        }
