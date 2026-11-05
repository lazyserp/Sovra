import threading
import time
from dataclasses import dataclass,field

@dataclass
class TokenBucket:
    rate: float
    capacity: float = 0.0 
    tokens: float = 0.0
    last_updated: float = field(default_factory=time.monotonic)


    def consume(self,amount:float = 1.0) -> tuple[bool,float]:
        """ Attempt to consume tokens from the bucket"""
        """Returns: (allowed: bool, retry_after_seconds) """

        now = time.monotonic()
        elapsed = now - self.last_updated
        self.last_updated = now

        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        else:
            missing = amount - self.tokens
            retry_after = missing / self.rate
            return False, round(retry_after,2)


class InMemoryRateLimiter:
    """Thread-safe in-memory rate limiter per API key using Token Bucket."""

    def __init__(self,requests_per_minute: int = 60, burst:int = 10):
        self.rate = requests_per_minute / 60.0
        self.capacity = float(burst)
        self._buckets: dict[str,TokenBucket] = {}
        self._lock = threading.Lock()

    
    def check(self,key_id: str) -> tuple[bool,float]:
        """
        Check if an API key is allowed to make a request.
        
        Returns:
            (allowed: bool, retry_after_seconds: float)
        """
        with self._lock:
            if key_id not in self._buckets:
                # Initialize new bucket with full burst capacity
                self._buckets[key_id] = TokenBucket(
                    rate=self.rate,
                    capacity=self.capacity,
                    tokens=self.capacity,
                )
            return self._buckets[key_id].consume(1.0)

