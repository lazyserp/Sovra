from concurrent.futures import ThreadPoolExecutor
import time
import pytest

from core.gateway.limiter import InMemoryRateLimiter, TokenBucket


def test_token_bucket_burst_and_exhaustion():
    """Verify that a token bucket allows burst up to capacity and rejects subsequent requests."""
    bucket = TokenBucket(rate=1.0, capacity=3.0, tokens=3.0)

    # First 3 requests should be allowed
    assert bucket.consume(1.0) == (True, 0.0)
    assert bucket.consume(1.0) == (True, 0.0)
    assert bucket.consume(1.0) == (True, 0.0)

    # 4th request must be rejected
    allowed, retry_after = bucket.consume(1.0)
    assert allowed is False
    assert retry_after > 0.0


def test_token_bucket_refill():
    """Verify that tokens refill proportionally over elapsed time."""
    # 10 tokens per second = 1 token every 0.1 seconds
    bucket = TokenBucket(rate=10.0, capacity=2.0, tokens=0.0)

    # Initially empty -> rejected
    allowed, retry_after = bucket.consume(1.0)
    assert allowed is False
    assert retry_after > 0.0

    # Wait 0.15 seconds (should generate >= 1 token)
    time.sleep(0.15)

    allowed, _ = bucket.consume(1.0)
    assert allowed is True


def test_rate_limiter_isolated_keys():
    """Verify that exhausting one API key's bucket does not throttle a different API key."""
    limiter = InMemoryRateLimiter(requests_per_minute=60, burst=2)

    # Exhaust key-A
    assert limiter.check("key-A")[0] is True
    assert limiter.check("key-A")[0] is True
    assert limiter.check("key-A")[0] is False  # key-A is throttled

    # key-B must still have its full burst capacity intact
    assert limiter.check("key-B")[0] is True
    assert limiter.check("key-B")[0] is True
    assert limiter.check("key-B")[0] is False  # now key-B is also throttled


def test_rate_limiter_thread_safety():
    """Verify that concurrent threads cannot bypass burst capacity due to race conditions."""
    burst_capacity = 5
    limiter = InMemoryRateLimiter(requests_per_minute=1, burst=burst_capacity)

    results = []

    def make_request():
        allowed, _ = limiter.check("concurrent-key")
        return allowed

    # Launch 20 concurrent threads trying to consume at the exact same instant
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(20)]
        results = [f.result() for f in futures]

    successes = [r for r in results if r is True]
    failures = [r for r in results if r is False]

    # Exactly burst_capacity must succeed, remainder must be throttled
    assert len(successes) == burst_capacity
    assert len(failures) == 20 - burst_capacity

