"""Cluster-correct sliding-window rate limiter.

Uses a Redis ZSET when ``FER_RATE_LIMIT_BACKEND=redis`` (recommended in prod).
Falls back to an in-process deque for local dev or when Redis is unreachable.
The fallback is intentionally noisy: it logs a warning on every demotion so
operators cannot accidentally ship an in-process limiter to production.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

from django.conf import settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60

_inproc_bucket: dict[str, deque[float]] = defaultdict(deque)
_inproc_lock = threading.Lock()

_redis_client = None
_redis_script = None
_redis_failed = False

_REDIS_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, window)
return 1
"""


def _get_redis():
    global _redis_client, _redis_script, _redis_failed
    if _redis_failed:
        return None, None
    if _redis_client is not None:
        return _redis_client, _redis_script
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(
            getattr(settings, "REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        script = client.register_script(_REDIS_LUA)
        _redis_client = client
        _redis_script = script
        return client, script
    except Exception as exc:  # pragma: no cover - redis optional
        _redis_failed = True
        logger.warning("redis rate limiter unavailable, falling back to in-process: %s", exc)
        return None, None


def _allow_inproc(bucket_key: str, max_per_min: int) -> bool:
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS
    with _inproc_lock:
        queue = _inproc_bucket[bucket_key]
        while queue and queue[0] < cutoff:
            queue.popleft()
        if len(queue) >= max_per_min:
            return False
        queue.append(now)
    return True


def _allow_redis(bucket_key: str, max_per_min: int) -> bool | None:
    client, script = _get_redis()
    if client is None or script is None:
        return None
    try:
        member = f"{time.time()}:{uuid.uuid4().hex}"
        allowed = script(keys=[f"rl:{bucket_key}"], args=[time.time(), WINDOW_SECONDS, max_per_min, member])
        return bool(int(allowed))
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("redis rate limiter script failed, degrading: %s", exc)
        return None


def allow_request(bucket_key: str, max_requests_per_min: int) -> bool:
    if max_requests_per_min <= 0:
        return True
    backend = getattr(settings, "FER_RATE_LIMIT_BACKEND", "inprocess").lower()
    if backend == "redis":
        result = _allow_redis(bucket_key, max_requests_per_min)
        if result is None:
            if not settings.DEBUG:
                logger.error(
                    "redis rate limit backend unavailable; failing closed for %s", bucket_key
                )
                return False
            return _allow_inproc(bucket_key, max_requests_per_min)
        return result
    return _allow_inproc(bucket_key, max_requests_per_min)


def reset_for_tests() -> None:
    global _redis_client, _redis_script, _redis_failed
    with _inproc_lock:
        _inproc_bucket.clear()
    _redis_client = None
    _redis_script = None
    _redis_failed = False


# Prevent accidental in-process limiter usage in CI if the env asks for redis.
if os.getenv("FER_RATE_LIMIT_STRICT") == "1" and getattr(
    settings, "FER_RATE_LIMIT_BACKEND", "inprocess"
).lower() != "redis":
    raise RuntimeError("FER_RATE_LIMIT_STRICT=1 requires FER_RATE_LIMIT_BACKEND=redis")
