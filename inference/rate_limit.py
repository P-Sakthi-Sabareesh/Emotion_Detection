import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60

_bucket: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def allow_request(bucket_key: str, max_requests_per_min: int) -> bool:
    if max_requests_per_min <= 0:
        return True
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS
    with _lock:
        queue = _bucket[bucket_key]
        while queue and queue[0] < cutoff:
            queue.popleft()
        if len(queue) >= max_requests_per_min:
            return False
        queue.append(now)
    return True
