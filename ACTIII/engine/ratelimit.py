import time
import threading

class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                self._next += self.interval
            else:
                self._next = now + self.interval

_limiter = RateLimiter(60)

def set_rpm(rpm: float):
    global _limiter
    _limiter = RateLimiter(rpm)

def wait():
    _limiter.acquire()
