"""Login rate limiter for tracking failed authentication attempts per email to prevent brute-force attacks."""
import time
from collections import defaultdict, deque
from app.core.exceptions import RateLimitExceededError

# {email_lower: deque of failure timestamps}
_attempts: dict[str, deque] = defaultdict(deque)

_MAX_FAILURES = 5
_WINDOW_SECONDS = 900   # 15 minutes


def check_login_allowed(email: str) -> None:
    """Raise RateLimitExceededError if this email has too many recent failures.

    Call BEFORE verifying credentials — we check the counter first
    so we don't leak timing information about whether the email exists.
    """
    key = email.strip().lower()
    now = time.monotonic()
    window = _attempts[key]

    # Evict timestamps outside the window
    while window and window[0] < now - _WINDOW_SECONDS:
        window.popleft()

    if len(window) >= _MAX_FAILURES:
        raise RateLimitExceededError(
            f"Too many failed login attempts for this account. "
            f"Please wait {_WINDOW_SECONDS // 60} minutes before trying again.",
            details={"retry_after_seconds": _WINDOW_SECONDS},
        )


def record_failure(email: str) -> None:
    """Record a failed login attempt for `email`."""
    key = email.strip().lower()
    _attempts[key].append(time.monotonic())


def record_success(email: str) -> None:
    """Reset the failure counter for `email` after a successful login."""
    key = email.strip().lower()
    _attempts.pop(key, None)
