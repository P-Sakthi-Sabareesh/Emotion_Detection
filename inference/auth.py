"""Uniform auth for prediction endpoints.

Accepts either:
- A Django session authenticated user (normal browser flow), or
- A Bearer token listed in ``settings.FER_API_TOKENS``.

Rejects cleanly with JSON ``401`` so API clients never receive HTML login pages.
"""
from __future__ import annotations

import hmac
from functools import wraps
from typing import Callable

from django.conf import settings
from django.http import HttpRequest, JsonResponse


def _extract_bearer(request: HttpRequest) -> str | None:
    header = request.META.get("HTTP_AUTHORIZATION", "")
    if not header.lower().startswith("bearer "):
        return None
    return header.split(" ", 1)[1].strip()


def _token_is_valid(token: str) -> bool:
    if not token:
        return False
    for candidate in settings.FER_API_TOKENS:
        if hmac.compare_digest(token, candidate):
            return True
    return False


def is_authenticated(request: HttpRequest) -> bool:
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        return True
    token = _extract_bearer(request)
    return bool(token and _token_is_valid(token))


def require_api_auth(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapper(request: HttpRequest, *args, **kwargs):
        if not settings.FER_REQUIRE_AUTH:
            return view_func(request, *args, **kwargs)
        if is_authenticated(request):
            return view_func(request, *args, **kwargs)
        return JsonResponse(
            {"error": "authentication_required"},
            status=401,
            headers={"WWW-Authenticate": 'Bearer realm="api"'},
        )

    return wrapper
