import contextvars
import logging
import uuid

_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()


class RequestIdMiddleware:
    """Attach a request id to each request, expose it to logs and response headers."""

    HEADER = "HTTP_X_REQUEST_ID"
    RESPONSE_HEADER = "X-Request-ID"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        incoming = request.META.get(self.HEADER)
        rid = incoming if incoming and len(incoming) <= 128 else uuid.uuid4().hex
        token = _REQUEST_ID.set(rid)
        request.request_id = rid
        try:
            response = self.get_response(request)
        finally:
            _REQUEST_ID.reset(token)
        response[self.RESPONSE_HEADER] = rid
        return response


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID.get()
        return True
