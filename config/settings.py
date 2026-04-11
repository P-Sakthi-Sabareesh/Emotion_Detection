import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# --- Core security ----------------------------------------------------------
# DEBUG defaults to FALSE. Flip it explicitly for local dev via .env.
DEBUG = env_bool("DJANGO_DEBUG", False)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "")
if not DEBUG:
    if not SECRET_KEY or len(SECRET_KEY) < 50:
        raise RuntimeError(
            "DJANGO_SECRET_KEY must be set to a value of at least 50 characters in production."
        )
elif not SECRET_KEY:
    # Local-dev only: the production guard above raises when DEBUG is False,
    # so this literal is only ever reachable in a local-dev shell.
    SECRET_KEY = "dev-only-insecure-key-regenerate-for-any-non-local-use-0123456789"  # nosec B105

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS")
if not DEBUG and not ALLOWED_HOSTS:
    raise RuntimeError("DJANGO_ALLOWED_HOSTS must be set (comma-separated) in production.")
if DEBUG and not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# Django's test runner injects its own 'testserver' host, so we never add it here.
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
    "inference",
    "history",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "inference.middleware.RequestIdMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
if not DEBUG:
    MIDDLEWARE.insert(1, "django.middleware.gzip.GZipMiddleware")

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

if os.getenv("DATABASE_URL"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "emotiondetect"),
            "USER": os.getenv("POSTGRES_USER", "emotiondetect"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "127.0.0.1"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": int(os.getenv("POSTGRES_CONN_MAX_AGE", "60")),
            "OPTIONS": {"sslmode": os.getenv("POSTGRES_SSLMODE", "prefer")},
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 12}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("APP_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Upload hardening -------------------------------------------------------
# Bound in-memory upload size so a multipart POST cannot balloon the worker.
FER_MAX_UPLOAD_MB = int(os.getenv("FER_MAX_UPLOAD_MB", "8"))
DATA_UPLOAD_MAX_MEMORY_SIZE = FER_MAX_UPLOAD_MB * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = FER_MAX_UPLOAD_MB * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FIELDS = 200

# --- FER runtime config -----------------------------------------------------
FER_MODEL_PATH = os.getenv(
    "FER_MODEL_PATH",
    str(BASE_DIR / "ml" / "artifacts" / "emotion_classifier.joblib"),
)
FER_MODEL_MANIFEST_PATH = os.getenv(
    "FER_MODEL_MANIFEST_PATH",
    str(BASE_DIR / "ml" / "artifacts" / "artifact_manifest.json"),
)
FER_MODEL_SIGNING_KEY = os.getenv("FER_MODEL_SIGNING_KEY", "")
FER_REQUIRE_SIGNED_ARTIFACT = env_bool("FER_REQUIRE_SIGNED_ARTIFACT", not DEBUG)
FER_MODEL_VERSION = os.getenv("FER_MODEL_VERSION", "fer2013-sklearn-v2")
FER_FALLBACK_MODEL_VERSION = "fallback-heuristic-v1"

FER_MAX_IMAGE_PIXELS = int(os.getenv("FER_MAX_IMAGE_PIXELS", "4000000"))
FER_MAX_BASE64_BYTES = int(os.getenv("FER_MAX_BASE64_BYTES", "10000000"))
FER_ALLOWED_IMAGE_FORMATS = set(
    env_list("FER_ALLOWED_IMAGE_FORMATS", "JPEG,PNG,WEBP")
) or {"JPEG", "PNG", "WEBP"}

FER_API_RATE_LIMIT_IMAGE_PER_MIN = int(os.getenv("FER_API_RATE_LIMIT_IMAGE_PER_MIN", "24"))
FER_API_RATE_LIMIT_LIVE_PER_MIN = int(os.getenv("FER_API_RATE_LIMIT_LIVE_PER_MIN", "60"))
FER_API_RATE_LIMIT_SAMPLE_PER_MIN = int(os.getenv("FER_API_RATE_LIMIT_SAMPLE_PER_MIN", "40"))
FER_RATE_LIMIT_BACKEND = os.getenv("FER_RATE_LIMIT_BACKEND", "inprocess")  # "redis" | "inprocess"

FER_TRUST_X_FORWARDED_FOR = env_bool("FER_TRUST_X_FORWARDED_FOR", False)
FER_REQUIRE_AUTH = env_bool("FER_REQUIRE_AUTH", not DEBUG)
FER_API_TOKENS = set(env_list("FER_API_TOKENS"))  # comma-separated bearer tokens

# --- Consent & retention ----------------------------------------------------
FER_REQUIRE_CONSENT = env_bool("FER_REQUIRE_CONSENT", True)
# Default OFF: we do not persist images unless the operator explicitly enables.
FER_STORE_IMAGES = env_bool("FER_STORE_IMAGES", False)
FER_STORE_PREDICTIONS = env_bool("FER_STORE_PREDICTIONS", True)
FER_RETENTION_DAYS = int(os.getenv("FER_RETENTION_DAYS", "30"))
FER_IP_ANONYMIZE = env_bool("FER_IP_ANONYMIZE", True)

# --- Rule-based post-processing --------------------------------------------
FER_ENABLE_RULE_BASED_CORRECTION = env_bool("FER_ENABLE_RULE_BASED_CORRECTION", True)
FER_ENABLE_CONFIDENCE_FALLBACK = env_bool("FER_ENABLE_CONFIDENCE_FALLBACK", False)
FER_MIN_CONFIDENCE = float(os.getenv("FER_MIN_CONFIDENCE", "0.35"))
FER_MIN_MARGIN = float(os.getenv("FER_MIN_MARGIN", "0.05"))
FER_FALLBACK_EMOTION = os.getenv("FER_FALLBACK_EMOTION", "neutral")

# --- Redis ------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# --- HTTPS / transport ------------------------------------------------------
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = env_bool("DJANGO_SESSION_COOKIE_SECURE", not DEBUG)
CSRF_COOKIE_SECURE = env_bool("DJANGO_CSRF_COOKIE_SECURE", not DEBUG)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "0" if DEBUG else "31536000"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", not DEBUG)
SECURE_HSTS_PRELOAD = env_bool("DJANGO_SECURE_HSTS_PRELOAD", False)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
X_FRAME_OPTIONS = "DENY"
SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", not DEBUG)

# --- Structured logging -----------------------------------------------------
LOG_LEVEL = os.getenv("DJANGO_LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("DJANGO_LOG_FORMAT", "json" if not DEBUG else "plain")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"},
        "json": {"()": "config.logging.JsonFormatter"},
    },
    "filters": {
        "request_id": {"()": "inference.middleware.RequestIdFilter"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": LOG_FORMAT,
            "filters": ["request_id"],
        }
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "django.request": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "inference": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "history": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}
