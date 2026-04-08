import io

from django.conf import settings
from PIL import Image, UnidentifiedImageError

from inference.services import PredictionInputError


def validate_image_bytes(content: bytes) -> None:
    max_bytes = settings.FER_MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise PredictionInputError(f"File is too large. Maximum allowed is {settings.FER_MAX_UPLOAD_MB} MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PredictionInputError("Unsupported or corrupted image file.") from exc

    if width * height > settings.FER_MAX_IMAGE_PIXELS:
        raise PredictionInputError(
            f"Image resolution is too high. Maximum pixels allowed: {settings.FER_MAX_IMAGE_PIXELS}."
        )
