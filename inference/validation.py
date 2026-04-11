import io

from django.conf import settings
from PIL import Image, ImageFile, UnidentifiedImageError

from inference.services import PredictionInputError

# Pillow raises DecompressionBombError when decoded pixel count crosses this.
Image.MAX_IMAGE_PIXELS = max(settings.FER_MAX_IMAGE_PIXELS, 1)
ImageFile.LOAD_TRUNCATED_IMAGES = False


def validate_image_bytes(content: bytes) -> None:
    max_bytes = settings.FER_MAX_UPLOAD_MB * 1024 * 1024
    if not content:
        raise PredictionInputError("Empty image payload.")
    if len(content) > max_bytes:
        raise PredictionInputError(
            f"File is too large. Maximum allowed is {settings.FER_MAX_UPLOAD_MB} MB."
        )

    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(content)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            frame_count = getattr(image, "n_frames", 1)
    except Image.DecompressionBombError as exc:
        raise PredictionInputError("Image is too large to decode safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PredictionInputError("Unsupported or corrupted image file.") from exc

    if image_format not in settings.FER_ALLOWED_IMAGE_FORMATS:
        raise PredictionInputError(
            f"Unsupported image format: {image_format or 'unknown'}."
        )
    if frame_count and frame_count > 1:
        raise PredictionInputError("Multi-frame images are not allowed.")
    if width <= 0 or height <= 0:
        raise PredictionInputError("Image has invalid dimensions.")
    if width * height > settings.FER_MAX_IMAGE_PIXELS:
        raise PredictionInputError(
            f"Image resolution is too high. Maximum pixels allowed: {settings.FER_MAX_IMAGE_PIXELS}."
        )
