import io
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIN_DIMENSION = 32
MAX_DIMENSION = 4096


async def validate_image_upload(file: UploadFile) -> Image.Image:
    """
    Validates uploaded image against production 'what-if' edge cases:
    - What if the client uploads a non-image MIME type?
    - What if the file is empty or exceeds 5 MB?
    - What if the file has a valid extension but corrupted binary bytes?
    - What if image dimensions are too small or excessively large?
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Allowed: JPEG, PNG, WEBP.",
        )

    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 5 MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()  # Verify binary integrity
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Corrupted or unreadable image binary stream.",
        )

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image dimensions ({width}x{height}) are below minimum {MIN_DIMENSION}x{MIN_DIMENSION}px.",
        )

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image dimensions ({width}x{height}) exceed maximum {MAX_DIMENSION}x{MAX_DIMENSION}px.",
        )

    return image
