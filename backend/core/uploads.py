"""Validation and storage for uploaded loan documents.

Only PDF and common image formats are accepted, and every file is capped at
:data:`MAX_UPLOAD_BYTES`. Files are stored on disk under
``backend/uploads/documents`` with a random name; the original filename and
metadata are kept on the ``DocumentRequest`` row so the officer can preview or
download exactly what the applicant submitted.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per document
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)

UPLOAD_ROOT = Path(__file__).resolve().parent.parent / "uploads"
DOCUMENTS_DIR = UPLOAD_ROOT / "documents"

_CHUNK_SIZE = 512 * 1024

# Canonical mime type -> extension used on disk.
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Extension -> mime type accepted from the multipart part metadata.
EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

ACCEPTED_LABEL = "PDF, JPG, PNG or WEBP"


@dataclass
class StoredFile:
    """Metadata for a file that has been written to disk."""

    stored_path: str  # relative to UPLOAD_ROOT, e.g. "documents/<uuid>.pdf"
    file_name: str  # original filename supplied by the client
    mime_type: str
    file_size: int


def _detect_mime(head: bytes) -> Optional[str]:
    """Best-effort content sniffing so a renamed .exe cannot pass as a PDF."""
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def save_upload(upload: UploadFile) -> StoredFile:
    """Validate an upload, persist it, and return its metadata.

    Raises ``HTTPException`` with 400/413/415 when the file is empty, too large,
    or not one of the accepted document formats.
    """
    original_name = Path((upload.filename or "").strip()).name
    if not original_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )
    if original_name.startswith(".") or len(original_name) > 160:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded filename is not valid.",
        )

    suffix = Path(original_name).suffix.lower()
    declared_type = (upload.content_type or "").split(";")[0].strip().lower()
    if suffix not in EXTENSION_TO_MIME and declared_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type. Upload {ACCEPTED_LABEL} "
                f"(max {MAX_UPLOAD_MB} MB)."
            ),
        )

    payload = bytearray()
    try:
        while True:
            chunk = upload.file.read(_CHUNK_SIZE)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File is larger than the {MAX_UPLOAD_MB} MB limit. "
                        "Compress the document or split it into smaller files."
                    ),
                )
    finally:
        upload.file.close()

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    sniffed_type = _detect_mime(bytes(payload[:16]))
    mime_type = sniffed_type or EXTENSION_TO_MIME.get(suffix) or declared_type
    if mime_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "The file contents do not match an allowed document format. "
                f"Upload {ACCEPTED_LABEL}."
            ),
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{ALLOWED_TYPES[mime_type]}"
    destination = DOCUMENTS_DIR / stored_name
    destination.write_bytes(bytes(payload))

    return StoredFile(
        stored_path=f"documents/{stored_name}",
        file_name=original_name,
        mime_type=mime_type,
        file_size=len(payload),
    )


def resolve_stored_path(stored_path: Optional[str]) -> Optional[Path]:
    """Resolve a stored path, refusing anything that escapes the upload root."""
    if not stored_path:
        return None
    candidate = (UPLOAD_ROOT / stored_path).resolve()
    try:
        candidate.relative_to(UPLOAD_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def delete_stored_file(stored_path: Optional[str]) -> None:
    """Remove a stored file if it exists; safe to call repeatedly."""
    path = resolve_stored_path(stored_path)
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass
