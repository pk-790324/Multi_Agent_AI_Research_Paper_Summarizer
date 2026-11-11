
import hashlib
import shutil

from pathlib import Path


class PDFService:
    """Validate, inspect and save uploaded research documents."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".ppt", ".pptx",
        ".txt", ".md", ".rtf",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024
    UPLOAD_DIR = Path("data/uploaded_papers")

    @classmethod
    def inspect_uploaded_document(cls, file_path: str) -> dict:
        """Validate the file and calculate its SHA-256 hash."""

        if not file_path:
            raise ValueError("No document was uploaded.")

        source = Path(file_path)

        if not source.is_file():
            raise FileNotFoundError(
                f"Uploaded document not found: {source}"
            )

        extension = source.suffix.lower()

        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        file_size = source.stat().st_size

        if file_size == 0:
            raise ValueError("The uploaded document is empty.")

        if file_size > cls.MAX_FILE_SIZE:
            raise ValueError("File exceeds the 50 MB limit.")

        # Calculate SHA-256 incrementally.
        sha256 = hashlib.sha256()

        with source.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024), b""
            ):
                sha256.update(chunk)

        return {
            "original_filename": source.name,
            "file_hash": sha256.hexdigest(),
            "file_extension": extension,
            "file_size_bytes": file_size,
        }

    @classmethod
    def save_uploaded_document(
        cls,
        file_path: str,
        document_id: str,
        extension: str,
    ) -> dict:
        """Save document using a unique document ID."""

        source = Path(file_path)

        cls.UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            cls.UPLOAD_DIR / f"{document_id}{extension}"
        )

        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

        return {
            "file_name": destination.name,
            "file_path": str(destination.resolve()),
        }
