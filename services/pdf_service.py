
from pathlib import Path
import shutil


class PDFService:
    """Validate and save uploaded research documents."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".ppt", ".pptx",
        ".txt", ".md", ".rtf",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    UPLOAD_DIR = Path("data/uploaded_papers")

    @classmethod
    def save_uploaded_document(cls, file_path: str) -> dict:
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

        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        destination = cls.UPLOAD_DIR / source.name

        # Avoid copying a file onto itself.
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

        return {
            "file_name": destination.name,
            "file_path": str(destination.resolve()),
            "file_extension": extension,
            "file_size_bytes": file_size,
        }
