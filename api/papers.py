from fastapi import APIRouter
from pathlib import Path

from config import UPLOADED_DIR

router = APIRouter(
    prefix="/papers",
    tags=["Uploaded Papers"]
)


@router.get("/uploaded")
async def list_uploaded_papers():
    """List all uploaded research papers."""

    if not UPLOADED_DIR.exists():
        return {
            "message": "No uploaded papers found yet.",
            "total_files": 0,
            "files": []
        }

    files = sorted(
        (
            path for path in UPLOADED_DIR.iterdir()
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        return {
            "message": "No research papers have been uploaded yet.",
            "total_files": 0,
            "files": []
        }

    papers = []

    for path in files:

        size_kb = path.stat().st_size / 1024

        papers.append({
            "filename": path.name,
            "size_kb": round(size_kb, 1),
        })

    return {
        "message": "Uploaded research papers retrieved successfully.",
        "total_files": len(papers),
        "files": papers
    }