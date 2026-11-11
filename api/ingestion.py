
import json

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from api.ingestion_service import run_ingestion_pipeline


router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion"]
)


from config import SUPPORTED_FILES,PROJECT_DIR,UPLOAD_DIR,UPLOADED_DIR




UPLOADED_DIR.mkdir(parents=True, exist_ok=True)




@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a document and stream ingestion progress.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    filename = Path(file.filename).name

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {extension}"
        )

    file_path = UPLOADED_DIR / filename

    try:

        # Save uploaded file.
        with file_path.open("wb") as buffer:

            while content := await file.read(1024 * 1024):
                buffer.write(content)

        await file.close()

    except Exception as exc:

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(exc)}"
        )

    def event_generator():

        for event in run_ingestion_pipeline(
            str(file_path),
            {"id": user_id},
        ):

            yield (
                "data: "
                + json.dumps(event, default=str)
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
