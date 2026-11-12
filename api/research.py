
import json

from pydantic import BaseModel, Field
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.research_service import run_research_pipeline
from fastapi import APIRouter, HTTPException, Depends
from api.auth_utils import get_current_user

router = APIRouter(
    prefix="/research",
    tags=["Research Pipeline"]
)


class ResearchRequest(BaseModel):

    user_query: str = Field(
        ...,
        min_length=1,
        description="Research question to analyze"
    )



@router.post("/run")
async def run_research(
    request: ResearchRequest,
    current_user: dict = Depends(get_current_user),
):

    user_query = request.user_query.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Research question cannot be empty."
        )

    # Identity comes from validated authentication
    user_id = str(current_user["_id"])

    def event_generator():

        for event in run_research_pipeline(
            user_query=user_query,
            user_id=user_id,
        ):

            yield (
                "data: "
                + json.dumps(event, default=str)
                + "\n\n"
            )

        yield "event: done\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

