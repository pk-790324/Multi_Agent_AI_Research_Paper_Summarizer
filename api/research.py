
import json

from pydantic import BaseModel, Field
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.research_service import run_research_pipeline


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
async def run_research(request: ResearchRequest):

    user_query = request.user_query.strip()

    if not user_query:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Research question cannot be empty."
        )

    def event_generator():

        for event in run_research_pipeline(user_query):

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
