from fastapi import FastAPI

from api.ingestion import router as ingestion_router
from api.research import router as research_router
from api.papers import router as papers_router
from api.auth import router as auth_router


app = FastAPI(
    title="Multi-Agent Research Paper Summarizer",
    version="1.0.0"
)


#routes for ingestion and agent
app.include_router(ingestion_router)
app.include_router(research_router)
app.include_router(papers_router)


#routes for login,signup i.e user information
app.include_router(auth_router)

@app.get("/")
async def home():
    return {
        "message": "Research Paper Summarizer API is running!"
    }