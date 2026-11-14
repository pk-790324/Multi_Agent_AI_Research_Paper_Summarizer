
# =========================================================
# RAG PIPELINE EVALUATION API
# =========================================================
from fastapi import APIRouter, Depends, HTTPException
from evals.eval_rag_pipeline import run_rag_evaluation
from api.auth_utils import get_current_user

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"]
)


@router.post("/rag")
def evaluate_rag_pipeline(
    current_user: dict = Depends(get_current_user),
):
    try:

        result = run_rag_evaluation(current_user)

        return result

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline evaluation failed: {str(exc)}",
        )
