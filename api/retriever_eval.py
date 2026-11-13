from fastapi import APIRouter, Depends, HTTPException

from services.hybrid_search_service import HybridSearchService
from evals.eval_retriever import run as run_retriever_eval
from evals.eval_retriever import format_evaluation_report
from api.auth_utils import get_current_user
from goldens.generator_goldens import generate_goldens


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"]
)


@router.post("/retriever")
def evaluate_retriever(
    current_user: dict = Depends(get_current_user)
):

    # Extract authenticated MongoDB user ID
    owner_id = str(current_user["_id"])

    try:
        retriever = HybridSearchService()
        golden_path = generate_goldens(current_user)

        result = run_retriever_eval(
            retriever=retriever,
            owner_id=owner_id,
            golden_path=golden_path
        )

        return {
            "message": "Retriever evaluation completed",
            "results": format_evaluation_report(result)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
