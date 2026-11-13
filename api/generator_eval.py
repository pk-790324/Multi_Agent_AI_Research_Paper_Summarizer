from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from evals.eval_generator import (
    run_generator_evaluation
)

from goldens.generator_faithfulness import (
    generate_faithfulness_goldens
)

from api.auth_utils import get_current_user


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"]
)


# ==========================================
# 1. GENERATE GENERATOR GOLDENS
# ==========================================

@router.post("/generator/generate-goldens")
def generate_generator_goldens_endpoint(
    current_user: dict = Depends(get_current_user)
):

    try:

        output_path = generate_faithfulness_goldens(
            current_user=current_user
        )

        return {

            "status": "success",

            "message": "Generator goldens generated successfully.",

            "golden_path": output_path,

        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print(
            f"Generator golden generation failed: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# 2. RUN GENERATOR EVALUATION
# ==========================================

@router.post("/generator")
def generator_evaluation_endpoint(
    current_user: dict = Depends(get_current_user)
):

    try:

        result = run_generator_evaluation(
            current_user=current_user
        )

        return result

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print(
            f"Generator evaluation failed: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )