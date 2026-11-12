from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from database.database import user_collection
from database.schemas import UserCreate, UserLogin, UserResponse
from database.models import hash_password, verify_password
from api.auth_utils import create_access_token

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# SIGNUP ENDPOINT
@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def signup(user: UserCreate):

    # Check whether email already exists
    existing_user = user_collection.find_one(
        {"email": user.email}
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # Hash password
    hashed_password = hash_password(user.password)

    # Create MongoDB document
    user_data = {
        "name": user.name,
        "email": user.email,
        "password_hash": hashed_password
    }

    # Insert into MongoDB
    try:
        result = user_collection.insert_one(user_data)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    return {
        "id": str(result.inserted_id),
        "name": user.name,
        "email": user.email
    }


# LOGIN ENDPOINT

@router.post("/login")
def login(user: UserLogin):

    # 1. Find user
    existing_user = user_collection.find_one(
        {"email": user.email}
    )

    if not existing_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # 2. Verify password
    is_valid = verify_password(
        user.password,
        existing_user["password_hash"]
    )

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # 3. Generate JWT
    user_id = str(existing_user["_id"])

    access_token = create_access_token(user_id)

    # 4. Return token and user details
    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": existing_user["name"],
            "email": existing_user["email"]
        }
    }
