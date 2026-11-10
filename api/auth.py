from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from database.database import user_collection
from database.schemas import UserCreate, UserLogin, UserResponse
from database.models import hash_password, verify_password

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

    # Find user by email
    existing_user = user_collection.find_one(
        {"email": user.email}
    )

    if not existing_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Verify password
    is_valid = verify_password(
        user.password,
        existing_user["password_hash"]
    )

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
        "message": "Login successful",
        "user": {
            "id": str(existing_user["_id"]),
            "name": existing_user["name"],
            "email": existing_user["email"]
        }
    }