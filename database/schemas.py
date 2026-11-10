from pydantic import BaseModel, EmailStr


# Data received during signup
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


# Data received during login
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# Data returned to the client
class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr