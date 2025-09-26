# app/auth/schemas.py
from pydantic import BaseModel, EmailStr, Field

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class AuthOut(BaseModel):
    email: EmailStr
    full_name: str | None = None
    role: str = "user"

    class Config:
        from_attributes = True  # pydantic v2
