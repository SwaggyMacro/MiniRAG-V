from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")

class StandardResponse(BaseModel, Generic[T]):
    """
    Standard unified response structure for all API endpoints.
    """
    code: int = Field(default=200, description="Business logic status code")
    message: str = Field(default="success", description="Status message")
    data: Optional[T] = Field(default=None, description="Actual data payload")
