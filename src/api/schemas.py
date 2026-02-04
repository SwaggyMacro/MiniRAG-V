from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from src.config import get_default_retrieval_config

# 获取配置默认值
_retrieval_config = get_default_retrieval_config()


class InfoResponse(BaseModel):
    app_name: str = Field(..., description="Name of the application")
    version: str = Field(..., description="Current version of the application")
    status: str = Field(..., description="System status")


class ExplainRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Query string. If not provided, uses config vlm.prompt.system as default")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for the explanation")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image for visual analysis")


class ExplainResponse(BaseModel):
    original_query: str = Field(..., description="The original query received")
    explanation: str = Field(..., description="The generated explanation")
    confidence: float = Field(..., description="Confidence score of the explanation")
    image_analysis: Optional[str] = Field(default=None, description="VLM image analysis result")
    inference_time_ms: Optional[float] = Field(default=None, description="VLM inference time in milliseconds")


# ============================================================================
# 图搜图相关模型
# ============================================================================

class ImageSearchRequest(BaseModel):
    """图搜图请求"""
    image_base64: str = Field(..., description="Base64 encoded image")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results to return (default from config)")
    include_regulations: bool = Field(default=False, description="Whether to include related regulations")
    
    def get_top_k(self) -> int:
        return self.top_k if self.top_k is not None else _retrieval_config.image_top_k


class SimilarCaseResponse(BaseModel):
    """相似案例响应"""
    case_id: int = Field(..., description="Case ID")
    score: float = Field(..., description="Similarity score (0-1)")
    description: str = Field(..., description="Case description")
    accident_type: str = Field(..., description="Accident type")
    image_path: Optional[str] = Field(default=None, description="Image path")
    legal_basis_ids: List[int] = Field(default_factory=list, description="Related regulation IDs")


class RegulationResponse(BaseModel):
    """法规信息响应"""
    regulation_id: int = Field(..., description="Regulation ID")
    content: str = Field(..., description="Regulation content")
    reference: str = Field(..., description="Reference source")
    source_type: str = Field(..., description="Source type")
    tags: List[str] = Field(default_factory=list, description="Tags")


class ImageSearchResponse(BaseModel):
    """图搜图响应"""
    cases: List[SimilarCaseResponse] = Field(..., description="Similar cases")
    related_regulations: Optional[List[RegulationResponse]] = Field(default=None, description="Related regulations")
    search_time_ms: float = Field(..., description="Total search time in milliseconds")
    embedding_time_ms: float = Field(default=0.0, description="Image embedding time in milliseconds")


class ImageWithRegulationsRequest(BaseModel):
    """图像+法规联合检索请求"""
    image_base64: str = Field(..., description="Base64 encoded image")
    top_k_cases: Optional[int] = Field(default=None, ge=1, le=20, description="Number of cases to return (default from config)")
    top_k_regulations: Optional[int] = Field(default=None, ge=1, le=10, description="Number of regulations to return (default from config)")
    
    def get_top_k_cases(self) -> int:
        return self.top_k_cases if self.top_k_cases is not None else _retrieval_config.image_top_k
    
    def get_top_k_regulations(self) -> int:
        return self.top_k_regulations if self.top_k_regulations is not None else _retrieval_config.kb_top_k


class AnalyzeRequest(BaseModel):
    """RAG + VLM 融合分析请求"""
    image_base64: str = Field(..., description="Base64 encoded image")
    query: Optional[str] = Field(default=None, description="Custom question (optional, uses default prompt if not provided)")
    top_k_cases: Optional[int] = Field(default=None, ge=1, le=10, description="Number of cases to retrieve")
    top_k_regulations: Optional[int] = Field(default=None, ge=1, le=10, description="Number of regulations to retrieve")
