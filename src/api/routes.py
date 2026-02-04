from fastapi import APIRouter, Depends, HTTPException
from src.services import SystemService, AIService, get_search_service, SearchService
from src.api.schemas import (
    InfoResponse, 
    ExplainRequest, 
    ExplainResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    SimilarCaseResponse,
    RegulationResponse,
    ImageWithRegulationsRequest,
    AnalyzeRequest
)
from src.api.responses import StandardResponse

router = APIRouter()

# Dependency providers
def get_system_service() -> SystemService:
    return SystemService()

def get_ai_service() -> AIService:
    return AIService()


@router.get("/info", response_model=StandardResponse[InfoResponse])
async def get_info(service: SystemService = Depends(get_system_service)):
    try:
        data = await service.get_info()
        return StandardResponse(data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=StandardResponse[ExplainResponse])
async def explain(
    request: ExplainRequest, 
    service: AIService = Depends(get_ai_service)
):
    try:
        data = await service.explain_query(request)
        return StandardResponse(data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/image", response_model=StandardResponse[ImageSearchResponse])
async def search_by_image(
    request: ImageSearchRequest,
    service: SearchService = Depends(get_search_service)
):
    """
    以图搜图 - 检索相似案例
    
    上传图片，返回 Milvus 中最相似的案例列表。
    可选包含关联的法规信息。
    """
    try:
        result = service.search_similar_cases(
            image_base64=request.image_base64,
            top_k=request.get_top_k(),
            include_regulations=request.include_regulations
        )
        
        # 转换为响应模型
        cases = [
            SimilarCaseResponse(
                case_id=c.case_id,
                score=c.score,
                description=c.description,
                accident_type=c.accident_type,
                image_path=c.image_path,
                legal_basis_ids=c.legal_basis_ids
            )
            for c in result.cases
        ]
        
        regulations = None
        if result.related_regulations:
            regulations = [
                RegulationResponse(
                    regulation_id=r.regulation_id,
                    content=r.content,
                    reference=r.reference,
                    source_type=r.source_type,
                    tags=r.tags
                )
                for r in result.related_regulations
            ]
        
        response_data = ImageSearchResponse(
            cases=cases,
            related_regulations=regulations,
            search_time_ms=result.search_time_ms,
            embedding_time_ms=result.embedding_time_ms
        )
        
        return StandardResponse(data=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/image-with-regulations", response_model=StandardResponse[ImageSearchResponse])
async def search_image_with_regulations(
    request: ImageWithRegulationsRequest,
    service: SearchService = Depends(get_search_service)
):
    """
    图像+法规联合检索
    
    上传图片，返回相似案例和语义匹配的相关法规。
    """
    try:
        result = service.search_with_semantic_regulations(
            image_base64=request.image_base64,
            top_k_cases=request.get_top_k_cases(),
            top_k_regulations=request.get_top_k_regulations()
        )
        
        cases = [
            SimilarCaseResponse(
                case_id=c.case_id,
                score=c.score,
                description=c.description,
                accident_type=c.accident_type,
                image_path=c.image_path,
                legal_basis_ids=c.legal_basis_ids
            )
            for c in result.cases
        ]
        
        regulations = None
        if result.related_regulations:
            regulations = [
                RegulationResponse(
                    regulation_id=r.regulation_id,
                    content=r.content,
                    reference=r.reference,
                    source_type=r.source_type,
                    tags=r.tags
                )
                for r in result.related_regulations
            ]
        
        response_data = ImageSearchResponse(
            cases=cases,
            related_regulations=regulations,
            search_time_ms=result.search_time_ms,
            embedding_time_ms=result.embedding_time_ms
        )
        
        return StandardResponse(data=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=StandardResponse[ExplainResponse])
async def analyze_with_rag(
    request: AnalyzeRequest,
    service: AIService = Depends(get_ai_service)
):
    """
    RAG + VLM 融合分析
    
    上传图片，检索相似案例和法规作为上下文，辅助 VLM 生成更准确的分析。
    """
    try:
        result = await service.analyze_with_rag(
            image_base64=request.image_base64,
            query=request.query,
            top_k_cases=request.top_k_cases,
            top_k_regulations=request.top_k_regulations
        )
        return StandardResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
