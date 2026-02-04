import base64
import logging
from typing import Optional, List

from .base import BaseService
from src.api.schemas import ExplainRequest, ExplainResponse
from .vlm_inference_service import VLMInferenceService, get_vlm_service_instance
from .search_service import SearchService, get_search_service, SearchResult
from src.config import VLMConfig, get_vlm_config, get_default_retrieval_config

logger = logging.getLogger(__name__)


class AIService(BaseService):
    """
    Service responsible for AI capabilities and explanations.
    
    Integrates VLM (Vision Language Model) for image analysis
    and RAG (Retrieval Augmented Generation) for enhanced accuracy.
    """

    def __init__(self):
        self._vlm_service: Optional[VLMInferenceService] = None
        self._vlm_config: Optional[VLMConfig] = None
        self._search_service: Optional[SearchService] = None
    
    def _get_vlm_config(self) -> VLMConfig:
        """懒加载 VLM 配置"""
        if self._vlm_config is None:
            self._vlm_config = get_vlm_config()
        return self._vlm_config
    
    def _get_vlm_service(self) -> VLMInferenceService:
        """懒加载 VLM 服务"""
        if self._vlm_service is None:
            self._vlm_service = get_vlm_service_instance()
        return self._vlm_service
    
    def _get_search_service(self) -> SearchService:
        """懒加载检索服务"""
        if self._search_service is None:
            self._search_service = get_search_service()
        return self._search_service
    
    def _clean_base64(self, image_base64: str) -> bytes:
        """清理 Base64 字符串并解码"""
        clean_b64 = image_base64
        
        # 移除 Data URI 前缀
        if ',' in clean_b64:
            clean_b64 = clean_b64.split(',', 1)[1]
        
        # 移除换行符、空格
        clean_b64 = clean_b64.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # 修复 Base64 填充
        remainder = len(clean_b64) % 4
        if remainder:
            clean_b64 += '=' * (4 - remainder)
        
        return base64.b64decode(clean_b64)
    
    def _build_rag_prompt(
        self, 
        base_query: str, 
        evidence: SearchResult
    ) -> str:
        """
        构建 RAG 增强的 Prompt
        
        将检索到的案例和法规注入 Prompt，辅助 VLM 生成。
        """
        prompt_parts = []
        
        # 添加参考案例
        if evidence.cases:
            cases_text = []
            for i, case in enumerate(evidence.cases[:3], 1):
                case_desc = case.description[:80] if case.description else ""
                cases_text.append(f"{i}. [{case.accident_type}] {case_desc}")
            
            prompt_parts.append("【相似案例参考】")
            prompt_parts.append("\n".join(cases_text))
        
        # 添加参考法规
        if evidence.related_regulations:
            regs_text = []
            for i, reg in enumerate(evidence.related_regulations[:3], 1):
                content = reg.content[:100] if reg.content else ""
                regs_text.append(f"{i}. {reg.reference}: {content}")
            
            prompt_parts.append("\n【相关法规】")
            prompt_parts.append("\n".join(regs_text))
        
        # 添加用户问题
        prompt_parts.append(f"\n【任务】{base_query}")
        
        return "\n".join(prompt_parts)

    async def explain_query(self, request: ExplainRequest) -> ExplainResponse:
        """
        处理解释查询请求（无 RAG）
        
        如果提供了 image_base64，将使用 VLM 进行图像分析。
        如果未提供 query，将使用配置中的默认 prompt。
        """
        image_analysis: Optional[str] = None
        inference_time_ms: Optional[float] = None
        
        vlm_config = self._get_vlm_config()
        prompt = request.query or vlm_config.default_prompt
        
        if request.image_base64:
            try:
                vlm_service = self._get_vlm_service()
                image_bytes = self._clean_base64(request.image_base64)
                vlm_result = vlm_service.analyze_image(image_bytes, prompt)
                image_analysis = vlm_result.generated_text
                inference_time_ms = vlm_result.inference_time_ms
                logger.info(f"VLM analysis completed in {inference_time_ms:.2f}ms")
            except Exception as e:
                logger.error(f"VLM analysis failed: {e}")
                raise
        
        explanation_text = image_analysis or f"Explanation for: {prompt}"
        confidence = 0.85 if image_analysis else 0.99
        
        return ExplainResponse(
            original_query=prompt,
            explanation=explanation_text,
            confidence=confidence,
            image_analysis=image_analysis,
            inference_time_ms=inference_time_ms
        )
    
    async def analyze_with_rag(
        self, 
        image_base64: str, 
        query: Optional[str] = None,
        top_k_cases: Optional[int] = None,
        top_k_regulations: Optional[int] = None
    ) -> ExplainResponse:
        """
        RAG + VLM 融合分析
        
        1. 检索相似案例和相关法规
        2. 构建 Evidence Pack 注入 Prompt
        3. VLM 生成增强回答（降低幻觉）
        
        Args:
            image_base64: Base64 编码的图像
            query: 用户问题，可选
            top_k_cases: 检索案例数量，可选
            top_k_regulations: 检索法规数量，可选
            
        Returns:
            ExplainResponse 包含 RAG 增强的分析结果
        """
        import time
        start_time = time.time()
        
        vlm_config = self._get_vlm_config()
        retrieval_config = get_default_retrieval_config()
        
        base_query = query or vlm_config.default_prompt
        cases_k = top_k_cases or retrieval_config.image_top_k
        regs_k = top_k_regulations or retrieval_config.kb_top_k
        
        try:
            # 1. 检索相似案例和法规
            logger.info("RAG: 检索相似案例和法规...")
            search_service = self._get_search_service()
            evidence = search_service.search_with_semantic_regulations(
                image_base64=image_base64,
                top_k_cases=cases_k,
                top_k_regulations=regs_k
            )
            logger.info(f"RAG: 检索到 {len(evidence.cases)} 个案例, {len(evidence.related_regulations or [])} 条法规")
            
            # 2. 构建 RAG Prompt
            rag_prompt = self._build_rag_prompt(base_query, evidence)
            logger.debug(f"RAG Prompt: {rag_prompt[:200]}...")
            
            # 3. VLM 生成
            logger.info("RAG: 调用 VLM 生成分析...")
            vlm_service = self._get_vlm_service()
            image_bytes = self._clean_base64(image_base64)
            vlm_result = vlm_service.analyze_image(image_bytes, rag_prompt)
            
            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"RAG + VLM 分析完成，耗时 {total_time_ms:.2f}ms")
            
            # 4. 构建响应
            return ExplainResponse(
                original_query=base_query,
                explanation=vlm_result.generated_text,
                confidence=0.90,  # RAG 增强后置信度更高
                image_analysis=vlm_result.generated_text,
                inference_time_ms=total_time_ms
            )
            
        except Exception as e:
            logger.error(f"RAG + VLM analysis failed: {e}")
            raise
