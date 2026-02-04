# -*- coding: utf-8 -*-
"""
RAG 搜索服务

整合图像向量提取和 Milvus 检索，实现以图搜图功能。
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.services.base import BaseService
from src.services.image_embedding_service import (
    ImageEmbeddingService,
    get_image_embedding_service
)
from src.milvus.client import MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class SimilarCase:
    """相似案例"""
    case_id: int
    score: float
    description: str
    accident_type: str
    image_path: Optional[str] = None
    legal_basis_ids: List[int] = field(default_factory=list)


@dataclass
class RegulationInfo:
    """法规信息"""
    regulation_id: int
    content: str
    reference: str
    source_type: str
    tags: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """搜索结果"""
    cases: List[SimilarCase]
    related_regulations: Optional[List[RegulationInfo]] = None
    search_time_ms: float = 0.0
    embedding_time_ms: float = 0.0


class SearchService(BaseService):
    """
    RAG 搜索服务
    
    整合图像向量提取和 Milvus 相似案例检索。
    """
    
    def __init__(self):
        self._embedding_service: Optional[ImageEmbeddingService] = None
        self._milvus_client: Optional[MilvusClient] = None
    
    def _get_embedding_service(self) -> ImageEmbeddingService:
        """懒加载图像向量提取服务"""
        if self._embedding_service is None:
            self._embedding_service = get_image_embedding_service()
        return self._embedding_service
    
    def _get_milvus_client(self) -> MilvusClient:
        """懒加载 Milvus 客户端"""
        if self._milvus_client is None:
            self._milvus_client = MilvusClient()
            self._milvus_client.connect()
        return self._milvus_client
    
    def _clean_base64(self, image_base64: str) -> bytes:
        """清理 Base64 字符串并解码"""
        clean_b64 = image_base64
        
        # 移除 Data URI 前缀
        if ',' in clean_b64:
            clean_b64 = clean_b64.split(',', 1)[1]
        
        # 移除换行符、空格
        clean_b64 = clean_b64.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # 修复填充
        remainder = len(clean_b64) % 4
        if remainder:
            clean_b64 += '=' * (4 - remainder)
        
        return base64.b64decode(clean_b64)
    
    def search_similar_cases(
        self,
        image_base64: str,
        top_k: int = 5,
        include_regulations: bool = False
    ) -> SearchResult:
        """
        以图搜图 - 检索相似案例
        
        Args:
            image_base64: Base64 编码的图像
            top_k: 返回结果数量
            include_regulations: 是否包含关联法规
            
        Returns:
            SearchResult 包含相似案例和可选的关联法规
        """
        total_start = time.time()
        
        try:
            # 1. 解码图像
            image_data = self._clean_base64(image_base64)
            
            # 2. 提取图像向量
            embedding_service = self._get_embedding_service()
            embedding_result = embedding_service.extract_embedding(image_data)
            embedding_time_ms = embedding_result.extraction_time_ms
            
            # 3. Milvus 相似案例检索
            milvus_client = self._get_milvus_client()
            raw_cases = milvus_client.search_similar_cases(
                vector=embedding_result.vector,
                top_k=top_k
            )
            
            # 4. 构建相似案例列表
            cases = []
            all_legal_basis_ids = set()
            
            for raw in raw_cases:
                legal_ids = raw.get("legal_basis_ids") or []
                if isinstance(legal_ids, str):
                    # 如果存储为字符串，尝试解析
                    try:
                        import json
                        legal_ids = json.loads(legal_ids)
                    except:
                        legal_ids = []
                
                case = SimilarCase(
                    case_id=raw.get("id", 0),
                    score=raw.get("score", 0.0),
                    description=raw.get("description", ""),
                    accident_type=raw.get("accident_type", ""),
                    image_path=raw.get("image_path"),
                    legal_basis_ids=legal_ids
                )
                cases.append(case)
                all_legal_basis_ids.update(legal_ids)
            
            # 5. [可选] 查询关联法规
            regulations = None
            if include_regulations and all_legal_basis_ids:
                raw_regulations = milvus_client.query_regulations_by_ids(
                    list(all_legal_basis_ids)
                )
                regulations = [
                    RegulationInfo(
                        regulation_id=reg.get("id", 0),
                        content=reg.get("content", ""),
                        reference=reg.get("reference", ""),
                        source_type=reg.get("source_type", ""),
                        tags=reg.get("tags") or []
                    )
                    for reg in raw_regulations
                ]
            
            total_time_ms = (time.time() - total_start) * 1000
            
            return SearchResult(
                cases=cases,
                related_regulations=regulations,
                search_time_ms=total_time_ms,
                embedding_time_ms=embedding_time_ms
            )
            
        except Exception as e:
            logger.error(f"Search similar cases failed: {e}")
            raise
    
    def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 5
    ) -> SearchResult:
        """
        使用已有向量检索相似案例
        
        Args:
            vector: 512 维图像向量
            top_k: 返回结果数量
            
        Returns:
            SearchResult 包含相似案例
        """
        start_time = time.time()
        
        try:
            milvus_client = self._get_milvus_client()
            raw_cases = milvus_client.search_similar_cases(
                vector=vector,
                top_k=top_k
            )
            
            cases = [
                SimilarCase(
                    case_id=raw.get("id", 0),
                    score=raw.get("score", 0.0),
                    description=raw.get("description", ""),
                    accident_type=raw.get("accident_type", ""),
                    image_path=raw.get("image_path"),
                    legal_basis_ids=raw.get("legal_basis_ids") or []
                )
                for raw in raw_cases
            ]
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return SearchResult(
                cases=cases,
                search_time_ms=elapsed_ms
            )
            
        except Exception as e:
            logger.error(f"Search by vector failed: {e}")
            raise
    
    def search_with_semantic_regulations(
        self,
        image_base64: str,
        top_k_cases: int = 5,
        top_k_regulations: int = 3
    ) -> SearchResult:
        """
        图像+法规联合检索（语义匹配）
        
        上传图片 → 检索相似案例 → 基于案例描述语义匹配相关法规
        
        Args:
            image_base64: Base64 编码的图像
            top_k_cases: 返回案例数量
            top_k_regulations: 返回法规数量
            
        Returns:
            SearchResult 包含相似案例和语义匹配的法规
        """
        from src.services.text_embedding_service import get_text_embedding_service
        
        total_start = time.time()
        
        try:
            # 1. 以图搜图
            image_data = self._clean_base64(image_base64)
            embedding_service = self._get_embedding_service()
            embedding_result = embedding_service.extract_embedding(image_data)
            embedding_time_ms = embedding_result.extraction_time_ms
            
            milvus_client = self._get_milvus_client()
            raw_cases = milvus_client.search_similar_cases(
                vector=embedding_result.vector,
                top_k=top_k_cases
            )
            
            # 2. 构建案例列表
            cases = []
            descriptions = []
            
            for raw in raw_cases:
                legal_ids = raw.get("legal_basis_ids") or []
                case = SimilarCase(
                    case_id=raw.get("id", 0),
                    score=raw.get("score", 0.0),
                    description=raw.get("description", ""),
                    accident_type=raw.get("accident_type", ""),
                    image_path=raw.get("image_path"),
                    legal_basis_ids=legal_ids
                )
                cases.append(case)
                if case.description:
                    descriptions.append(case.description)
            
            # 3. 基于案例描述语义匹配法规
            regulations = []
            if descriptions and top_k_regulations > 0:
                # 合并描述文本
                combined_text = " ".join(descriptions[:3])  # 取前3个描述
                if len(combined_text) > 500:
                    combined_text = combined_text[:500]
                
                # 提取文本向量
                text_service = get_text_embedding_service()
                text_embedding = text_service.extract_embedding(combined_text)
                
                # 检索法规
                raw_regulations = milvus_client.search_related_regulations(
                    vector=text_embedding.vector,
                    top_k=top_k_regulations
                )
                
                regulations = [
                    RegulationInfo(
                        regulation_id=reg.get("id", 0),
                        content=reg.get("content", ""),
                        reference=reg.get("reference", ""),
                        source_type=reg.get("source_type", ""),
                        tags=reg.get("tags") or []
                    )
                    for reg in raw_regulations
                ]
            
            total_time_ms = (time.time() - total_start) * 1000
            
            return SearchResult(
                cases=cases,
                related_regulations=regulations if regulations else None,
                search_time_ms=total_time_ms,
                embedding_time_ms=embedding_time_ms
            )
            
        except Exception as e:
            logger.error(f"Search with semantic regulations failed: {e}")
            raise


# 单例相关
_search_service_instance: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """获取搜索服务单例"""
    global _search_service_instance
    
    if _search_service_instance is None:
        _search_service_instance = SearchService()
    
    return _search_service_instance
