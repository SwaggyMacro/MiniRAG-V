# -*- coding: utf-8 -*-
"""
Collection 管理模块

负责 Milvus Collection 的创建、删除、查询和索引管理。
同时封装了针对交通案例和法规的特定检索逻辑。
"""

import logging
from typing import List, Optional, Dict, Any

from pymilvus import Collection, utility, CollectionSchema

from src.config import MilvusConfig, get_milvus_config

from src.milvus.model import (
    CollectionInfo,
    CollectionStats,
    EMBEDDING_DIM,
    IMAGE_COLLECTION_NAME,
    IMAGE_COLLECTION_DESCRIPTION,
    KB_COLLECTION_NAME,
    KB_COLLECTION_DESCRIPTION,
    get_image_collection_schema,
    get_kb_collection_schema,
    get_hnsw_index_params,
)

logger = logging.getLogger(__name__)

class CollectionManager:
    def __init__(
        self,
        config: Optional[MilvusConfig] = None,
        config_path: Optional[str] = None,
        using: str = "default"
    ):
        if config is not None:
            self._config = config
        else:
            self._config = get_milvus_config(config_path)
        
        self._using = using
    
    def _create_collection(
        self,
        name: str,
        schema: CollectionSchema,
        description: str
    ) -> CollectionInfo:
        # 检查是否已存在
        if utility.has_collection(name, using=self._using):
            logger.info(f"Collection '{name}' already exists, skipping creation")
            collection = Collection(name=name, using=self._using)
            
            # 确保已加载
            try:
                collection.load()
            except Exception as e:
                logger.warning(f"Failed to load existing collection '{name}': {e}")
            
            return CollectionInfo(
                name=name,
                description=description,
                num_fields=len(schema.fields),
                vector_dim=EMBEDDING_DIM,
                index_type="HNSW",
                metric_type="COSINE",
                created=False
            )
        
        logger.info(f"Creating collection '{name}'...")
        
        try:
            collection = Collection(
                name=name,
                schema=schema,
                using=self._using
            )
        except Exception as e:
            raise Exception(f"创建 Collection '{name}' 失败: {e}")
        
        self._create_index(collection, name)
        
        logger.info(f"Loading collection '{name}' into memory...")
        collection.load()
        
        logger.info(f"Collection '{name}' created and loaded successfully")
        
        return CollectionInfo(
            name=name,
            description=description,
            num_fields=len(schema.fields),
            vector_dim=EMBEDDING_DIM,
            index_type="HNSW",
            metric_type="COSINE",
            created=True
        )
    
    def _create_index(self, collection: Collection, name: str) -> None:
        index_params = get_hnsw_index_params(
            m=self._config.retry_attempts * 5 + 1, # 这里只是为了使用config中的值，实际应该在config中定义index参数
            ef_construction=256 # 默认值
        )
        # 修正：直接使用默认参数，因为 config 暂时没有深度的 index 字典
        index_params = get_hnsw_index_params()
        
        # 确定向量字段名
        vector_field = "visual_vector"
        for field in collection.schema.fields:
            if field.dtype == 101: # FLOAT_VECTOR
                vector_field = field.name
                break

        logger.info(f"Creating HNSW index for '{name}' on field '{vector_field}'")
        
        try:
            collection.create_index(
                field_name=vector_field,
                index_params=index_params
            )
        except Exception as e:
            raise Exception(f"创建索引 '{name}' 失败: {e}")
    
    def create_image_collection(self) -> CollectionInfo:
        schema = get_image_collection_schema()
        return self._create_collection(IMAGE_COLLECTION_NAME, schema, IMAGE_COLLECTION_DESCRIPTION)
    
    def create_kb_collection(self) -> CollectionInfo:
        schema = get_kb_collection_schema()
        return self._create_collection(KB_COLLECTION_NAME, schema, KB_COLLECTION_DESCRIPTION)
    
    def search_similar_cases(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        expr: str = None
    ) -> List[Dict[str, Any]]:
        if not self.collection_exists(IMAGE_COLLECTION_NAME):
            logger.error(f"Collection {IMAGE_COLLECTION_NAME} not found.")
            return []

        collection = Collection(IMAGE_COLLECTION_NAME, using=self._using)
        
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        output_fields = ["description", "accident_type", "image_path", "legal_basis_ids"]
        
        try:
            results = collection.search(
                data=[vector], 
                anns_field="visual_vector", 
                param=search_params, 
                limit=top_k, 
                expr=expr,
                output_fields=output_fields
            )
            
            hits = []
            for hit in results[0]:
                hits.append({
                    "id": hit.id,
                    "score": hit.score,
                    "description": hit.entity.get("description"),
                    "accident_type": hit.entity.get("accident_type"),
                    "image_path": hit.entity.get("image_path"),
                    "legal_basis_ids": hit.entity.get("legal_basis_ids")
                })
            return hits
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e

    def search_related_regulations(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        expr: str = None
    ) -> List[Dict[str, Any]]:
        if not self.collection_exists(KB_COLLECTION_NAME):
            logger.error(f"Collection {KB_COLLECTION_NAME} not found.")
            return []

        collection = Collection(KB_COLLECTION_NAME, using=self._using)
        
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        output_fields = ["content", "reference", "source_type", "tags"]
        
        try:
            results = collection.search(
                data=[vector], 
                anns_field="vector", 
                param=search_params, 
                limit=top_k, 
                expr=expr,
                output_fields=output_fields
            )
            
            hits = []
            for hit in results[0]:
                hits.append({
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.entity.get("content"),
                    "reference": hit.entity.get("reference"),
                    "source_type": hit.entity.get("source_type"),
                    "tags": hit.entity.get("tags")
                })
            return hits
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e

    def query_regulations_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
            
        if not self.collection_exists(KB_COLLECTION_NAME):
            logger.error(f"Collection {KB_COLLECTION_NAME} not found.")
            return []

        collection = Collection(KB_COLLECTION_NAME, using=self._using)
        
        expr = f"id in {ids}"
        output_fields = ["id", "content", "reference", "source_type", "tags"]
        
        try:
            results = collection.query(
                expr=expr,
                output_fields=output_fields
            )
            return results
        except Exception as e:
            logger.error(f"Query regulations by IDs failed: {e}")
            raise e

    def query_case_by_id(self, case_id: int) -> Optional[Dict[str, Any]]:
        if not self.collection_exists(IMAGE_COLLECTION_NAME):
            logger.error(f"Collection {IMAGE_COLLECTION_NAME} not found.")
            return None

        collection = Collection(IMAGE_COLLECTION_NAME, using=self._using)
        
        expr = f"id == {case_id}"
        output_fields = ["id", "description", "accident_type", "image_path", "legal_basis_ids"]
        
        try:
            results = collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=1
            )
            if results:
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Query case by ID failed: {e}")
            raise e

    def list_collections(self) -> List[str]:
        return utility.list_collections(using=self._using)
    
    def collection_exists(self, name: str) -> bool:
        return utility.has_collection(name, using=self._using)
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """
        获取 Collection 对象
        
        Args:
            name: Collection 名称
            
        Returns:
            Collection 对象，如果不存在则返回 None
        """
        if not self.collection_exists(name):
            logger.warning(f"Collection '{name}' does not exist")
            return None
        
        collection = Collection(name=name, using=self._using)
        return collection
    
    def ensure_collections_exist(self) -> dict:
        return {
            "image": self.create_image_collection(),
            "kb": self.create_kb_collection()
        }
    
    def drop_collection(self, name: str, confirm: bool = False) -> bool:
        if not confirm:
            return False
        if utility.has_collection(name, using=self._using):
            utility.drop_collection(name, using=self._using)
            return True
        return True
