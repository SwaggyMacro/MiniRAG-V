# -*- coding: utf-8 -*-
"""
MilvusClient 主入口模块
提供统一的 Milvus 操作接口。
"""

import logging
from typing import List, Optional, Dict, Any

from pymilvus import Collection

from src.config import MilvusConfig, get_milvus_config

from src.milvus.connection import MilvusConnection
from src.milvus.manager import CollectionManager
from src.milvus.model import CollectionInfo, CollectionStats, IMAGE_COLLECTION_NAME, KB_COLLECTION_NAME

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self, config_path: Optional[str] = None):
        self._milvus_config: MilvusConfig = get_milvus_config(config_path)
        
        self._connection: MilvusConnection = MilvusConnection(
            host=self._milvus_config.host,
            port=self._milvus_config.port,
            retry_attempts=self._milvus_config.retry_attempts,
            retry_delay=self._milvus_config.retry_delay
        )
        self._collection_manager: Optional[CollectionManager] = None
       
    def __enter__(self) -> "MilvusClient":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    def connect(self) -> None:
        self._connection.connect()
        # CollectionManager 初始化依赖于已建立的连接上下文
        self._collection_manager = CollectionManager(
            config=self._milvus_config,
            using=self._connection.alias
        )
    
    def disconnect(self) -> None:
        self._connection.disconnect()
        self._collection_manager = None
    
    def is_connected(self) -> bool:
        return self._connection.is_connected()
    
    def reconnect(self) -> None:
        self._connection.reconnect()
        self._collection_manager = CollectionManager(
            config=self._milvus_config,
            using=self._connection.alias
        )
    
    def get_server_version(self) -> str:
        return self._connection.get_server_version()
    
    # ========================================================================
    # Collection 创建与管理
    # ========================================================================
    
    def _ensure_connected(self) -> CollectionManager:
        if self._collection_manager is None:
            raise Exception(
                f"Not connected to Milvus ({self._milvus_config.host}:{self._milvus_config.port}). Call connect() first."
            )
        return self._collection_manager
    
    def create_image_collection(self) -> CollectionInfo:
        """创建图像案例库 Collection"""
        manager = self._ensure_connected()
        return manager.create_image_collection()
    
    def create_kb_collection(self) -> CollectionInfo:
        """创建文本知识库 Collection"""
        manager = self._ensure_connected()
        return manager.create_kb_collection()
    
    def ensure_collections_exist(self) -> dict:
        """确保两个核心 Collection 都存在"""
        manager = self._ensure_connected()
        return manager.ensure_collections_exist()
    
    def list_collections(self) -> List[str]:
        """列出所有 Collection"""
        manager = self._ensure_connected()
        return manager.list_collections()
    
    def collection_exists(self, name: str) -> bool:
        """检查 Collection 是否存在"""
        manager = self._ensure_connected()
        return manager.collection_exists(name)
    
    def get_collection_stats(self, name: str) -> Optional[CollectionStats]:
        """获取 Collection 统计信息"""
        manager = self._ensure_connected()
        return manager.get_collection_stats(name)
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """获取 Collection 对象"""
        manager = self._ensure_connected()
        return manager.get_collection(name)
    
    def drop_collection(self, name: str, confirm: bool = False) -> bool:
        """删除 Collection"""
        manager = self._ensure_connected()
        return manager.drop_collection(name, confirm=confirm)
      
    def insert(self, collection_name: str, data: List[dict]) -> List:
        manager = self._ensure_connected()
        collection = manager.get_collection(collection_name)
        
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            # 插入数据
            result = collection.insert(data)
            # 刷新以持久化
            collection.flush()
            logger.info(f"Inserted {len(data)} records into '{collection_name}'")
            return result.primary_keys
        except Exception as e:
            logger.error(f"Insert failed for '{collection_name}': {e}")
            raise e
    
    def search_similar_cases(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        业务检索包装：以图搜图（搜案例）
        """
        manager = self._ensure_connected()
        return manager.search_similar_cases(vector, top_k, expr)

    def search_related_regulations(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        业务检索包装：以文搜文（搜法规）
        """
        manager = self._ensure_connected()
        return manager.search_related_regulations(vector, top_k, expr)

    def query_regulations_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        业务检索包装：根据ID查法规
        """
        manager = self._ensure_connected()
        return manager.query_regulations_by_ids(ids)

    def query_case_by_id(self, case_id: int) -> Optional[Dict[str, Any]]:
        """
        业务检索包装：根据ID查案例
        """
        manager = self._ensure_connected()
        return manager.query_case_by_id(case_id)
