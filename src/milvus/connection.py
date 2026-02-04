# -*- coding: utf-8 -*-
"""
Milvus 连接管理模块

负责建立、维护和管理与 Milvus 向量数据库的连接。
支持重试机制、健康检查和上下文管理器模式。
"""

import time
import logging
from typing import Optional

from pymilvus import connections, utility
from pymilvus.exceptions import MilvusException

# 配置日志
logger = logging.getLogger(__name__)


class MilvusConnection: 
    DEFAULT_ALIAS = "default"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        alias: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._alias = alias or self.DEFAULT_ALIAS
        self._connected = False
    
    @property
    def alias(self) -> str:
        return self._alias
    
    def __enter__(self) -> "MilvusConnection":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    def connect(self) -> None:
        if self._connected:
            logger.debug(f"Already connected to Milvus (alias={self._alias})")
            return
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.info(
                    f"Connecting to Milvus (attempt {attempt}/{self.retry_attempts}): "
                    f"host={self.host}, port={self.port}"
                )
                
                connections.connect(
                    alias=self._alias,
                    host=self.host,
                    port=self.port
                )
                
                self._connected = True
                logger.info(f"Successfully connected to Milvus (alias={self._alias})")
                return
                
            except MilvusException as e:
                last_error = e
                error_msg = str(e).lower()
                
                if "auth" in error_msg or "permission" in error_msg:
                    raise Exception(f"Milvus 认证失败: {e}")
                
                logger.warning(
                    f"Connection attempt {attempt} failed: {e}. "
                    f"Retrying in {self.retry_delay}s..."
                )
                
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
            
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Connection attempt {attempt} failed with unexpected error: {e}. "
                    f"Retrying in {self.retry_delay}s..."
                )
                
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
        
        raise Exception(
            f"Milvus 连接失败 ({self.host}:{self.port}): {last_error}"
        )
    
    def disconnect(self) -> None:
        if not self._connected:
            logger.debug(f"Not connected, skip disconnect (alias={self._alias})")
            return
        
        try:
            connections.disconnect(alias=self._alias)
            self._connected = False
            logger.info(f"Disconnected from Milvus (alias={self._alias})")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
            self._connected = False
    
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        
        try:
            # 尝试列出 collections 来验证连接
            utility.list_collections(using=self._alias)
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            self._connected = False
            return False
    
    def reconnect(self) -> None:
        self.disconnect()
        self.connect()
    
    def get_server_version(self) -> str:
        if not self.is_connected():
            raise Exception("Milvus 未连接")
        
        try:
            return utility.get_server_version(using=self._alias)
        except Exception as e:
            raise Exception(f"获取 Milvus 版本失败: {e}")
