# -*- coding: utf-8 -*-
"""
文本向量提取服务

使用 Chinese-CLIP 文本编码器提取 512 维文本向量。
"""

import warnings
import os

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
transformers.logging.set_verbosity_error()

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from src.config import VLMConfig, get_vlm_config

logger = logging.getLogger(__name__)


@dataclass
class TextEmbedding:
    """文本向量提取结果"""
    vector: List[float]
    extraction_time_ms: float


class TextEmbeddingService:
    """
    文本向量提取服务
    
    使用 Chinese-CLIP 模型提取文本的 512 维向量表示，
    用于 Milvus 相似度检索。
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        self._config = config or get_vlm_config()
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._loaded = False
        self._load_lock = threading.Lock()
    
    def load_model(self) -> None:
        """加载 Chinese-CLIP 模型"""
        if self._loaded:
            return
        
        with self._load_lock:
            if self._loaded:
                return
            
            logger.info("Loading Chinese-CLIP for text embedding...")
            start_time = time.time()
            
            try:
                from transformers import CLIPModel, CLIPProcessor
                
                self._device = self._config.device
                logger.info(f"Using device: {self._device}")
                
                vision_model_path = self._config.vision_model_path
                self._model = CLIPModel.from_pretrained(vision_model_path)
                self._processor = CLIPProcessor.from_pretrained(vision_model_path)
                
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model = self._model.eval().to(self._device)
                
                self._loaded = True
                elapsed = time.time() - start_time
                logger.info(f"Chinese-CLIP text encoder loaded in {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to load Chinese-CLIP: {e}")
                raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def extract_embedding(self, text: str) -> TextEmbedding:
        """
        提取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            TextEmbedding 包含 512 维向量
        """
        if not self._loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # 预处理
            inputs = self._processor(
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=77  # CLIP 默认最大长度
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 提取文本特征
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            vector = text_features.squeeze().cpu().tolist()
            elapsed_ms = (time.time() - start_time) * 1000
            
            return TextEmbedding(
                vector=vector,
                extraction_time_ms=elapsed_ms
            )
            
        except Exception as e:
            logger.error(f"Text embedding extraction failed: {e}")
            raise
        
        finally:
            if 'inputs' in locals():
                del inputs
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def extract_embedding_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """批量提取文本向量"""
        if not self._loaded:
            self.load_model()
        
        results = []
        for text in texts:
            result = self.extract_embedding(text)
            results.append(result)
        
        return results
    
    def unload_model(self) -> None:
        """卸载模型"""
        if not self._loaded:
            return
        
        with self._load_lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            self._loaded = False
            logger.info("Text embedding model unloaded")


# 单例
_text_embedding_service_instance: Optional[TextEmbeddingService] = None
_text_embedding_instance_lock = threading.Lock()


def get_text_embedding_service(config: Optional[VLMConfig] = None) -> TextEmbeddingService:
    """获取文本向量提取服务单例"""
    global _text_embedding_service_instance
    
    if _text_embedding_service_instance is None:
        with _text_embedding_instance_lock:
            if _text_embedding_service_instance is None:
                _text_embedding_service_instance = TextEmbeddingService(config)
    
    return _text_embedding_service_instance
