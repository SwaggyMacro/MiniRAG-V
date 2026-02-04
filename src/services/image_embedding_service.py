# -*- coding: utf-8 -*-
"""
图像向量提取服务

使用 Chinese-CLIP 提取图像的 512 维向量表示。
"""

import warnings
import os

# 抑制 HuggingFace transformers 警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 抑制 transformers 日志
import transformers
transformers.logging.set_verbosity_error()

import gc
import logging
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import torch
from PIL import Image

from src.config import VLMConfig, get_vlm_config

logger = logging.getLogger(__name__)


@dataclass
class ImageEmbedding:
    """图像向量提取结果"""
    vector: List[float]
    extraction_time_ms: float


class ImageEmbeddingService:
    """
    图像向量提取服务
    
    使用 Chinese-CLIP 模型提取图像的 512 维向量表示，
    用于 Milvus 相似度检索。
    """
    
    _instance: Optional["ImageEmbeddingService"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        初始化图像向量提取服务
        
        Args:
            config: VLM 配置，如果为 None 则从 config.yaml 加载
        """
        self._config = config or get_vlm_config()
        self._vision_model = None
        self._processor = None
        self._device: Optional[str] = None
        self._loaded = False
        self._load_lock = threading.Lock()
    
    def load_model(self) -> None:
        """加载 Chinese-CLIP 视觉编码器"""
        if self._loaded:
            logger.debug("Image embedding model already loaded")
            return
        
        with self._load_lock:
            if self._loaded:
                return
            
            logger.info("Loading Chinese-CLIP for image embedding...")
            start_time = time.time()
            
            try:
                from transformers import CLIPModel, CLIPProcessor
                
                self._device = self._config.device
                logger.info(f"Using device: {self._device}")
                
                # 加载 Chinese-CLIP 模型
                vision_model_path = self._config.vision_model_path
                self._vision_model = CLIPModel.from_pretrained(vision_model_path)
                self._processor = CLIPProcessor.from_pretrained(vision_model_path)
                
                # 冻结参数并设置为评估模式
                for param in self._vision_model.parameters():
                    param.requires_grad = False
                self._vision_model = self._vision_model.eval().to(self._device)
                
                self._loaded = True
                elapsed = time.time() - start_time
                logger.info(f"Chinese-CLIP loaded in {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to load Chinese-CLIP: {e}")
                raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def extract_embedding(self, image_data: bytes) -> ImageEmbedding:
        """
        提取单张图像的向量表示
        
        Args:
            image_data: Base64 解码后的图像字节数据
            
        Returns:
            ImageEmbedding 包含 512 维向量
        """
        if not self._loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # 1. 加载图像
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # 2. 预处理
            inputs = self._processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self._device)
            
            # 3. 提取图像特征向量
            with torch.no_grad():
                image_features = self._vision_model.get_image_features(pixel_values=pixel_values)
                # 归一化
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 4. 转换为列表
            vector = image_features.squeeze().cpu().tolist()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return ImageEmbedding(
                vector=vector,
                extraction_time_ms=elapsed_ms
            )
            
        except Exception as e:
            logger.error(f"Image embedding extraction failed: {e}")
            raise
        
        finally:
            if 'inputs' in locals():
                del inputs
            if 'pixel_values' in locals():
                del pixel_values
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def extract_embedding_batch(self, images_data: List[bytes]) -> List[ImageEmbedding]:
        """
        批量提取图像向量
        
        Args:
            images_data: 图像字节数据列表
            
        Returns:
            ImageEmbedding 列表
        """
        if not self._loaded:
            self.load_model()
        
        results = []
        for image_data in images_data:
            result = self.extract_embedding(image_data)
            results.append(result)
        
        return results
    
    def unload_model(self) -> None:
        """卸载模型，释放资源"""
        if not self._loaded:
            return
        
        with self._load_lock:
            if self._vision_model is not None:
                del self._vision_model
                self._vision_model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            self._loaded = False
            logger.info("Image embedding model unloaded")


# 单例获取函数
_embedding_service_instance: Optional[ImageEmbeddingService] = None
_embedding_instance_lock = threading.Lock()


def get_image_embedding_service(config: Optional[VLMConfig] = None) -> ImageEmbeddingService:
    """
    获取图像向量提取服务单例
    
    Args:
        config: 可选的配置，仅在首次创建时生效
        
    Returns:
        ImageEmbeddingService 单例实例
    """
    global _embedding_service_instance
    
    if _embedding_service_instance is None:
        with _embedding_instance_lock:
            if _embedding_service_instance is None:
                _embedding_service_instance = ImageEmbeddingService(config)
    
    return _embedding_service_instance
