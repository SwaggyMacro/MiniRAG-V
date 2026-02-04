# -*- coding: utf-8 -*-
"""
VLM 推理服务模块

封装 MiniMind2-V 模型的加载和推理逻辑。
参考 scripts/eval_vlm.py 实现。
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from transformers import AutoTokenizer

from src.inference.model_vlm import MiniMindVLM, VLMConfig as VLMModelConfig
from src.config import VLMConfig, get_vlm_config

logger = logging.getLogger(__name__)


@dataclass
class VLMResult:
    """VLM 推理结果"""
    generated_text: str
    inference_time_ms: float
    model_name: str = "MiniMind2-V"


class VLMInferenceService:
    """
    VLM 推理服务
    
    封装 MiniMind2-V 模型，提供图像分析功能。
    采用单例模式，首次调用时懒加载模型。
    """
    
    _instance: Optional["VLMInferenceService"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        初始化 VLM 推理服务
        
        Args:
            config: VLM 配置，如果为 None 则从 config.yaml 加载
        """
        self._config = config or get_vlm_config()
        self._model: Optional[MiniMindVLM] = None
        self._tokenizer = None
        self._processor = None
        self._device: Optional[str] = None
        self._loaded = False
        self._load_lock = threading.Lock()
    
    def load_model(self) -> None:
        """
        加载 VLM 模型
        
        流程参考 eval_vlm.py:init_model：
        1. 加载 tokenizer
        2. 创建 MiniMindVLM 模型
        3. 加载权重
        4. 动态更新 image_ids
        """
        if self._loaded:
            logger.debug("VLM model already loaded")
            return
        
        with self._load_lock:
            if self._loaded:
                return
            
            logger.info("Loading VLM model...")
            start_time = time.time()
            
            try:
                # 1. 确定设备
                self._device = self._config.device
                logger.info(f"Using device: {self._device}")
                
                # 2. 加载 tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._config.model_path,
                    trust_remote_code=True
                )
                
                # 3. 创建模型
                model_config = VLMModelConfig(
                    hidden_size=self._config.hidden_size,
                    num_hidden_layers=self._config.num_hidden_layers,
                    use_moe=self._config.use_moe
                )
                self._model = MiniMindVLM(
                    params=model_config,
                    vision_model_path=self._config.vision_model_path
                )
                
                # 4. 加载权重
                moe_suffix = '_moe' if self._config.use_moe else ''
                weight_path = f"{self._config.model_path}/{self._config.weight_name}_{self._config.hidden_size}{moe_suffix}.pth"
                
                logger.info(f"Loading weights from: {weight_path}")
                state_dict = torch.load(weight_path, map_location=self._device, weights_only=False)
                self._model.load_state_dict(
                    {k: v for k, v in state_dict.items() if 'mask' not in k},
                    strict=False
                )
                
                # 5. 设置为评估模式并移到目标设备
                self._model = self._model.eval().to(self._device)
                self._processor = self._model.processor
                
                # 6. 动态更新 image_ids（参考 eval_vlm.py）
                if hasattr(self._model, 'params') and hasattr(self._model.params, 'image_special_token'):
                    ids = self._tokenizer.encode(
                        self._model.params.image_special_token,
                        add_special_tokens=False
                    )
                    self._model.params.image_ids = ids
                    
                    if len(ids) != 196:
                        logger.warning(
                            f"Tokenized image placeholder length ({len(ids)}) does not match "
                            f"expected length (196). This might cause alignment issues."
                        )
                
                self._loaded = True
                elapsed = time.time() - start_time
                param_count = sum(p.numel() for p in self._model.parameters() if p.requires_grad) / 1e6
                logger.info(f"VLM model loaded successfully in {elapsed:.2f}s ({param_count:.2f}M parameters)")
                
            except Exception as e:
                logger.error(f"Failed to load VLM model: {e}")
                raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None
    ) -> VLMResult:
        """
        分析图像
        
        Args:
            image_data: Base64 解码后的图像字节数据
            prompt: 提示词，如果为 None 则使用配置中的默认 prompt
            
        Returns:
            VLMResult 推理结果
        """
        # 确保模型已加载
        if not self._loaded:
            self.load_model()
        
        # 使用默认 prompt
        if prompt is None:
            prompt = self._config.default_prompt
        
        start_time = time.time()
        
        try:
            # 1. 加载图像
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # 2. 图像预处理（参考 eval_vlm.py）
            pixel_values = MiniMindVLM.image2tensor(image, self._processor)
            pixel_values = pixel_values.to(self._device).unsqueeze(0)
            
            # 3. 构建 prompt（替换图像占位符）
            image_special_token = self._model.params.image_special_token
            # 如果 prompt 中没有 <image> 占位符，添加到开头
            if '<image>' not in prompt and image_special_token not in prompt:
                full_prompt = f"{image_special_token}\n\n{prompt}"
            else:
                full_prompt = prompt.replace('<image>', image_special_token)
            
            # 4. 构建消息并 tokenize（参考 eval_vlm.py）
            messages = [{"role": "user", "content": full_prompt}]
            inputs_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self._tokenizer(
                inputs_text,
                return_tensors="pt",
                truncation=True
            ).to(self._device)
            
            # 5. 生成
            with torch.no_grad():
                output_ids = self._model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=self._config.do_sample,
                    top_p=self._config.top_p,
                    temperature=self._config.temperature,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pixel_values=pixel_values
                )
            
            # 6. 解码输出（跳过输入部分）
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return VLMResult(
                generated_text=generated_text,
                inference_time_ms=elapsed_ms,
                model_name="MiniMind2-V"
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
        
        finally:
            if 'inputs' in locals():
                del inputs
            if 'pixel_values' in locals():
                del pixel_values
            if 'output_ids' in locals():
                del output_ids
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def unload_model(self) -> None:
        if not self._loaded:
            return
        
        with self._load_lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            self._loaded = False
            logger.info("VLM model unloaded")

_service_instance: Optional[VLMInferenceService] = None
_instance_lock = threading.Lock()


def get_vlm_service_instance(config: Optional[VLMConfig] = None) -> VLMInferenceService:
    """
    获取 VLM 推理服务单例
    """
    global _service_instance
    
    if _service_instance is None:
        with _instance_lock:
            if _service_instance is None:
                _service_instance = VLMInferenceService(config)
    
    return _service_instance
