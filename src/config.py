import yaml
import os
from dataclasses import dataclass
from typing import Optional


def get_project_root() -> str:
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    # src/config.py -> src -> root
    return os.path.dirname(os.path.dirname(current_file))


def _resolve_path(path: str, project_root: str) -> str:
    """
    解析路径，将相对路径转换为绝对路径
    
    Args:
        path: 配置中的路径（可能是相对路径或绝对路径）
        project_root: 项目根目录
        
    Returns:
        绝对路径
    """
    if os.path.isabs(path):
        return path
    # 移除开头的 ./ 如果存在
    if path.startswith('./'):
        path = path[2:]
    return os.path.join(project_root, path)


def _resolve_device(device_config: str) -> str:
    """解析设备配置，auto 时自动检测"""
    if device_config == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def _load_yaml_config(config_path: Optional[str] = None) -> dict:
    """加载 YAML 配置文件"""
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, "config.yaml")
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================================
# Milvus 配置
# ============================================================================

@dataclass
class MilvusConfig:
    """Milvus 向量数据库配置"""
    host: str
    port: int
    retry_attempts: int
    retry_delay: float


def get_milvus_config(config_path: Optional[str] = None) -> MilvusConfig:
    """加载 Milvus 配置"""
    data = _load_yaml_config(config_path)
    milvus_data = data.get("milvus", {})
    
    return MilvusConfig(
        host=milvus_data.get("host", "localhost"),
        port=milvus_data.get("port", 19530),
        retry_attempts=milvus_data.get("retry_attempts", 3),
        retry_delay=milvus_data.get("retry_delay", 1.0)
    )


# ============================================================================
# VLM 配置
# ============================================================================

@dataclass
class VLMConfig:
    """VLM 视觉语言模型配置"""
    model_path: str
    vision_model_path: str
    weight_name: str
    hidden_size: int
    num_hidden_layers: int
    use_moe: bool
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float
    default_prompt: str


def get_vlm_config(config_path: Optional[str] = None) -> VLMConfig:
    """加载 VLM 配置"""
    root = get_project_root()
    data = _load_yaml_config(config_path)
    
    # 默认配置
    defaults = VLMConfig(
        model_path=os.path.join(root, "model/MiniMind2-V"),
        vision_model_path=os.path.join(root, "model/vision_model/chinese-clip-vit-base-patch16"),
        weight_name="sft_vlm",
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=False,
        device="cpu",
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        default_prompt="你是一个图像分析助手。请简洁描述图片中的可见内容，不要推测原因，不要给建议。"
    )
    
    vlm_data = data.get("vlm", {})
    generation_data = vlm_data.get("generation", {})
    prompt_data = vlm_data.get("prompt", {})
    
    # 解析设备配置
    device_config = vlm_data.get("device", "auto")
    resolved_device = _resolve_device(device_config)
    
    # 解析模型路径
    raw_model_path = vlm_data.get("model_path", "model/MiniMind2-V")
    raw_vision_path = vlm_data.get("vision_model_path", "model/vision_model/chinese-clip-vit-base-patch16")
    
    return VLMConfig(
        model_path=_resolve_path(raw_model_path, root),
        vision_model_path=_resolve_path(raw_vision_path, root),
        weight_name="sft_vlm",
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=False,
        device=resolved_device,
        max_new_tokens=generation_data.get("max_new_tokens", defaults.max_new_tokens),
        temperature=generation_data.get("temperature", defaults.temperature),
        top_p=generation_data.get("top_p", defaults.top_p),
        do_sample=generation_data.get("do_sample", defaults.do_sample),
        repetition_penalty=generation_data.get("repetition_penalty", defaults.repetition_penalty),
        default_prompt=prompt_data.get("system", defaults.default_prompt).strip()
    )


# ============================================================================
# 检索配置
# ============================================================================

@dataclass
class RetrievalConfig:
    """检索配置"""
    # 图像检索
    image_top_k: int
    image_score_threshold: float
    
    # 知识库检索
    kb_top_k: int
    kb_score_threshold: float


def get_retrieval_config(config_path: Optional[str] = None) -> RetrievalConfig:
    """加载检索配置"""
    data = _load_yaml_config(config_path)
    retrieval_data = data.get("retrieval", {})
    image_data = retrieval_data.get("image", {})
    kb_data = retrieval_data.get("kb", {})
    
    return RetrievalConfig(
        image_top_k=image_data.get("top_k", 5),
        image_score_threshold=image_data.get("score_threshold", 0.6),
        kb_top_k=kb_data.get("top_k", 5),
        kb_score_threshold=kb_data.get("score_threshold", 0.5)
    )


# 缓存的配置实例
_retrieval_config: Optional[RetrievalConfig] = None


def get_default_retrieval_config() -> RetrievalConfig:
    """获取检索配置单例"""
    global _retrieval_config
    if _retrieval_config is None:
        _retrieval_config = get_retrieval_config()
    return _retrieval_config
