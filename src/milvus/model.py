# -*- coding: utf-8 -*-
"""
Schema 定义模块

定义 Milvus Collection 的 Schema、索引参数和数据类。
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set

from pymilvus import DataType, FieldSchema, CollectionSchema

@dataclass
class CollectionInfo:
    """Collection 信息"""
    name: str
    description: str
    num_fields: int
    vector_dim: int
    index_type: str
    metric_type: str
    created: bool  # True=新创建, False=已存在


@dataclass
class CollectionStats:
    """Collection 统计信息"""
    name: str
    row_count: int
    index_status: str 
    loaded: bool

EMBEDDING_DIM = 512  # 向量维度


# ============================================================================
# 图像案例库 Schema (case_images)
# ============================================================================

IMAGE_COLLECTION_NAME = "case_images"
IMAGE_COLLECTION_DESCRIPTION = "交通多模态事故案例库，用于 Image→Image 相似检索"

IMAGE_COLLECTION_FIELDS: List[FieldSchema] = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="唯一标识"),
    FieldSchema(name="visual_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM, description="视觉向量 Embedding"),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512, description="图片存储路径"),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096, description="VLM场景描述"),
    FieldSchema(name="accident_type", dtype=DataType.VARCHAR, max_length=64, description="事故类型"),

    FieldSchema(name="legal_basis_ids", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=100, description="关联法规ID"),
]

# 图像案例库必需字段集合（不含 auto_id 的字段）
IMAGE_REQUIRED_FIELDS = {
    "visual_vector", "image_path", "description", "accident_type"
}


def get_image_collection_schema() -> CollectionSchema:
    """获取图像案例库 Schema"""
    return CollectionSchema(
        fields=IMAGE_COLLECTION_FIELDS,
        description=IMAGE_COLLECTION_DESCRIPTION
    )


# ============================================================================
# 文本知识库 Schema (security_kb_chunks)
# ============================================================================

KB_COLLECTION_NAME = "security_kb_chunks"
KB_COLLECTION_DESCRIPTION = "交通法规与标准安全知识库，用于 Text→Text 检索"

KB_COLLECTION_FIELDS: List[FieldSchema] = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="唯一标识"),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM, description="文本向量 Embedding"),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="条款内容"),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64, description="来源类型(law/standard等)"),
    FieldSchema(name="reference", dtype=DataType.VARCHAR, max_length=256, description="引用出处"),
    FieldSchema(name="tags", dtype=DataType.JSON, description="标签(JSON列表)"),
]

# 文本知识库必需字段集合（不含 auto_id 的字段）
KB_REQUIRED_FIELDS = {
    "vector", "content", "source_type", "reference", "tags"
}


def get_kb_collection_schema() -> CollectionSchema:
    """获取文本知识库 Schema"""
    return CollectionSchema(
        fields=KB_COLLECTION_FIELDS,
        description=KB_COLLECTION_DESCRIPTION
    )


def get_hnsw_index_params(m: int = 16, ef_construction: int = 256) -> Dict[str, Any]:
    """
    获取 HNSW 索引参数
    
    Args:
        m: 每层最大连接数，默认 16
        ef_construction: 构建时搜索范围，默认 256
        
    Returns:
        索引参数字典
    """
    return {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": m,
            "efConstruction": ef_construction
        }
    }


# 默认索引参数
DEFAULT_HNSW_INDEX_PARAMS = get_hnsw_index_params(m=16, ef_construction=256)

def validate_image_schema_fields(field_names: Set[str]) -> bool:
    """验证图像案例库 Schema"""
    return IMAGE_REQUIRED_FIELDS.issubset(field_names)


def validate_kb_schema_fields(field_names: Set[str]) -> bool:
    """验证文本知识库 Schema"""
    return KB_REQUIRED_FIELDS.issubset(field_names)


def get_all_image_field_names() -> Set[str]:
    """获取图像案例库所有字段名"""
    return {field.name for field in IMAGE_COLLECTION_FIELDS}


def get_all_kb_field_names() -> Set[str]:
    """获取文本知识库所有字段名"""
    return {field.name for field in KB_COLLECTION_FIELDS}

