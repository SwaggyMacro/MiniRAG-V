# -*- coding: utf-8 -*-
"""
数据清洗模块

提供交通事故数据集的验证、清洗和去重功能。
作为独立模块供 init_milvus_collections.py 调用。
"""

import warnings
import os

# 抑制 HuggingFace 和 PyTorch 警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_jsonl_records(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件中的所有记录
    
    Args:
        file_path: JSONL 文件的绝对或相对路径
        
    Returns:
        解析后的记录列表
    """
    records = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return records
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
    
    logger.info(f"从 {file_path} 加载了 {len(records)} 条记录")
    return records


def validate_record(record: Dict[str, Any], data_dir: Path) -> bool:
    """
    验证单条记录的完整性
    
    检查项：
    - image 字段存在且非空
    - 图片文件实际存在
    - conversations 字段存在且至少有 2 条
    
    Args:
        record: 数据记录
        data_dir: 数据集根目录（用于验证图片路径）
        
    Returns:
        True 表示记录有效
    """
    # 检查 image 字段
    image_path = record.get("image", "")
    if not image_path:
        return False
    
    # 检查图片文件存在
    full_image_path = data_dir / image_path
    if not full_image_path.exists():
        logger.debug(f"图片不存在: {full_image_path}")
        return False
    
    # 检查 conversations 字段
    conversations = record.get("conversations", [])
    if len(conversations) < 2:
        return False
    
    return True


def clean_description_text(text: str, max_length: int = 4096) -> str:
    """
    清洗描述文本
    
    处理：
    - 去除首尾空白
    - 移除特殊控制字符
    - 截断至最大长度
    
    Args:
        text: 原始文本
        max_length: 最大字符数
        
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 去除首尾空白
    cleaned = text.strip()
    
    # 移除控制字符（保留换行符）
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    
    # 截断
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned


def extract_description_from_record(record: Dict[str, Any]) -> str:
    """
    从记录中提取场景描述
    
    conversations 格式: [{"content": "问题"}, {"content": "回答"}]
    
    Args:
        record: 数据记录
        
    Returns:
        提取的描述文本
    """
    conversations = record.get("conversations", [])
    
    if len(conversations) >= 2:
        raw_desc = conversations[1].get("content", "")
        return clean_description_text(raw_desc)
    
    return ""


def deduplicate_by_image_path(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    基于图片路径去重
    
    保留第一次出现的记录
    
    Args:
        records: 记录列表
        
    Returns:
        去重后的记录列表
    """
    seen_paths = set()
    unique_records = []
    
    for record in records:
        image_path = record.get("image", "")
        if image_path and image_path not in seen_paths:
            seen_paths.add(image_path)
            unique_records.append(record)
    
    removed_count = len(records) - len(unique_records)
    if removed_count > 0:
        logger.info(f"去重移除了 {removed_count} 条重复记录")
    
    return unique_records


def clean_dataset(
    jsonl_path: str,
    data_dir: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    完整的数据清洗流程
    
    流程：加载 → 验证 → 去重 → 返回
    
    Args:
        jsonl_path: JSONL 数据文件路径
        data_dir: 数据集根目录，用于验证图片。默认为 jsonl 文件所在目录
        limit: 限制返回的记录数
        
    Returns:
        清洗后的有效记录列表
    """
    jsonl_path = Path(jsonl_path)
    
    if data_dir is None:
        data_dir = jsonl_path.parent
    else:
        data_dir = Path(data_dir)
    
    logger.info(f"开始清洗数据集: {jsonl_path}")
    logger.info(f"数据目录: {data_dir}")
    
    # 1. 加载数据
    records = load_jsonl_records(str(jsonl_path))
    if not records:
        return []
    
    # 2. 验证记录
    valid_records = [r for r in records if validate_record(r, data_dir)]
    invalid_count = len(records) - len(valid_records)
    if invalid_count > 0:
        logger.info(f"验证过滤了 {invalid_count} 条无效记录")
    
    # 3. 去重
    unique_records = deduplicate_by_image_path(valid_records)
    
    # 4. 限制数量
    if limit and limit > 0:
        unique_records = unique_records[:limit]
        logger.info(f"限制返回 {limit} 条记录")
    
    logger.info(f"清洗完成，共 {len(unique_records)} 条有效记录")
    return unique_records


# ============================================================================
# 向量提取与入库功能
# ============================================================================

# 事故类型关键词映射
ACCIDENT_TYPE_KEYWORDS = {
    "追尾": ["追尾", "后方撞击", "跟车"],
    "碰撞": ["碰撞", "相撞", "撞击", "冲撞"],
    "翻车": ["翻车", "侧翻", "倾覆"],
    "刮擦": ["刮擦", "剐蹭", "擦碰"],
    "行人事故": ["行人", "路人", "横穿"],
    "车辆故障": ["故障", "抛锚", "熄火"],
}


def classify_accident_type(description: str) -> str:
    """
    基于关键词分类事故类型
    
    Args:
        description: 场景描述文本
        
    Returns:
        事故类型字符串
    """
    if not description:
        return "traffic_accident"
    
    for accident_type, keywords in ACCIDENT_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in description:
                return accident_type
    
    return "traffic_accident"


def build_milvus_records(
    records: List[Dict[str, Any]],
    data_dir: Path,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    构建 Milvus 入库记录
    
    包含向量提取和字段映射。
    
    Args:
        records: 清洗后的原始记录
        data_dir: 数据目录
        dry_run: 演练模式（使用零向量）
        
    Returns:
        可直接入库的记录列表
    """
    from src.services.image_embedding_service import get_image_embedding_service
    
    if not dry_run:
        embedding_service = get_image_embedding_service()
        embedding_service.load_model()
    
    milvus_records = []
    failed_count = 0
    
    for i, record in enumerate(records):
        try:
            image_rel_path = record.get("image", "")
            image_abs_path = data_dir / image_rel_path
            
            # 提取描述
            description = extract_description_from_record(record)
            
            # 分类事故类型
            accident_type = classify_accident_type(description)
            
            # 提取向量
            if dry_run:
                visual_vector = [0.0] * 512
            else:
                with open(image_abs_path, 'rb') as f:
                    image_data = f.read()
                embedding_result = embedding_service.extract_embedding(image_data)
                visual_vector = embedding_result.vector
            
            milvus_record = {
                "visual_vector": visual_vector,
                "image_path": str(image_rel_path),
                "description": description,
                "accident_type": accident_type,
                "legal_basis_ids": [],
            }
            milvus_records.append(milvus_record)
            
            if (i + 1) % 50 == 0:
                logger.info(f"已处理 {i + 1}/{len(records)} 条记录")
                
        except Exception as e:
            logger.error(f"处理记录 {i} 失败: {e}")
            failed_count += 1
    
    logger.info(f"构建完成: 成功 {len(milvus_records)} 条, 失败 {failed_count} 条")
    return milvus_records


def insert_to_milvus(
    records: List[Dict[str, Any]],
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """
    批量插入数据到 Milvus
    
    Args:
        records: 准备入库的记录
        batch_size: 批量大小
        dry_run: 演练模式
        
    Returns:
        成功插入的记录数
    """
    from src.milvus.client import MilvusClient
    
    if dry_run:
        logger.info(f"[演练模式] 将插入 {len(records)} 条记录到 Milvus")
        for rec in records[:3]:
            logger.info(f"  - {rec['image_path']}: {rec['accident_type']}")
        return len(records)
    
    logger.info("连接 Milvus...")
    client = MilvusClient()
    client.connect()
    client.ensure_collections_exist()
    
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            primary_keys = client.insert("case_images", batch)
            total_inserted += len(primary_keys)
            logger.info(f"批次 {i // batch_size + 1}: 插入 {len(primary_keys)} 条")
        except Exception as e:
            logger.error(f"批次插入失败 (offset {i}): {e}")
    
    client.disconnect()
    logger.info(f"入库完成，共插入 {total_inserted} 条记录")
    return total_inserted


def process_and_ingest(
    jsonl_path: str,
    data_dir: str = None,
    limit: int = None,
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """
    完整的数据处理与入库流程
    
    流程: 清洗 → 向量提取 → Milvus 入库
    
    Args:
        jsonl_path: JSONL 数据文件路径
        data_dir: 数据目录（默认为 jsonl 所在目录）
        limit: 限制处理数量
        batch_size: 批量插入大小
        dry_run: 演练模式
        
    Returns:
        成功入库的记录数
    """
    jsonl_path = Path(jsonl_path)
    if data_dir is None:
        data_dir = jsonl_path.parent
    else:
        data_dir = Path(data_dir)
    
    logger.info("=" * 60)
    logger.info("开始数据处理与入库流程")
    logger.info("=" * 60)
    
    # 1. 清洗
    logger.info("步骤 1: 清洗数据...")
    cleaned = clean_dataset(str(jsonl_path), str(data_dir), limit)
    if not cleaned:
        logger.warning("无有效数据")
        return 0
    
    # 2. 构建入库记录
    logger.info("步骤 2: 提取向量并构建记录...")
    milvus_records = build_milvus_records(cleaned, data_dir, dry_run)
    
    # 3. 入库
    logger.info("步骤 3: 写入 Milvus...")
    inserted = insert_to_milvus(milvus_records, batch_size, dry_run)
    
    logger.info("=" * 60)
    logger.info(f"流程完成，共入库 {inserted} 条记录")
    logger.info("=" * 60)
    
    return inserted
