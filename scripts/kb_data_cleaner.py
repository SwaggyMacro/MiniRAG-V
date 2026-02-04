# -*- coding: utf-8 -*-
"""
法规知识库入库模块

将法规数据（law_data.jsonl）导入 Milvus security_kb_chunks Collection。
"""

import warnings
import os

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
transformers.logging.set_verbosity_error()

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_kb_records(file_path: str) -> List[Dict[str, Any]]:
    """
    加载法规 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        法规记录列表
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
    
    logger.info(f"从 {file_path} 加载了 {len(records)} 条法规记录")
    return records


def validate_kb_record(record: Dict[str, Any]) -> bool:
    """
    验证法规记录完整性
    
    必需字段: content, source_type, reference
    """
    required = ["content", "source_type", "reference"]
    for field in required:
        if not record.get(field):
            return False
    return True


def extract_text_vector(text: str, embedding_service) -> List[float]:
    """
    提取文本向量
    
    使用 Chinese-CLIP 的文本编码器
    """
    return embedding_service.extract_text_embedding(text)


def build_kb_milvus_records(
    records: List[Dict[str, Any]],
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    构建法规入库记录
    
    Args:
        records: 法规记录列表
        dry_run: 演练模式
        
    Returns:
        可入库的记录列表
    """
    import torch
    from src.config import get_vlm_config
    
    model = None
    processor = None
    device = None
    
    if not dry_run:
        from transformers import CLIPModel, CLIPProcessor
        
        config = get_vlm_config()
        device = config.device
        vision_model_path = config.vision_model_path
        
        logger.info(f"加载 Chinese-CLIP 文本编码器 ({vision_model_path})...")
        model = CLIPModel.from_pretrained(vision_model_path)
        processor = CLIPProcessor.from_pretrained(vision_model_path)
        model = model.eval().to(device)
        logger.info("模型加载完成")
    
    milvus_records = []
    failed_count = 0
    
    for i, record in enumerate(records):
        try:
            if not validate_kb_record(record):
                logger.warning(f"记录 {i} 验证失败，跳过")
                failed_count += 1
                continue
            
            content = record.get("content", "")
            source_type = record.get("source_type", "")
            reference = record.get("reference", "")
            tags = record.get("tags", [])
            
            # 提取文本向量
            if dry_run:
                vector = [0.0] * 512
            else:
                inputs = processor(text=content, return_tensors="pt", padding=True, truncation=True, max_length=52)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                vector = text_features.squeeze().cpu().tolist()
            
            milvus_record = {
                "vector": vector,
                "content": content[:65535],
                "source_type": source_type[:64],
                "reference": reference[:256],
                "tags": tags,
            }
            milvus_records.append(milvus_record)
            
        except Exception as e:
            logger.error(f"处理记录 {i} 失败: {e}")
            failed_count += 1
    
    # 清理模型
    if model is not None:
        del model
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()
    
    logger.info(f"构建完成: 成功 {len(milvus_records)} 条, 失败 {failed_count} 条")
    return milvus_records


def insert_kb_to_milvus(
    records: List[Dict[str, Any]],
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """
    批量插入法规数据到 Milvus
    """
    from src.milvus.client import MilvusClient
    
    if dry_run:
        logger.info(f"[演练模式] 将插入 {len(records)} 条法规到 Milvus")
        for rec in records[:3]:
            logger.info(f"  - {rec['reference']}: {rec['content'][:50]}...")
        return len(records)
    
    logger.info("连接 Milvus...")
    client = MilvusClient()
    client.connect()
    client.ensure_collections_exist()
    
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            primary_keys = client.insert("security_kb_chunks", batch)
            total_inserted += len(primary_keys)
            logger.info(f"批次 {i // batch_size + 1}: 插入 {len(primary_keys)} 条")
        except Exception as e:
            logger.error(f"批次插入失败 (offset {i}): {e}")
    
    client.disconnect()
    logger.info(f"法规入库完成，共插入 {total_inserted} 条记录")
    return total_inserted


def process_and_ingest_kb(
    jsonl_path: str,
    limit: int = None,
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """
    完整的法规数据处理与入库流程
    
    Args:
        jsonl_path: 法规 JSONL 文件路径
        limit: 限制处理数量
        batch_size: 批量大小
        dry_run: 演练模式
        
    Returns:
        成功入库的记录数
    """
    logger.info("=" * 60)
    logger.info("开始法规知识库入库流程")
    logger.info("=" * 60)
    
    # 1. 加载
    logger.info("步骤 1: 加载法规数据...")
    records = load_kb_records(jsonl_path)
    if not records:
        logger.warning("无法规数据")
        return 0
    
    if limit and limit > 0:
        records = records[:limit]
        logger.info(f"限制处理 {limit} 条")
    
    # 2. 构建入库记录
    logger.info("步骤 2: 提取文本向量...")
    milvus_records = build_kb_milvus_records(records, dry_run)
    
    # 3. 入库
    logger.info("步骤 3: 写入 Milvus...")
    inserted = insert_kb_to_milvus(milvus_records, batch_size, dry_run)
    
    logger.info("=" * 60)
    logger.info(f"法规入库完成，共入库 {inserted} 条记录")
    logger.info("=" * 60)
    
    return inserted
