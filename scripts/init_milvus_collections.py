import sys
import os
import argparse
from pathlib import Path

# 将项目根目录添加到 python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.milvus.client import MilvusClient
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_database():
    """初始化两类核心 Collection (使用 MilvusClient)"""
    
    # 使用上下文管理器自动处理连接与断开
    try:
        with MilvusClient() as client:
            logger.info("Connected to Milvus via MilvusClient.")
            
            # 一键确保所有 Collections 存在
            result = client.ensure_collections_exist()
            
            logger.info("Database initialization summary:")
            for name, info in result.items():
                status = "Created New" if info.created else "Already Existed"
                logger.info(f" - {name.upper()}: {info.name} ({status})")
                
    except Exception as e:
        logger.error(f"Initialization failed: {e}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus 初始化与数据入库")
    parser.add_argument("--ingest", action="store_true", help="执行图像数据入库")
    parser.add_argument("--ingest-kb", action="store_true", help="执行法规数据入库")
    parser.add_argument("--data-path", type=str, default="dataset/accidents_export/all_data.jsonl")
    parser.add_argument("--kb-path", type=str, default="dataset/law_data.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="演练模式")
    
    args = parser.parse_args()
    
    # 始终先初始化数据库
    init_database()
    
    # 如果指定 --ingest 则执行图像入库
    if args.ingest:
        from scripts.data_cleaner import process_and_ingest
        
        full_path = Path(project_root) / args.data_path
        process_and_ingest(
            jsonl_path=str(full_path),
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    
    # 如果指定 --ingest-kb 则执行法规入库
    if args.ingest_kb:
        from scripts.kb_data_cleaner import process_and_ingest_kb
        
        kb_path = Path(project_root) / args.kb_path
        process_and_ingest_kb(
            jsonl_path=str(kb_path),
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
