from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Decoupled FastAPI Application")
    
    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(api_router, prefix="/v1")
    
    # 模型与服务预加载
    @app.on_event("startup")
    async def preload_services():
        """应用启动时预加载模型和连接"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. 预加载 VLM 模型
        logger.info("Preloading VLM model...")
        from src.services import get_vlm_service_instance
        vlm_service = get_vlm_service_instance()
        vlm_service.load_model()
        logger.info("VLM model preload completed")
        
        # 2. 预加载 CLIP 图像编码模型
        logger.info("Preloading CLIP model for image embedding...")
        from src.services import get_image_embedding_service
        embedding_service = get_image_embedding_service()
        embedding_service.load_model()
        logger.info("CLIP model preload completed")
        
        # 3. 连接 Milvus 知识库
        logger.info("Connecting to Milvus...")
        from src.milvus.client import MilvusClient
        milvus_client = MilvusClient()
        try:
            milvus_client.connect()
            logger.info("Milvus connection established")
        except Exception as e:
            logger.warning(f"Milvus connection failed (service may not be running): {e}")
    
    return app

app = create_app()


