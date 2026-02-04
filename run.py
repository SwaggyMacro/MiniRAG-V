import logging
import uvicorn
from src.main import app

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

