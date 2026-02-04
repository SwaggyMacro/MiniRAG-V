from .base import BaseService
from src.api.schemas import InfoResponse

class SystemService(BaseService):
    """
    Service responsible for system-level operations and information.
    """
    
    async def get_info(self) -> InfoResponse:
        # Real logic would go here, for now returning static/env-based info
        return InfoResponse(
            app_name="My Decoupled FastAPI App",
            version="1.0.0",
            status="healthy"
        )
