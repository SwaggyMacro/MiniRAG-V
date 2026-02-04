from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseService(ABC):
    """
    Abstract base class for all services to ensure a consistent interface.
    """

    # Removed abstract execute method to allow flexible service methods

    def get_service_name(self) -> str:
        """
        Returns the name of the service.
        """
        return self.__class__.__name__
