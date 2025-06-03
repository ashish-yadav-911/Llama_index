# AGENTIC_MIRAI/shared/validation/base.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# Example of a base model if you have common fields
class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation successful"
    data: Any = None
    errors: Optional[List[Dict[str, Any]]] = None