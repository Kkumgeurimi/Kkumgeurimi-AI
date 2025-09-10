# apps/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatReq(BaseModel):
    user_id: str = "demo"
    profession: str = Field(
        default="진로 상담 지식을 갖춘 직장인",
        description="예: 소프트웨어 엔지니어, 데이터 사이언티스트, 산업디자이너 등"
    )
    query: str
    top_k: int = Field(default=3, ge=1, le=10)

class Profile(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    interests: Optional[List[str]] = None
    grade: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class EventIn(BaseModel):
    type: str
    user_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

class ProgramMatch(BaseModel):
    program_id: Optional[str] = None
    title: Optional[str] = None
    program_type: Optional[str] = None
    target_audience: Optional[str] = None
    region: Optional[str] = None
    fee: Any = None
    score: float

class ChatResp(BaseModel):
    answer: str
    top_matches: List[ProgramMatch]
    used_profile: bool
