# apps/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- ⭐️ 1. 채팅 요청/응답 스키마 단순화 ---
class ChatReq(BaseModel):
    student_id: str
    profession: str = Field(
        default="진로 상담 지식을 갖춘 직장인",
        description="예: 소프트웨어 엔지니어, 데이터 사이언티스트, 산업디자이너 등"
    )
    query: str
    # 추천 관련 필드는 모두 제거합니다.

class ChatResp(BaseModel):
    answer: str
    # top_matches 필드를 제거합니다.

# --- ⭐️ 2. 추천 전용 요청/응답 스키마 신규 추가 ---
class RecommendReq(BaseModel):
    student_id: str
    profession: str
    top_k: int = Field(default=3, ge=1, le=10)

class RecommendResp(BaseModel):
    message: str
    top_matches: List['ProgramMatch'] # ProgramMatch는 아래에 정의


# --- 공통 사용 스키마 ---

class ProgramMatch(BaseModel):
    program_id: Optional[str] = None
    title: Optional[str] = None
    provider: Optional[str] = None
    date: Optional[str] = None
    program_type: Optional[str] = None
    target_audience: Optional[str] = None
    major: Optional[str] = None
    region: Optional[str] = None
    fee: Any = None
    score: float

class Profile(BaseModel):
    student_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    interests: Optional[List[str]] = None
    grade: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class EventIn(BaseModel):
    type: str
    student_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


RecommendResp.update_forward_refs()
