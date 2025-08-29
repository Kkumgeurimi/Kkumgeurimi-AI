# ragapi.py (직업별 동적 프롬프트 버전)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os, pathlib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------
# 환경설정
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")  # <- 이 줄이 핵심

HOME = pathlib.Path.home()
CSV_PROGRAM = os.getenv("CSV_PROGRAM", str(HOME / "hackathon/artifacts/data/program.csv"))
MODEL_NAME  = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai_client = None

MONGO_URI = os.getenv("MONGO_URI", "")
users = events = None
if MONGO_URI:
    try:
        from pymongo import MongoClient
        mcli = MongoClient(MONGO_URI)
        db = mcli["ggoomgil"]      # 요청대로 ggoomgil DB 사용
        users = db["users"]
        events = db["events"]
    except Exception:
        users = events = None

# -----------------------------
# 앱 & CORS
# -----------------------------
app = FastAPI(title="Chat (RAG) API - Dynamic Profession")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# -----------------------------
# 글로벌 캐시 (데이터/임베딩)
# -----------------------------
prog: Optional[pd.DataFrame] = None
texts: List[str] = []
C: Optional[np.ndarray] = None
model = None

# -----------------------------
# 유틸
# -----------------------------
def fmt_price(v):
    if v is None or str(v).strip() == "" or str(v).lower() == "nan":
        return "미정"
    try:
        f = float(v)
        return "무료" if f == 0 else f"{int(f):,}원"
    except Exception:
        return str(v)

def build_text_row(row: Dict) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("program_type", "")),
        str(row.get("target_audience", "")),
        str(row.get("eligible_region", "")),
        str(row.get("provider", "")),
        str(row.get("description", "")),
    ]
    return " ".join(p for p in parts if p and str(p).lower() != "nan").strip()

# -----------------------------
# 사용자 프롬프트(문장형)
# -----------------------------
def build_user_prompt(query: str, matches: List[Dict], profile: Dict[str, Any]) -> str:
    # 추천 후보 요약(문장형 지시와 충돌 없도록 번호만 유지)
    lines = []
    for i, p in enumerate(matches, 1):
        lines.append(
            f"{i}. {p.get('title','-')} / 유형:{p.get('program_type','-')} "
            f"/ 대상:{p.get('target_audience','-')} / 지역:{p.get('region','-')} "
            f"/ 참가비:{fmt_price(p.get('fee'))}"
        )
    programs_block = "\n".join(lines) if lines else "(관련 프로그램 없음)"

    name = profile.get("name") or "사용자"
    age  = profile.get("age")
    grade = profile.get("grade")
    interests = ", ".join(profile.get("interests", [])) if profile.get("interests") else "미지정"

    # 출력 형식은 문장형(마크다운 금지), 내용 구조만 안내
    return (
        f"[프로필] 이름:{name}, 나이:{age}, 학력/학년:{grade}, 관심사:{interests}\n"
        f"[질문] {query}\n\n"
        f"[관련 프로그램 상위 {len(matches)}개]\n{programs_block}\n\n"
        "출력은 반드시 문장형으로 작성한다. 1) 먼저 공감 문단, 2) 다음 문단에서 무료/유료 여부와 대상, 활동 포인트를 포함한 간단하고 명확한 솔루션을 제시한다. "
        "추천은 최대 3개 이내에서 선택적으로 언급한다. 3) 마지막 문장에 자연스러운 연결멘트를 덧붙인다. "
        "마크다운 불릿/헤더/코드블록 등 형식은 절대 사용하지 않는다."
    )

# -----------------------------
# 동적 System 프롬프트(직업별)
# -----------------------------
def build_system_prompt(profession: str) -> str:
    # 입력이 비어 있거나 지나치게 길면 기본값/제약
    profession = (profession or "").strip()
    if not profession:
        profession = "진로 상담 지식을 갖춘 직장인"
    if len(profession) > 60:
        profession = profession[:60]

    # 여기서 직업 기반 지침을 통합
    return (
        f"너는 {profession}을(를) 가진 직장인이야. 중고등학교 학생들에게 진로 상담을 제공해줘. "
        "적절한 진로 지식은 넣되, 너무 어려운 단어를 쓰지 않고 친절하게 말해줘. "
        "말투는 따뜻하고 편안해야 하며, 학생의 상황에 맞는 실천 가능한 조언을 포함해. "
        "마크다운 문법은 절대 사용하지 말고, 문장형으로만 답해."
    )

# -----------------------------
# OpenAI 호출
# -----------------------------
def run_llm(system_prompt: str, user_prompt: str) -> str:
    if not oai_client:
        return ""
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=700,
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# 스키마
# -----------------------------
class ChatReq(BaseModel):
    user_id: Optional[str] = "demo"
    profession: Optional[str] = Field(default="진로 상담 지식을 갖춘 직장인", description="예: 소프트웨어 엔지니어, 데이터 사이언티스트, 산업디자이너 등")
    query: str
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

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

# -----------------------------
# 스타트업: 데이터/모델 로드
# -----------------------------
@app.on_event("startup")
def _startup():
    global prog, texts, C, model
    if not pathlib.Path(CSV_PROGRAM).exists():
        raise FileNotFoundError(f"program CSV not found: {CSV_PROGRAM}")
    prog = pd.read_csv(CSV_PROGRAM)
    if "price" in prog.columns:
        prog["price"] = prog["price"].apply(lambda x: x if pd.notna(x) else None)
    texts[:] = [build_text_row(r) for _, r in prog.iterrows()]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    C = model.encode(texts, normalize_embeddings=True)  # [N, D]

# -----------------------------
# 엔드포인트
# -----------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "items": 0 if prog is None else int(len(prog)),
        "mongo": bool(users is not None and events is not None),
        "model": bool(model is not None),
    }

# (선택) 프로필 관리/로그 기능 유지
@app.post("/profile/upsert")
def upsert_profile(p: Profile):
    if users is None:
        raise HTTPException(503, "MongoDB is not configured")
    users.update_one({"_id": p.user_id}, {"$set": p.model_dump(exclude={"user_id"})}, upsert=True)
    return {"ok": True}

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    if users is None:
        raise HTTPException(503, "MongoDB is not configured")
    doc = users.find_one({"_id": user_id})
    if not doc:
        raise HTTPException(404, "profile not found")
    doc["user_id"] = doc.pop("_id")
    return doc

@app.post("/log/event")
def log_event(body: EventIn):
    if events is None:
        raise HTTPException(503, "MongoDB is not configured")
    events.insert_one({"type": body.type, "user_id": body.user_id, "payload": body.payload})
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatReq):
    # 0) 쿼리 확인
    q = (req.query or "").strip()
    if not q:
        return {"answer_md": "검색어가 비어있어요.", "top_matches": []}

    # 1) 임베딩 검색 (Top-K)
    qv = model.encode([q], normalize_embeddings=True)[0]
    sims = C @ qv
    k = int(min(req.top_k or 3, len(sims)))
    top_idx = np.argsort(-sims)[:k]

    matches: List[Dict] = []
    for i in top_idx:
        row = prog.iloc[int(i)].to_dict()
        matches.append({
            "program_id": row.get("program_id"),
            "title": row.get("title"),
            "program_type": row.get("program_type"),
            "target_audience": row.get("target_audience"),
            "region": row.get("eligible_region"),
            "fee": row.get("price"),
            "score": float(sims[int(i)]),
        })

    # 2) 프로필 조회(있으면)
    profile: Dict[str, Any] = {}
    if users is not None and req.user_id:
        doc = users.find_one({"_id": req.user_id})
        if doc:
            profile = {**doc}
            profile.pop("_id", None)

    # 3) 동적 system + user 프롬프트
    system_prompt = build_system_prompt(req.profession)
    user_prompt   = build_user_prompt(q, matches, profile)

    # 4) LLM 호출 (fallback 문장형)
    answer = run_llm(system_prompt, user_prompt)
    if not answer:
        if matches:
            parts = []
            for m in matches:
                parts.append(
                    f"{m['title']}는 {m.get('program_type','-')} 형태로, 대상은 {m.get('target_audience','-')}, "
                    f"지역은 {m.get('region','-')}이며, 참가비는 {fmt_price(m.get('fee'))}이다."
                )
            body = " ".join(parts)
            answer = f"좋아, 네 고민 이해해. {body} 지금 마음에 끌리는 것부터 하나씩 시도해보자. 더 궁금한 게 있으면 편하게 물어봐."
        else:
            answer = "당장 추천할 정보를 찾지 못했어. 대신 네 관심사나 지역을 조금 더 알려주면 더 잘 도울 수 있어. 다른 궁금한 점 있으면 편하게 물어봐."


    return {"answer_md": answer, "top_matches": matches, "used_profile": bool(profile)}
