# ragapi.py (직업별 동적 프롬프트 버전, history 컬렉션 적용)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os, pathlib, re
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

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
users = history = None
if MONGO_URI:
    try:
        from pymongo import MongoClient
        mcli = MongoClient(MONGO_URI)
        db = mcli["ggoomgil"]      # 요청대로 ggoomgil DB 사용
        users = db["users"]
        history = db["history"]    # ← 채팅 히스토리 컬렉션명: history
        # (선택) 인덱스(최초 1회 실행되어도 무해)
        try:
            history.create_index([("user_id", 1), ("profession", 1), ("ts", -1)])
        except Exception:
            pass
    except Exception:
        users = history = None

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

def is_reco_intent(q: str) -> bool:
    """'추천', '추천해줘', '프로그램 추천' 등 포함 시 True"""
    if not q:
        return False
    q = q.strip().lower()
    patterns = [
        r"프로그램\s*추천",    # 프로그램 추천
        r"recommend",         # 영문 recommend
    ]
    return any(re.search(p, q) for p in patterns)

def fetch_recent_chat_history(
    user_id: Optional[str],
    profession: Optional[str],
    limit: int = 3
) -> List[Dict[str, str]]:
    """
    history 컬렉션에서 (user_id, profession) 쌍으로 최근 대화 turn 'limit'개 조회.
    오래된 → 최근 순으로 반환.
    """
    if history is None or not user_id or not profession:
        return []

    cur = history.find(
        {
            "type": "chat",
            "user_id": user_id,
            "profession": profession,
            "query":   {"$exists": True},
            "answer":  {"$exists": True},
        }
    ).sort("ts", -1).limit(limit)

    hist_docs = list(cur)[::-1]  # 오래된 → 최신
    out: List[Dict[str, str]] = []
    for h in hist_docs:
        uq = (h.get("query") or "").strip()
        ua = (h.get("answer") or "").strip()
        if uq and ua:
            out.append({"query": uq, "answer": ua})
    return out

# -----------------------------
# 사용자 프롬프트(문장형)
# -----------------------------
def build_user_prompt(query: str, matches: List[Dict], profile: Dict[str, Any], recommend: bool) -> str:
    # 최소한의 컨텍스트만 남기기 (대화 톤은 모델이 결정)
    name = profile.get("name") or "사용자"
    age  = profile.get("age")
    grade = profile.get("grade")
    interests = ", ".join(profile.get("interests", [])) if profile.get("interests") else None

    profile_bits = [f"이름:{name}"]
    if age is not None:   profile_bits.append(f"나이:{age}")
    if grade:             profile_bits.append(f"학력/학년:{grade}")
    if interests:         profile_bits.append(f"관심사:{interests}")
    profile_line = " / ".join(profile_bits)

    # 추천 의도일 때만 후보를 간단 리스트로 첨부 (참고용 메모)
    programs_block = ""
    if recommend:
        rows = []
        for i, p in enumerate(matches, 1):
            rows.append(
                f"{i}. {p.get('title','-')} | 유형:{p.get('program_type','-')} | 대상:{p.get('target_audience','-')} | "
                f"지역:{p.get('region','-')} | 참가비:{fmt_price(p.get('fee'))}"
            )
        programs_block = "\n참고용 프로그램 후보:\n" + ("\n".join(rows) if rows else "(없음)")

    # 👉 형식 지시(1,2,3 단계), 마크다운 금지 등 빡센 규칙 제거
    # 👉 “대화 흐름에 맞춰 자연스럽게”만 가볍게 힌트
    return (
        f"[프로필] {profile_line}\n"
        f"[사용자 질문] {query}\n"
        f"{programs_block}\n\n"
        "위 후보 목록은 참고용 메모일 뿐이야. 대화 흐름과 이전 맥락을 존중해서, 필요할 때만 자연스럽게 언급해 줘."
    )

# -----------------------------
# 동적 System 프롬프트(직업별)
# -----------------------------
def build_system_prompt(profession: str) -> str:
    profession = (profession or "").strip()
    if not profession:
        profession = "진로 상담 지식을 갖춘 직장인"
    if len(profession) > 60:
        profession = profession[:60]

    return (
        f"너는 {profession}을(를) 직업으로 가진 직장인이야. 중고등학교 학생들에게 진로 상담을 제공해줘. "
        "적절한 진로 지식은 넣되, 너무 어려운 단어를 쓰지 않고 친절하게 말해줘. "
        "말투는 친절하고 편안해야 하며, 학생의 상황에 맞는 실천 가능한 조언을 포함해. "
        "이전 대화 맥락을 반드시 존중하고, 전체적인 대화의 톤을 유지하며, 자연스러운 대화를 추구해."
        "이미 한 학생과 편오래 대화를 한 상태라면, 처음 말하는 것처럼 하지 않도록 더 신경쓰도록 해"
        "마크다운 문법은 절대 사용하지 말고, 문장형으로만 답해."
    )

# -----------------------------
# OpenAI 호출
# -----------------------------
def run_llm_with_history(system_prompt: str, history_turns: List[Dict[str, str]], current_user_prompt: str) -> str:
    """
    messages = [system] + (u,a 반복) + [user] 형태로 구성해 호출.
    oai_client가 없거나 오류 시 빈 문자열 반환(폴백 로직이 처리).
    """
    if not oai_client:
        return ""
    try:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history_turns:
            messages.append({"role": "user", "content": turn["query"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": current_user_prompt})

        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=700,
            top_p=1.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[OpenAI with history error]", repr(e))
        return ""

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
        "mongo": (users is not None and history is not None),
        "model": bool(model is not None),
    }

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
    if history is None:
        raise HTTPException(503, "MongoDB is not configured")
    history.insert_one({
        "type": body.type,
        "user_id": body.user_id,
        "payload": body.payload,
        "ts": datetime.now(timezone.utc),
    })
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatReq):
    # 0) 쿼리 확인
    q = (req.query or "").strip()
    if not q:
        return {"answer_md": "검색어가 비어있어요.", "top_matches": []}

    # 0.5) 의도 감지
    recommend = is_reco_intent(q)

    matches: List[Dict] = []

    # 1) 추천 의도일 때만 임베딩 검색
    if recommend:
        context_text = " ".join(turn["query"] for turn in history_turns[-2:])
        query_text = f"{context_text} {q} {req.profession}"
        qv = model.encode([query_text], normalize_embeddings=True)[0]
        sims = C @ qv
        k = int(min(req.top_k or 3, len(sims)))
        top_idx = np.argsort(-sims)[:k]

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

    # 3) 프롬프트 구성 & 히스토리 로드
    system_prompt = build_system_prompt(req.profession)
    user_prompt   = build_user_prompt(q, matches, profile, recommend=recommend)
    history_turns = fetch_recent_chat_history(req.user_id, req.profession, limit=3)

    # 4) LLM 호출 (히스토리 포함) → 실패 시 폴백
    answer = run_llm_with_history(system_prompt, history_turns, user_prompt)

    if not answer:
        if recommend and matches:
            parts = []
            for m in matches:
                parts.append(
                    f"{m['title']}는 {m.get('program_type','-')} 형태로, 대상은 {m.get('target_audience','-')}, "
                    f"지역은 {m.get('region','-')}이며, 참가비는 {fmt_price(m.get('fee'))}이다."
                )
            body = " ".join(parts)
            answer = f"좋아, 네 고민 이해해. {body} 지금 마음에 끌리는 것부터 하나씩 시도해보자. 더 궁금한 게 있으면 편하게 물어봐."
        else:
            name = profile.get("name") or "너"
            answer = (
                f"오, {name}가 이야기한 주제 흥미롭다! 처음엔 막막할 수 있지만 가볍게 시도해보면 금방 감이 와."
                f" 네 상황에 맞는 한두 가지 활동부터 시작해보자. 필요하면 내가 단계별로 같이 정리해줄게!"
            )

    # 5) 히스토리 저장
    if history is not None:
        history.insert_one({
            "type": "chat",
            "user_id": req.user_id,
            "profession": req.profession,
            "query": q,
            "top_k": req.top_k,
            "matches": matches,
            "answer": answer,
            "ts": datetime.now(timezone.utc),   # 타임스탬프(정렬용)
        })

    return {
        "answer_md": answer,
        "top_matches": matches if recommend else [],
        "used_profile": bool(profile),
        "recommend": recommend,
    }
