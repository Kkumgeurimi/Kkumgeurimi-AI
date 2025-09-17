from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os, re, pathlib, time, json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pydantic import BaseModel

# ---------- 설정 ----------
HOME = pathlib.Path.home()
CSV_PROGRAM = os.getenv("CSV_PROGRAM", str(HOME / "Kkumgeurimi-AI/artifacts/data/program.csv"))
EMB_PATH    = os.getenv("EMB_PATH",    str(HOME / "Kkumgeurimi-AI/artifacts/emb/items.npy"))
MODEL_NAME  = os.getenv("MODEL_NAME",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MONGO_URI   = os.getenv("MONGO_URI",   "")
MONGO_DB    = os.getenv("MONGO_DB",    "ggoomgil")  # DB 이름 고정
EVENTS_COL  = os.getenv("EVENTS_COL",  "events")

# 가중치/파라미터 (환경변수로 오버라이드 가능)
ALPHA = float(os.getenv("REC_ALPHA", "0.7"))  # 글로벌 인기
BETA  = float(os.getenv("REC_BETA",  "0.3"))  # 쿼리-관련 인기
SIM_THRESHOLD = float(os.getenv("REC_TAU", "0.4"))  # 쿼리-아이템 유사도 임계
DAYS = int(os.getenv("REC_DAYS", "30"))  # 최근 N일 이벤트로 인기 집계
WEIGHTS = json.loads(os.getenv("REC_EVENT_WEIGHTS", '{"click":1,"like":3}'))

# ---------- 앱 기본 ----------
app = FastAPI(title="Recommend API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- 유틸 ----------
def price_to_int(x):
    if pd.isna(x): return None
    s = str(x)
    if "무료" in s: return 0
    s = s.replace(",", "")
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def fmt_price(v):
    if v is None: return "-"
    try:
        v = int(float(v))
        return f"{v:,d}"
    except:
        return str(v)

# ---------- 데이터 로드 ----------
prog = pd.read_csv(CSV_PROGRAM)
prog.fillna("", inplace=True)
# price 정리(있으면)
if "price" in prog.columns:
    prog["price"] = prog["price"].apply(price_to_int)

# 표시용 컬럼 보정
display_cols = {
    "program_id": "program_id",
    "title": "title",
    "program_type": "program_type",
    "target_audience": "target_audience",
    "eligible_region": "eligible_region",
    "price": "price"
}
for k, v in display_cols.items():
    if v not in prog.columns:
        prog[v] = ""

# 임베딩
model = SentenceTransformer(MODEL_NAME)
# items 텍스트 빌드 (제목 + 유형 + 지역 등)
def build_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title","")),
        str(row.get("program_type","")),
        str(row.get("eligible_region","")),
        str(row.get("target_audience",""))
    ]
    return " ".join([p for p in parts if p])

ITEM_TEXTS = [build_text(r) for _, r in prog.iterrows()]
ITEM_EMB = np.load(EMB_PATH) if os.path.exists(EMB_PATH) else model.encode(ITEM_TEXTS, normalize_embeddings=True)
if ITEM_EMB.shape[0] != len(prog):
    # 임베딩 파일과 행 불일치 시, 즉석 인코딩
    ITEM_EMB = model.encode(ITEM_TEXTS, normalize_embeddings=True)

# Mongo
client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client[MONGO_DB] if client is not None else None
events = db[EVENTS_COL] if db is not None else None

# ---------- API 모델 ----------
class EventIn(BaseModel):
    user_id: str
    event: str   # "click" | "like" | ...
    program_id: str
    ts: Optional[int] = None

# ---------- 헬퍼들 ----------
def now_epoch() -> int:
    return int(time.time())

def get_recent_events(days: int = 30) -> List[Dict[str,Any]]:
    """최근 N일 이벤트 불러오기"""
    if events is None:
        return []
    since = now_epoch() - days*86400
    cur = events.find({"ts": {"$gte": since}}, {"_id":0, "user_id":1, "program_id":1, "event":1, "ts":1})
    return list(cur)

def popularity_from_events(evts: List[Dict[str,Any]], program_filter: Optional[set] = None) -> Dict[str, float]:
    """이벤트를 가중합으로 프로그램 인기 점수로 변환"""
    score = {}
    for e in evts:
        pid = str(e.get("program_id",""))
        if not pid: continue
        if program_filter is not None and pid not in program_filter:
            continue
        w = WEIGHTS.get(e.get("event",""), 0)
        if w == 0: continue
        score[pid] = score.get(pid, 0.0) + float(w)
    # 정규화
    if not score:
        return {}
    mx = max(score.values())
    if mx <= 0:
        return {k:0.0 for k in score}
    return {k: v/mx for k,v in score.items()}

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ---------- 라우트 ----------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "items": len(prog),
        "db": MONGO_DB,
        "mongo": events is not None
    }

@app.post("/log/event")
def log_event(body: EventIn):
    if events is None:
        return {"ok": False, "reason": "mongo_not_configured"}
    doc = body.dict()
    if not doc.get("ts"):
        doc["ts"] = now_epoch()
    events.insert_one(doc)
    return {"ok": True}

@app.get("/recommend")
def recommend(
    query: str = Query(..., description="검색 쿼리"),
    k: int = 5,
    user_id: Optional[str] = Query(None, description="요청 유저"),
):
    q = (query or "").strip()
    if not q:
        return {"items": [], "count": 0, "used_profile": False}

    # 1) 코사인 유사도
    qv = model.encode([q], normalize_embeddings=True)[0]
    sims = ITEM_EMB @ qv
    idx_sorted = np.argsort(-sims)

    # 2) 최근 이벤트 → (a) 전체 인기 (b) 쿼리-관련 인기
    pop_global = {}
    pop_related = {}
    if events is not None:
        evts = get_recent_events(days=DAYS)

        # 전체 인기
        pop_global = popularity_from_events(evts)

        # 쿼리와 관련된 아이템만 필터 (sims >= SIM_THRESHOLD)
        related_ids = set()
        for i, s in enumerate(sims):
            if s >= SIM_THRESHOLD:
                pid = str(prog.iloc[i].get("program_id",""))
                if pid: related_ids.add(pid)
        pop_related = popularity_from_events(evts, program_filter=related_ids)

    # 3) 가중치 블렌딩
    scored = []
    for i in idx_sorted[:max(k*10, k)]:  # 약간 넉넉히 보고 상위 k 추림
        row = prog.iloc[i]
        pid = str(row.get("program_id",""))
        base = safe_float(sims[i], 0.0)
        g = pop_global.get(pid, 0.0)
        r = pop_related.get(pid, 0.0)
        final = base + ALPHA*g + BETA*r
        scored.append((final, i))

    scored.sort(key=lambda x: -x[0])
    top_idx = [i for _, i in scored[:k]]

    items = []
    for i in top_idx:
        row = prog.iloc[i].to_dict()
        items.append({
            "program_id": str(row.get("program_id","")),
            "title": row.get("title",""),
            "program_type": row.get("program_type",""),
            "target_audience": row.get("target_audience",""),
            "region": row.get("eligible_region",""),
            "price": fmt_price(row.get("price", "")),
        })

    return {"items": items, "count": len(items), "used_profile": True}
