# recapi.py
import os, time
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from collections import Counter, defaultdict

MONGO_URI = os.environ.get("MONGO_URI")  # 예: mongodb+srv://.../ggoomgil
client = MongoClient(MONGO_URI) if MONGO_URI else None
db_name = (MONGO_URI.split("/")[-1].split("?")[0] if MONGO_URI and "/" in MONGO_URI else "ggoomgil")
db = client[db_name] if client is not None else None
events = db["events"] if db is not None else None

CSV_PROGRAM = os.path.expanduser("~/hackathon/artifacts/data/program.csv")
EMB_PATH    = os.path.expanduser("~/hackathon/artifacts/emb/items.npy")

prog = pd.read_csv(CSV_PROGRAM, dtype=str)
# 수치형 칼럼 캐스팅
for col in ["price", "program_id"]:
    if col in prog.columns:
        prog[col] = pd.to_numeric(prog[col], errors="ignore")
prog.fillna("", inplace=True)

C = np.load(EMB_PATH)  # (N, dim)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI(title="Recommend API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _now() -> int:
    return int(time.time())

def get_user_log_weights(user_id: str):
    """유저 로그로 program_id별 가중치 계산 (시연용 극적 보정)"""
    if events is None or not user_id:
        return {}

    logs = list(events.find({"user_id": user_id}))
    if not logs:
        return {}

    # 이벤트별 기본 가중치 (시연용 과감한 값)
    BASE = {"click": 1.0, "like": 2.0, "purchase": 4.0}
    # 최근성 보정용 하이퍼파라미터
    # 최근일수(day)로 감쇠: newer → 큰 가중치
    RECENCY_HALF_LIFE_DAYS = 5  # 5일마다 절반
    sec_half_life = RECENCY_HALF_LIFE_DAYS * 24 * 3600
    now = _now()

    weights = defaultdict(float)
    for log in logs:
        pid = str(log.get("program_id", ""))
        if not pid:
            continue
        etype = str(log.get("type", "click"))
        base = BASE.get(etype, 0.5)  # 모르는 타입은 낮게
        ts = int(log.get("ts", now))
        age = max(0, now - ts)
        recency = 0.5 ** (age / sec_half_life)  # 0~1
        weights[pid] += base * (0.6 + 0.4 * recency)  # 최근일수 보정

    # 정규화
    if not weights:
        return {}
    mx = max(weights.values())
    return {pid: w / mx for pid, w in weights.items()}

@app.get("/healthz")
def healthz():
    return {"ok": True, "items": len(prog), "db": db_name, "mongo": events is not None}

@app.get("/recommend")
def recommend(user_id: str, query: str, k: int = Query(5, ge=1, le=20), lambda_log: float = 1.0):
    """
    lambda_log: 로그 영향력 (시연용으로 0.9~1.2 추천)
    """
    q = (query or "").strip()
    if not q:
        return {"items": [], "count": 0, "used_profile": False, "note": "empty query"}

    qv = model.encode([q], normalize_embeddings=True)[0]  # (dim,)
    sims = C @ qv  # (N,)

    # 기본 임베딩 점수
    prog = pd.read_csv(CSV_PROGRAM, dtype=str)
    for col in ["price", "program_id"]:
        if col in prog.columns:
            prog[col] = pd.to_numeric(prog[col], errors="ignore")
    prog.fillna("", inplace=True)
    prog["score"] = sims

    # 로그 가중치
    log_w = get_user_log_weights(user_id)
    used_profile = bool(log_w)
    if used_profile:
        def _boost(row):
            pid = str(row.get("program_id"))
            return row["score"] + lambda_log * log_w.get(pid, 0.0)
        prog["score"] = prog.apply(_boost, axis=1)

    top = prog.sort_values("score", ascending=False).head(k)
    items = top[["program_id", "title", "program_type", "target_audience", "eligible_region", "price", "score"]].to_dict(orient="records")
    return {"items": items, "count": len(items), "used_profile": used_profile, "lambda_log": lambda_log}
