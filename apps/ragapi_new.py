# apps/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import uuid

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from . import config
from . import schemas

# ======= S3 다운로드 함수 ===========
def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, local_path)
        print("✅ Download successful.")
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"🔥 Failed to download from S3: {e}")
        return False

# ======================================================================================
# 1. 앱 초기화 및 미들웨어 설정
# ======================================================================================
app = FastAPI(title="진로 체험 추천 챗봇 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# ======================================================================================
# 2. 유틸리티 함수
# ======================================================================================
def fmt_price(v: Any) -> str:
    """가격 포맷팅 유틸리티"""
    if v is None or str(v).strip() == "" or pd.isna(v):
        return "미정"
    try:
        f = float(v)
        return "무료" if f == 0 else f"{int(f):,}원"
    except (ValueError, TypeError):
        return str(v)

# ======================================================================================
# 3. FastAPI 시작/종료 이벤트
# ======================================================================================
@app.on_event("startup")
def startup_event():
    """애플리케이션 시작 시 DB 연결 및 모델/데이터 로드"""
    # MongoDB 연결
    try:
        app.state.mcli = MongoClient(config.MONGO_URI)
        app.state.db = app.state.mcli[config.DB_NAME]
        app.state.users = app.state.db[config.USERS_COLLECTION]
        app.state.history = app.state.db[config.HISTORY_COLLECTION]
        # ⭐️ 추천 기록 컬렉션 초기화
        app.state.recommendations = app.state.db[config.RECOMMENDATIONS_COLLECTION]
        
        # 인덱스 설정
        app.state.history.create_index([("student_id", 1), ("profession", 1), ("ts", -1)])
        # ⭐️ 추천 기록 컬렉션 인덱스 추가
        app.state.recommendations.create_index([("student_id", 1), ("ts", -1)])
        
        print("✅ MongoDB connected successfully.")

    except Exception as e:
        print(f"🔥 MongoDB connection failed: {e}")
        app.state.mcli = app.state.db = app.state.users = app.state.history = None

    # OpenAI 클라이언트 초기화
    app.state.oai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
    if app.state.oai_client: print("✅ OpenAI client initialized.")
    else: print("⚠️ OpenAI client not available (API key missing).")

    # S3에서 최신 원본 CSV 다운로드 (화면 표시용)
    local_csv_path = "downloaded_program_data.csv"
    if not download_from_s3(config.S3_BUCKET_NAME, config.S3_PROGRAM_CSV_KEY, local_csv_path):
        raise RuntimeError("🔥 Failed to download program CSV from S3. Server cannot start.")

    print(f"💿 Loading data from {local_csv_path}.")
    app.state.prog_df = pd.read_csv(local_csv_path, dtype={'program_id':str})
    
    # 로컬에 저장된 임베딩 파일(.npy) 로드 (검색용)
    if not config.ITEMS_NPY_PATH.exists():
        raise FileNotFoundError(f"Embeddings .npy file not found: {config.ITEMS_NPY_PATH}")
    
    print(f"⚡️ Loading pre-computed embeddings from: {config.ITEMS_NPY_PATH}")
    app.state.program_embeddings = np.load(config.ITEMS_NPY_PATH)
    
    print(f"🧠 Loading embedding model: {config.EMBEDDING_MODEL}")
    app.state.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("✅ All models and data are ready.")   

@app.on_event("shutdown")
def shutdown_event():
    """애플리케이션 종료 시 DB 연결 해제"""
    if app.state.mcli is not None:
        app.state.mcli.close()
        print("🔌 MongoDB connection closed.")

# ======================================================================================
# 4. 핵심 서비스 로직 (함수로 분리)
# ======================================================================================
def search_programs(query: str, profession: str, history_turns: List[Dict]) -> List[Dict]:
    """대화 내용 기반으로 관련 진로 체험 프로그램을 검색"""
    context_text = " ".join(turn["query"] for turn in history_turns)
    full_query = f"{context_text} {query} {profession}"
    query_embedding = app.state.embedding_model.encode([full_query], normalize_embeddings=True)[0]
    sims = app.state.program_embeddings @ query_embedding
    
    top_k = getattr(config, 'DEFAULT_TOP_K', 3)
    matches = []
    seen_titles = set()
    num_candidates = top_k * 2 
    top_idx = np.argsort(-sims)[:num_candidates]

    for i in top_idx:
        row = app.state.prog_df.iloc[int(i)].to_dict()
        program_name = row.get("program_title")

        if program_name in seen_titles:
            continue
        
        seen_titles.add(program_name)

        output_data = {
	    "program_id": row.get("program_id"),
            "title": program_name,
            "provider": row.get("provider"),
            "date": row.get("체험일"),
            "program_type": row.get("program_type"),
            "target_audience": row.get("target_audience"),
	    "major": row.get("related_major"),
            "region": row.get("venue_region"),
            "fee": row.get("price"),
            "score": float(sims[int(i)])
        }
        matches.append(output_data)

        if len(matches) >= top_k:
            break
            
    return matches

def build_system_prompt(profession: str) -> str:
    """LLM의 역할을 정의하는 시스템 프롬프트 생성"""
    profession = (profession or "진로 상담 전문가").strip()[:60]
    return (
        f"너는 '{profession}'라는 직업을 가진 전문가야. 중고등학생들에게 진로 상담을 해줘. "
        "어려운 용어 대신 친절하고 따뜻한 말투를 사용해. 학생의 상황에 맞는 실용적인 조언을 해주고, "
        "이전 대화의 흐름을 자연스럽게 이어가줘. 절대로 마크다운 문법은 쓰지 마."
    )

def build_user_prompt(query: str, profile: Dict) -> str:
    """LLM에 전달할 사용자 프롬프트를 생성 (프로그램 추천 언급 없음)"""
    profile_parts = [f"이름:{profile.get('name', '학생')}"]
    if profile.get('grade'): profile_parts.append(f"학년:{profile.get('grade')}")
    if profile.get('interests'): profile_parts.append(f"관심사:{', '.join(profile.get('interests'))}")
    profile_line = " / ".join(profile_parts)
    return (
        f"[학생 프로필] {profile_line}\n"
        f"[학생 질문] {query}\n\n"
        "---"
        "위 학생의 질문에 대해, 설정된 직업인의 입장에서 친절하고 따뜻하게 대답해줘."
    )

def run_llm_chat(system_prompt: str, history_turns: List[Dict], user_prompt: str) -> str:
    """OpenAI LLM을 호출하여 답변 생성"""
    if app.state.oai_client is None: return ""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history_turns:
        messages.append({"role": "user", "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": user_prompt})
    try:
        resp = app.state.oai_client.chat.completions.create(
            model=config.CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"🔥 OpenAI API call failed: {e}")
        return ""

# ⭐️ 함수 인자 변경
def fetch_chat_history(student_id: str, profession: str) -> List[Dict[str, str]]:
    """MongoDB에서 최근 대화 기록 조회"""
    if app.state.history is None:
        return []
    cursor = app.state.history.find(
        # ⭐️ DB 쿼리 필드 변경
        {"type": "chat", "student_id": student_id, "profession": profession}
    ).sort("ts", -1).limit(config.HISTORY_LIMIT)
    history_docs = list(cursor)[::-1]
    return [{"query": h.get("query", ""), "answer": h.get("answer", "")} for h in history_docs]


# 5. API 엔드포인트
# ======================================================================================
@app.get("/healthz")
def health_check():
    """서버 상태 체크 엔드포인트"""
    return {
        "ok": True,
        "mongo_connected": app.state.db is not None,
        "models_loaded": hasattr(app.state, "embedding_model"),
        "program_count": len(app.state.prog_df) if hasattr(app.state, "prog_df") else 0,
    }



# ⭐️ `/recommendations` 엔드포인트 로직 대폭 수정
@app.post("/recommendations", response_model=schemas.RecommendResp)
def get_recommendations(req: schemas.RecommendReq):
    """DB의 채팅 기록 또는 profession을 바탕으로 프로그램을 추천하고 결과를 저장합니다."""
    
    history_turns = fetch_chat_history(req.student_id, req.profession)
    
    # ⭐️ 1. 채팅 기록 유무에 따라 검색 쿼리 결정
    if not history_turns:
        # 기록이 없으면 profession 자체를 검색어로 사용
        search_query = req.profession
        print(f"No chat history for {req.student_id}. Recommending based on profession: {req.profession}")
    else:
        # 기록이 있으면 최근 대화 내용을 검색어로 사용
        search_query = " ".join(turn['query'] for turn in history_turns)
        print(f"Recommending for {req.student_id} based on chat history.")

    all_matches = search_programs(search_query, req.profession, history_turns)
    
    SCORE_THRESHOLD = 0.5
    top_matches = [m for m in all_matches if float(m.get("score", 0)) >= SCORE_THRESHOLD]
    if not top_matches and all_matches:
        top_matches = all_matches[:1]

    # ⭐️ 3. 추천 결과를 DB에 저장
    if top_matches and app.state.recommendations is not None:
        recommendation_id = str(uuid.uuid4()) # 추천 이벤트 고유 ID 생성
        docs_to_save = []
        for match in top_matches:
            docs_to_save.append({
                "program_recommendation_id": recommendation_id,
                "student_id": req.student_id,
		"profession": req.profession,
                "program_id": match.get("program_id"),
                "ts": datetime.now(timezone.utc)
            })
        
        try:
            app.state.recommendations.insert_many(docs_to_save)
            print(f"✅ Saved {len(docs_to_save)} recommendations to DB for student {req.student_id}")
        except Exception as e:
            print(f"🔥 Failed to save recommendations to DB: {e}")

    # ⭐️ 2. 응답 메시지와 함께 최종 결과 반환
    return schemas.RecommendResp(
        message=f"{req.profession} 관련 프로그램을 추천해드릴게요!",
        top_matches=[schemas.ProgramMatch(**m) for m in top_matches]
    )



# ⭐️ 2. 기존 `/chat` 엔드포인트 단순화
@app.post("/chat", response_model=schemas.ChatResp)
def chat(req: schemas.ChatReq):
    """LLM과 대화하고, 그 내용을 DB에 저장합니다. (추천 기능 없음)"""
    
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    history_turns = fetch_chat_history(req.student_id, req.profession)
    
    # --- 추천 관련 로직 모두 제거 ---
    
    profile: Dict[str, Any] = {}
    used_profile = False
    if app.state.users is not None and req.student_id:
        doc = app.state.users.find_one({"_id": req.student_id})
        if doc:
            profile = {**doc, "student_id": req.student_id}; profile.pop("_id", None); used_profile = True
    
    system_prompt = build_system_prompt(req.profession)
    user_prompt = build_user_prompt(query, profile)
    answer = run_llm_chat(system_prompt, history_turns, user_prompt)

    if not answer:
        answer = "미안, 지금은 답변하기 조금 어려워. 잠시 후에 다시 물어봐 줄래?"

    if app.state.history is not None:
        app.state.history.insert_one({
            "type": "chat", "student_id": req.student_id, "profession": req.profession,
            "query": query, "answer": answer, 
            "ts": datetime.now(timezone.utc),
        })

    # ⭐️ 순수하게 답변만 반환
    return schemas.ChatResp(answer=answer)


@app.post("/profile/upsert", status_code=200)
def upsert_profile(p: schemas.Profile):
    """사용자 프로필 생성 또는 업데이트"""
    if app.state.users is None:
        raise HTTPException(503, "MongoDB is not configured")
    # ⭐️ p.student_id 사용
    update_data = p.model_dump(exclude={"student_id"})
    app.state.users.update_one({"_id": p.student_id}, {"$set": update_data}, upsert=True)
    return {"ok": True}

# ⭐️ 경로 변수 및 함수 인자 변경
@app.get("/profile/{student_id}", response_model=schemas.Profile)
def get_profile(student_id: str):
    """특정 사용자 프로필 조회"""
    if app.state.users is None:
        raise HTTPException(503, "MongoDB is not configured")
    # ⭐️ DB 쿼리 필드 변경
    doc = app.state.users.find_one({"_id": student_id})
    if not doc:
        raise HTTPException(404, "Profile not found")
    
    # ⭐️ 반환 필드명 변경
    return schemas.Profile(student_id=doc.pop("_id"), **doc)

@app.post("/log/event", status_code=200)
def log_event(body: schemas.EventIn):
    """클라이언트의 다양한 이벤트를 DB에 기록"""
    if app.state.history is None:
        raise HTTPException(503, "MongoDB is not configured")
    app.state.history.insert_one({
        "type": body.type,
        # ⭐️ body.student_id 사용
        "student_id": body.student_id,
        "payload": body.payload,
        "ts": datetime.now(timezone.utc),
    })
    return {"ok": True}
