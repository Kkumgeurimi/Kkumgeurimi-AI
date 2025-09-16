# apps/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import uuid
import tempfile
import os

import json
import re

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from . import config
from . import schemas

# ======= S3 다운로드 함수 ===========
def download_from_s3(bucket: str, key: str) -> Optional[str]:
    """S3에서 파일을 고유한 임시 파일로 다운로드하고, 그 경로를 반환합니다."""
    
    # ⭐️ 겹치지 않는 고유한 이름의 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_f:
        temp_file_path = temp_f.name
    
    print(f"Attempting to download s3://{bucket}/{key} to temporary file {temp_file_path}...")
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, temp_file_path)
        print("✅ Download successful.")
        # ⭐️ 성공 시, 생성된 임시 파일의 경로를 반환
        return temp_file_path
    except (NoCredentialsError, ClientError) as e:
        print(f"🔥 Failed to download from S3: {e}")
        # 실패 시, 생성했던 임시 파일 삭제 후 None 반환
        os.remove(temp_file_path)
        return None

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

    # S3에서 최신 원본 CSV 다운로드
    local_csv_path = None # ⭐️ 변수 초기화
    try:
        # ⭐️ 고유한 임시 파일 경로를 받음
        local_csv_path = download_from_s3(config.S3_BUCKET_NAME, config.S3_PROGRAM_CSV_KEY)
        if not local_csv_path:
            raise RuntimeError("🔥 Failed to download program CSV from S3.")

        print(f"💿 Loading data from {local_csv_path}.")
        app.state.prog_df = pd.read_csv(local_csv_path, dtype={'program_id': str})
        app.state.prog_df = app.state.prog_df.replace({np.nan})
    finally:
        # ⭐️ 작업이 성공하든 실패하든, 사용이 끝난 임시 파일은 반드시 삭제
        if local_csv_path and os.path.exists(local_csv_path):
            os.remove(local_csv_path)
            print(f"✅ Cleaned up temporary file: {local_csv_path}")

    
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

   # ⭐️ 사용자와 AI의 대화를 모두 합쳐서 전체 맥락(context) 생성
    context_parts = []
    for turn in history_turns:
        context_parts.append(turn.get('query', ''))
        context_parts.append(turn.get('answer', ''))
    context_text = " ".join(context_parts)
    
    # 최종 검색어 = 전체 대화 맥락 + 현재 핵심 질문 + 직업
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
            "program_title": program_name,
            "provider": row.get("provider"),
            "start_date": row.get("start_date"),
            "end_date": row.get("end_date"),
            "program_type": row.get("program_type"),
            "target_audience": row.get("target_audience"),
	    "related_major": row.get("related_major"),
            "venue_region": row.get("venue_region"),
            "cost_type": row.get("cost_type"),
        #    "price": row.get("price"),
            "score": float(sims[int(i)])
        }
        matches.append(output_data)

        if len(matches) >= top_k:
            break
            
    return matches

def build_system_prompt(profession: str, name: Optional[str] = None) -> str:
    """LLM의 역할을 정의하는 시스템 프롬프트 생성"""
    profession = (profession or "진로 상담 전문가").strip()[:60]

    rules = (
        "1. 당신의 역할: 당신은 '{profession}' 직업을 가진 전문가이며, 중고등학생들에게 진로 상담을 해주는 커리어 멘토입니다.\n"
        "2. 말투: 항상 친절하고 따뜻한 존댓말을 사용하세요. 어려운 전문 용어는 피하고, 학생의 눈높이에 맞춰 쉽게 설명해야 합니다.\n"
        "3. 대화 맥락 유지: 이전 대화의 흐름을 자연스럽게 이어가세요.\n"
        "4. 절대 규칙: 어떠한 경우에도 마크다운 문법(`*`, `-`, `#`, `1.` 등)을 절대 사용하지 마세요. 모든 답변은 오직 일반 텍스트(Plain Text)와 간단한 줄바꿈으로만 구성해야 합니다."
    ).format(profession=profession)

    # 이름이 전달된 경우, 개인화 규칙 추가
    if name:
        personalization_rule = f"\n5. 개인화: 학생의 이름은 '{name}'입니다. 대화 중에 '{name} 님' 또는 '{name} 학생'처럼 자연스럽게 이름을 불러주며 친근하게 응대해주세요."
        rules += personalization_rule

    # '모범 답안' 예시 제공
    good_example = (
        "### 모범 답변 예시 (이런 스타일로 답변해야 함):\n\n"
        "사용자 질문: 코딩이랑 디자인 둘 다 배우고 싶은데, 뭐부터 할까요?\n\n"
        "당신의 답변: 디자인과 코딩 두 분야에 모두 관심이 있으시군요! 정말 멋진 조합이에요. 두 분야의 공통점인 사용자 경험(UX)에 대해 먼저 알아보는 것을 추천해요. 그 다음으로는 피그마 같은 디자인 툴을 익히고, 마지막으로 간단한 웹사이트를 만들어보면 큰 도움이 될 거예요."
    )

    # 최종 프롬프트 조합
    system_prompt = (
        f"## 지시사항\n{rules}\n\n"
        f"{good_example}"
    )
    
    return system_prompt

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


    # ⭐️ 1. 채팅 기록 유무에 따라 검색 쿼리 결정 (수정된 로직)
    if not history_turns:
        # 기록이 없으면 profession 자체를 검색어로 사용
        search_query = req.profession
        print(f"No chat history for {req.student_id}. Recommending based on profession: {req.profession}")
    else:
        # 기록이 있으면 '가장 마지막 사용자 질문'을 핵심 검색어로 사용
        search_query = history_turns[-1]['query']
        print(f"Recommending for {req.student_id} based on last query: '{search_query}'")

    # ⭐️ 2. search_programs 호출
    # search_query: 현재 핵심 검색어 (목적지)
    # history_turns: 전체 대화 맥락 (이동 경로)
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
    
    system_prompt = build_system_prompt(req.profession, req.name)
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

# ======================================================================================
# 6. 커리어맵 API 엔드포인트
# ======================================================================================
@app.get("/careermap/{student_id}", response_model=schemas.CareerMapResponse)
def get_careermap(student_id: str):
    """학생의 프로그램 참여 기록을 바탕으로 커리어맵 키워드를 생성합니다."""
    
    # --- 1. S3에서 학생 참여 정보 CSV 다운로드 및 처리 ---
    local_participation_csv = None
    
    try:
        # 고유한 임시 파일 경로를 받음
        local_participation_csv = download_from_s3(config.S3_BUCKET_NAME, config.S3_PARTICIPATION_CSV_KEY)
        if not local_participation_csv:
            raise HTTPException(status_code=503, detail="Could not load participation data.")

        participation_df = pd.read_csv(local_participation_csv, dtype={'program_id': str, 'student_id': str})
        
        # 1-1. 해당 학생의 데이터만 필터링
        student_programs_df = participation_df[participation_df['student_id'] == student_id]
        
        if student_programs_df.empty:
            # 참여 기록이 없으면 빈 결과 반환
            return schemas.CareerMapResponse(student_id=student_id, results=[])

        # 1-2. 최신순(program_registration_id가 높은 순)으로 정렬 후 최대 7개 선택
        recent_programs_df = student_programs_df.sort_values(
            by='program_registration_id', ascending=False
        ).head(7)
        
        # 1-3. GPT에 보낼 프로그램 제목 목록과, 나중에 사용할 ID-제목 매핑 생성
        program_info_list = []
        for index, row in recent_programs_df.iterrows():
            title = row.get("program_title", "")
            major = row.get("related_major", "정보 없음")
            program_info_list.append(f"{title} (관련 전공: {major})")

        # 나중에 사용할 ID-제목 매핑은 그대로 유지
        program_id_title_map = pd.Series(
            recent_programs_df.program_id.values, index=recent_programs_df.program_title
        ).to_dict()

    except Exception as e:
        print(f"🔥 Failed to process participation CSV: {e}")
        raise HTTPException(status_code=500, detail="Error processing participation data.")

    finally:
        if local_participation_csv and os.path.exists(local_participation_csv):
            os.remove(local_participation_csv)
            print(f"✅ Cleaned up temporary file: {local_participation_csv}")

    # --- 2. GPT에 전달할 프롬프트 생성 ---
    program_list_str = "\n- ".join(program_info_list)
    prompt_for_gpt = (
        f"당신은 학생들의 활동 기록을 바탕으로 '나의 진로맵'을 생성하는 전문 커리어 컨설턴트입니다.\n"
        f"학생이 최근 참여한 프로그램 목록은 다음과 같습니다:\n- {program_list_str}\n\n"
        f"당신의 목표는 이 활동들을 종합적으로 분석하여, 학생의 핵심 관심사를 나타내는 키워드와 관련 직무를 찾아내는 것입니다.\n\n"
        f"## 작업 지침:\n"
        f"1. **키워드 생성**: 아래 규칙에 따라 1~4개의 '핵심 키워드'를 추출합니다.\n"
        f"    - **규칙 1**: 키워드는 반드시 단어 한 개로 구성해야 합니다.\n"
        f"    - **규칙 2**: 키워드는 진로/직업과 관련된 개념이어야 하지만, '서비스 기획자'와 같은 구체적인 직업명은 절대 키워드로 사용할 수 없습니다. (좋은 예: '서비스 기획', '사용자 경험' / 나쁜 예: '서비스 기획자')\n"
        f"2. **정보 생성**: 각 키워드에 대해 아래 정보를 정리합니다.\n"
        f"   - `keyword_description`: 키워드에 대한 간결한 설명, 3문장 이내.\n"
        f"   - `related_participated_programs`: 제시된 학생 참여 프로그램 목록 중, 해당 키워드와 직접 관련된 프로그램의 '제목'만 정확히 골라 리스트에 담아주세요.\n"
        f"   - `related_jobs`: 키워드와 관련된 직무를 **단 하나만** 제안하고, 소개를 포함해주세요. 소개는 3문장 이내여야 합니다.\n\n"
        f"## 출력 형식 (매우 중요):\n"
        f"반드시 아래의 예시와 동일한 JSON 구조로만 응답해주세요. 절대로 JSON 객체 이외의 다른 설명이나 부가적인 텍스트를 포함하지 마세요.\n\n"
        f"### 예시:\n"
        f"{{\n"
        f'  "keywords": [\n'
        f'    {{\n'
        f'      "keyword": "기획",\n'
        f'      "keyword_description": "사용자에게 필요한 서비스를 구상하고 구체화하여 비즈니스 가치를 만들어내는 과정입니다.",\n'
        f'      "related_participated_programs": ["PM 직무 멘토링", "사용자 리서치 워크샵"],\n'
        f'      "related_jobs": [\n'
        f'       {{ "job_title": "서비스 기획자", "job_description": "사용자 문제를 해결하고 비즈니스 목표를 달성하는 서비스를 설계하는 전문가입니다." }}\n'
        f'      ]\n'
        f'    }},\n'
        f'    {{\n'
        f'      "keyword": "콘텐츠",\n'
        f'      "keyword_description": "글, 이미지, 영상 등 다양한 형태의 콘텐츠를 통해 메시지를 전달하는 활동입니다.",\n'
        f'      "related_participated_programs": ["콘텐츠 마케팅 스쿨"],\n'
        f'      "related_jobs": [\n'
        f'        {{ "job_title": "콘텐츠 마케터", "job_description": "고객에게 유용한 콘텐츠를 통해 브랜드의 가치를 효과적으로 전달하는 전문가입니다." }}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n'
   )

    # --- 3. GPT 호출 및 결과 파싱 ---
    try:
        gpt_response_str = run_llm_chat("", [], prompt_for_gpt)
        
        # ⭐️ 1. GPT의 원본 응답을 그대로 출력해서 확인 (디버깅용)
        print("--- GPT Raw Response ---")
        print(gpt_response_str)
        print("------------------------")
        
        # ⭐️ 2. JSON 부분만 더 똑똑하게 추출 (정규 표현식 사용)
        match = re.search(r"\{.*\}", gpt_response_str, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found in GPT response", gpt_response_str, 0)
        
        json_part = match.group(0)
        gpt_response_json = json.loads(json_part)

    except (json.JSONDecodeError, Exception) as e:
        print(f"🔥 Failed to call or parse GPT response: {e}")
        return schemas.CareerMapResponse(student_id=student_id, results=[])

    # --- 4. GPT 결과를 최종 응답 형식으로 조립 ---
    final_results = []
    for item in gpt_response_json.get("keywords", []):
        # GPT가 반환한 프로그램 제목을 실제 program_id와 매핑
        related_programs_with_id = [
            schemas.ParticipatedProgram(
                program_id=program_id_title_map.get(title),
                program_title=title
            )
            for title in item.get("related_participated_programs", [])
            if title in program_id_title_map # GPT가 없는 프로그램을 지어내지 않도록 방지
        ]
        
        final_results.append(
            schemas.KeywordResult(
                keyword=item.get("keyword"),
                keyword_description=item.get("keyword_description"),
                related_participated_programs=related_programs_with_id,
                related_jobs=[schemas.RelatedJob(**job) for job in item.get("related_jobs", [])]
            )
        )
    
    return schemas.CareerMapResponse(student_id=student_id, results=final_results)


# ======================================================================================	
# 7. 추천 내역 조회 API 엔드포인트
# ======================================================================================
@app.get("/recommendations/history/{student_id}", response_model=schemas.RecommendationHistoryResponse)
def get_recommendation_history(student_id: str):
    """학생 ID로 과거에 추천받았던 프로그램 목록 전체를 '최신순'으로 조회합니다."""
    
    if app.state.recommendations is None:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
        
    # ⭐️ 1. MongoDB에서 'ts'(timestamp)를 기준으로 최신순(-1)으로 정렬하여 조회
    recommended_docs = app.state.recommendations.find(
        {"student_id": student_id}
    ).sort("ts", -1)
    
    # ⭐️ 2. 중복을 제거하면서도 순서를 유지하는 로직
    unique_ordered_ids = []
    seen_ids = set()
    for doc in recommended_docs:
        prog_id = doc.get('program_id')
        if prog_id and prog_id not in seen_ids:
            unique_ordered_ids.append(prog_id)
            seen_ids.add(prog_id)
    
    if not unique_ordered_ids:
        return schemas.RecommendationHistoryResponse(recommended_programs=[])

    # 3. pandas DataFrame에서 정보 가져오기
    recommended_programs_df = app.state.prog_df[app.state.prog_df['program_id'].isin(unique_ordered_ids)]
    
    # ⭐️ 4. 결과를 MongoDB에서 조회한 최신순으로 다시 정렬
    # program_id를 카테고리 타입으로 변환하여 순서를 지정
    recommended_programs_df['program_id'] = pd.Categorical(recommended_programs_df['program_id'], categories=unique_ordered_ids, ordered=True)
    sorted_df = recommended_programs_df.sort_values('program_id')
    sorted_df = sorted_df.replace({np.nan: None})
    
    recommended_programs_list = sorted_df.to_dict('records')

    return schemas.RecommendationHistoryResponse(
        recommended_programs=[schemas.ProgramMatch(**prog) for prog in recommended_programs_list]
    )
