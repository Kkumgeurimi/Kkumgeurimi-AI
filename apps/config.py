# apps/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- 기본 경로 설정 ---
# 이 파일(config.py)은 'apps' 폴더 안에 있으므로, .env 파일이 있는
# 프로젝트 루트는 parent.parent로 한 단계 더 올라가야 합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- 데이터 및 임베딩 경로 ---
CSV_PROGRAM_PATH = PROJECT_ROOT / "artifacts" / "data" / "program.csv"
ITEMS_NPY_PATH = PROJECT_ROOT / "artifacts" / "emb" / "items.npy"

# --- 모델 설정 ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHAT_MODEL = "gpt-4o-mini"

# --- OpenAI API 키 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- MongoDB 설정 ---
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = "ggoomgil"
USERS_COLLECTION = "users"
HISTORY_COLLECTION = "history"

# --- RAG/Chat 설정 ---
DEFAULT_TOP_K = 3
HISTORY_LIMIT = 3
