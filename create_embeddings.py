# create_embeddings.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from sentence_transformers import SentenceTransformer

# --- 설정 (Configuration) ---
S3_BUCKET_NAME = "ggoomgil-raw"  # 👈 config.py와 동일한 S3 버킷 이름
S3_PROGRAM_CSV_KEY = "ggoomgil_surface_seongnam_with_category.csv" # 👈 config.py와 동일한 CSV 경로

OUTPUT_DIR = Path("./artifacts/emb")
OUTPUT_FILENAME = OUTPUT_DIR / "items.npy"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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

def process_and_combine_text(csv_path: str) -> list[str]:
    print("Processing CSV and combining text for embedding...")
    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        major = str(row.get("체험직무학과", ""))

        # ⭐️ 1. 상세 설명을 description 변수로 가져옵니다.
        description = str(row.get("수준별 정보", ""))

        # 2. 기존처럼 제목과 전공을 3번 반복해서 핵심 정보(core_info)를 만듭니다.
        core_info = " ".join([title,major] * 3)
        
        # ⭐️ 3. 핵심 정보 뒤에 상세 설명을 합쳐서 최종 텍스트를 만듭니다.
        full_text = core_info + " " + description
        texts.append(full_text.strip())
        
    print(f"✅ Combined text for {len(texts)} programs.")
    return texts


def embed_texts(texts: list[str]) -> np.ndarray:
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding texts...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print("✅ Embedding complete.")
    return embeddings

def save_embeddings(embeddings: np.ndarray, output_path: Path):
    print(f"Saving embeddings to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print("✅ Saved successfully.")

if __name__ == "__main__":
    local_csv_path = "temp_program_data.csv"
    if download_from_s3(S3_BUCKET_NAME, S3_PROGRAM_CSV_KEY, local_csv_path):
        combined_texts = process_and_combine_text(local_csv_path)
        program_embeddings = embed_texts(combined_texts)
        save_embeddings(program_embeddings, OUTPUT_FILENAME)
        os.remove(local_csv_path)
        print("🚀 Embedding creation complete.")
