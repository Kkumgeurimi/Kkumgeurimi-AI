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
S3_PROGRAM_CSV_KEY = "ggoomgil_test_통합.csv" # 👈 config.py와 동일한 CSV 경로

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
        # ⭐️ 1. 'title'을 'program_type'으로 변경
        program_name = str(row.get("program_title", "")) 
        major = str(row.get("related_major", ""))
        # ⭐️ 2. 'description'을 'level_info'로 변경
        level_info = str(row.get("level_info", ""))

        weighted_text = (program_name + " ") * 10 + (major + " ") * 4

        # ⭐️ 3. 새로운 변수명을 사용해 텍스트 조합
        full_text = weighted_text + level_info
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
