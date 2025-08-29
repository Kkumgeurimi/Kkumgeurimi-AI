# seed_logs.py
import os, time, re
import pandas as pd
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수를 설정하세요.")

client = MongoClient(MONGO_URI)
db_name = MONGO_URI.split("/")[-1].split("?")[0]
db = client[db_name]
events = db["events"]

CSV_PROGRAM = os.path.expanduser("~/hackathon/artifacts/data/program.csv")
prog = pd.read_csv(CSV_PROGRAM, dtype=str)
prog.fillna("", inplace=True)

def find_ids_by_keywords(keywords):
    ids = []
    for _, r in prog.iterrows():
        title = (r.get("title") or "").lower()
        pid = str(r.get("program_id") or "").strip()
        if not pid:
            continue
        if any(kw in title for kw in keywords):
            ids.append(pid)
    return list(dict.fromkeys(ids))  # unique order

now = int(time.time())

def bulk_insert(user_id, program_ids, n_click=6, n_like=3):
    logs = []
    for pid in program_ids[:8]:  # 너무 많지 않게 선별
        # 클릭 여러 번 + 좋아요 몇 번
        for i in range(n_click):
            logs.append({"type": "click", "user_id": user_id, "program_id": pid, "ts": now - (i * 120 + 5)})
        for j in range(n_like):
            logs.append({"type": "like", "user_id": user_id, "program_id": pid, "ts": now - (j * 240 + 15)})
    if logs:
        events.insert_many(logs)
        print(f"✅ {user_id}: {len(logs)} logs inserted for {len(program_ids[:8])} programs")
    else:
        print(f"⚠️ {user_id}: matching programs not found, no logs inserted")

# u_1: 바리스타/제과제빵 계열
barista_ids = find_ids_by_keywords(["바리스타", "카페", "커피", "제과", "제빵"])
bulk_insert("u_1", barista_ids)

# u_2: AI/코딩/로봇 계열
coding_ids = find_ids_by_keywords(["ai", "인공지능", "코딩", "파이썬", "python", "로봇", "로보", "sw", "소프트웨어"])
bulk_insert("u_2", coding_ids)

# u_3: 콜드스타트 — 아무것도 안 넣음
print("✅ u_3: cold start (no logs)")

