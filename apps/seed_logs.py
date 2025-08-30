# apps/seed_logs.py
import os, time, json, random, pathlib
from pymongo import MongoClient
import pandas as pd

HOME = pathlib.Path.home()
CSV_PROGRAM = os.getenv("CSV_PROGRAM", str(HOME / "hackathon/artifacts/data/program.csv"))
MONGO_URI   = os.getenv("MONGO_URI", "")
MONGO_DB    = os.getenv("MONGO_DB", "ggoomgil")
EVENTS_COL  = os.getenv("EVENTS_COL", "events")

WEIGHTS = {"click":1, "like":3}

client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client[MONGO_DB] if client is not None else None
events = db[EVENTS_COL] if db is not None else None

if events is None:
    print("❌ Mongo is not configured. Set MONGO_URI.")
    raise SystemExit(1)

prog = pd.read_csv(CSV_PROGRAM).fillna("")
# 간단한 필터: 제목/유형에 'AI','코딩','SW','소프트웨어','인공지능' 포함
KEYS = ["AI", "코딩", "소프트웨어", "SW", "인공지능", "로봇", "데이터", "프로그래밍"]
def is_ai_row(r):
    txt = f"{r.get('title','')} {r.get('program_type','')} {r.get('related_major','')}".lower()
    return any(k.lower() in txt for k in KEYS)

ai_df = prog[prog.apply(is_ai_row, axis=1)]
print(f"AI 후보 개수: {len(ai_df)}")

now = int(time.time())
user = "u_ai_lover"

bulk = []
for _, r in ai_df.sample(min(60, len(ai_df))).iterrows():
    pid = str(r.get("program_id",""))
    if not pid: continue
    # like 1~2번, click 1~3번
    for _ in range(random.randint(1,2)):
        bulk.append({"user_id": user, "program_id": pid, "event": "like", "ts": now - random.randint(0, 7*86400)})
    for _ in range(random.randint(1,3)):
        bulk.append({"user_id": user, "program_id": pid, "event": "click","ts": now - random.randint(0, 7*86400)})

if bulk:
    events.insert_many(bulk)
    print(f"✅ insert {len(bulk)} events for {user}")
else:
    print("⚠️ No AI-like items found to seed.")

