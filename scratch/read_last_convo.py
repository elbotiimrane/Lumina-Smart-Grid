import json
import sys

path = r"C:\Users\elbot\.gemini\antigravity\brain\76d478b9-276d-4938-8d5d-b10a657275de\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        t = data.get("type")
        c = data.get("content")
        s = data.get("source")
        if t in ["USER_INPUT", "PLANNER_RESPONSE"] and c:
            print(f"=== SOURCE: {s} | TYPE: {t} ===")
            safe_c = c[:3000].encode('ascii', 'ignore').decode('ascii')
            print(safe_c)
            print("-" * 50)
