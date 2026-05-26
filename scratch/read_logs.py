import json

path = r"C:\Users\elbot\.gemini\antigravity\brain\cbeffb9f-0b8e-4ca0-b7c2-17d022a442a5\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        t = data.get("type")
        c = data.get("content")
        if t in ["USER_INPUT", "PLANNER_RESPONSE"]:
            print(f"=== TYPE: {t} ===")
            print(c)
            print("-" * 40)
