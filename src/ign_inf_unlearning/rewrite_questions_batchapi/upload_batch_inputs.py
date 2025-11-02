# upload_batch_inputs_autodetect.py
import os, json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env")

# === CONFIG ===
exp_name = ["gibberish_words", "real_words_sciency", "nonsensical_biology"][0]
JSONL_PATH = Path(f"batch_inputs/batch_{exp_name}__responses.jsonl")  # set your path
JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# === Detect endpoint from first non-empty line ===
def detect_endpoint(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            url = obj.get("url")
            if url not in ("/v1/chat/completions", "/v1/responses"):
                raise RuntimeError(f"Unsupported per-line url '{url}'. Expected '/v1/chat/completions' or '/v1/responses'.")
            return url
    raise RuntimeError("JSONL appears empty.")

line_url = detect_endpoint(JSONL_PATH)
endpoint = line_url  # batch endpoint must exactly match per-line url

print(f"Detected per-line url: {line_url} -> using batch endpoint: {endpoint}")

# === Optional hard validation: ensure all lines use same url ===
with JSONL_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("url") != line_url:
            raise RuntimeError(f"Line {i} has mismatched url {obj.get('url')} (expected {line_url}).")

# === Upload & create batch ===
client = OpenAI()

with JSONL_PATH.open("rb") as f:
    file_obj = client.files.create(file=f, purpose="batch")
print("Uploaded file:", file_obj.id)

batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/responses",                  # must match JSONL url
    completion_window="24h",
    metadata={"description":f"rewrite_wmdpbio_mpc_{exp_name}__responses"}
)
print("Created batch:", batch.id, "| status:", batch.status)

# === Persist info ===
record = {
    "batch_id": batch.id,
    "input_file_id": file_obj.id,
    "status": batch.status,
    "endpoint": endpoint,
    "description": batch.metadata.get("description"),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "jsonl_path": str(JSONL_PATH),
}
with open("last_batch.json", "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)

expname = JSONL_PATH.stem
with open(f"logs/{expname}_batches.ndjson", "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")

print(f"Saved last_batch.json and appended logs/{expname}_batches.ndjson")
