import os
from openai import OpenAI
from dotenv import load_dotenv
import time, json, os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
BATCH_ID = "batch_68b33aee11b08190bff43cc2d4217db0"  # paste yours


client = OpenAI()

batch = client.batches.retrieve(BATCH_ID)

print(batch)

done = {"completed","failed","expired","cancelled","cancelling"}
last = None
while True:
    b = client.batches.retrieve(BATCH_ID)
    rc = getattr(b, "request_counts", None)
    c = getattr(rc, "completed", 0) if rc else 0
    f = getattr(rc, "failed", 0) if rc else 0
    t = getattr(rc, "total", 0) if rc else 0
    msg = f"{b.status}: {c+f}/{t} done"
    if msg != last:
        print(msg)
        last = msg

    if b.status in done:
        break
    time.sleep(15)

if b.status == "completed" and b.output_file_id:
    out = client.files.content(b.output_file_id).content
    with open("results.jsonl","wb") as fh: fh.write(out)
    print("Wrote results.jsonl")
elif b.error_file_id:
    err = client.files.content(b.error_file_id).content
    with open("errors.jsonl","wb") as fh: fh.write(err)
    print("Wrote errors.jsonl")
else:
    print(f"Batch ended with status={b.status}, no output file.")
