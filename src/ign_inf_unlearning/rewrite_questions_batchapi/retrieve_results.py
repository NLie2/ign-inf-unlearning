# download_results_jsonl.py
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

OUTPUT_FILE_ID =  "file-BHrrcumozqJJhJpc6uedTm"
exp_name = ["gibberish_words", "real_words_sciency", "nonsensical_biology"][1]
OUT_PATH = f"batch_outputs/{exp_name}.jsonl"

blob = client.files.content(OUTPUT_FILE_ID).content  # bytes
with open(OUT_PATH, "wb") as f:
    f.write(blob)

print(f"Saved {OUT_PATH}")
