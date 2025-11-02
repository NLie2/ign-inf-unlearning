from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env")


client = OpenAI()

with open("batch_inputs/batch_gibberish_words__responses.jsonl","r",encoding="utf-8") as f:
    first = json.loads(f.readline())

assert first["url"] == "/v1/responses"
resp = client.responses.create(**first["body"])
print(resp.output_text)  # should print {"rewritten":"..."}
