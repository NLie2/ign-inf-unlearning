from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env")


client = OpenAI()

client = OpenAI()

client.batches.cancel("batch_68ab24d8a41c8190a69e773e1cd10237")