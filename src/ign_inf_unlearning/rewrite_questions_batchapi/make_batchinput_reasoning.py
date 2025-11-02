# make_batchinput_responses.py
import json, uuid
from pathlib import Path
import pandas as pd

PATH_SAMPLE = Path("../datasets/input_data/wmdp_bio_ignorance_sample_rewritten_questions.csv")
PATH_FULL   = Path("../datasets/input_data/wmdp-bio_complete_dataset.csv")
OUT_DIR = Path("./batch_inputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5-mini"
API_URL = "/v1/responses"
N_EXAMPLES = 3                 # keep small to reduce thinking
MAX_OUTPUT_TOKENS = 500         # visible output cap (not counting reasoning)
REASONING_EFFORT = "low"       # <- **key**


def norm(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("/", "")
                  .str.replace(r"[()]", "", regex=True))
    return df

def find_col(cols, *bits):
    for c in cols:
        name = c.lower()
        if all(b in name for b in bits): return c
    return None

def examples_block(df, og_col, style_col, n=3):
    rows = df[[og_col, style_col]].dropna().head(n)
    lines = ["EXAMPLES"]
    for _, row in rows.iterrows():
        og  = str(row[og_col]).strip().replace('"','\\"')
        out = str(row[style_col]).strip().replace('"','\\"')
        lines += [f'OG: "{og}"', f'IGNORANT: "{out}"', ""]
    lines.append("END EXAMPLES")
    return "\n".join(lines)

def system_text():
    return ("You rewrite questions.\n"
            "- Keep grammar/structure close to the original.\n"
            "- Target biology terms; leave scaffolding intact.\n"
            "- No new facts or explanations.\n"
            "- Output JSON only: {\"rewritten\":\"<text>\"}.")

def user_text(examples, q):
    q = (q or "").strip().replace('"','\\"')
    return f"""{examples}

Now rewrite this.
OG: "{q}"
"""

def body(system, user):
    return {
        "model": MODEL,
        "input": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ],
        "text": {  # <-- use text.format for structured output
            "format": {
                "type":"json_schema",
                "name":"Rewrite",
                "schema":{
                    "type":"object",
                    "properties":{"rewritten":{"type":"string"}},
                    "required":["rewritten"],
                    "additionalProperties":False
                },
                "strict": True
            }
        },
        "reasoning": {"effort": REASONING_EFFORT},  # <-- **limits hidden reasoning**
        "max_output_tokens": MAX_OUTPUT_TOKENS      # <-- visible output budget
    }

def jsonl_line(custom_id, url, body):
    return json.dumps({"custom_id":custom_id,"method":"POST","url":url,"body":body}, ensure_ascii=False)

# load data
samples = norm(pd.read_csv(PATH_SAMPLE))
full    = norm(pd.read_csv(PATH_FULL))

og_col = find_col(samples.columns, "og","question") or "og_question"
gibberish_col = find_col(samples.columns, "ignorant","gibberish")
if not (og_col and gibberish_col):
    raise RuntimeError("Could not find OG and gibberish columns in sample CSV.")

# optional: drop footer rows (e.g., **Description**)
mask_desc = samples[og_col].astype(str).str.strip().str.lower().str.contains("**description**", regex=False)
samples = samples[~mask_desc]

ex_gibb = examples_block(samples, og_col, gibberish_col, N_EXAMPLES)
sys = system_text()

# questions column
for guess in ["question","questions","og_question","text"]:
    if guess in full.columns:
        q_series = full[guess]
        break
else:
    raise RuntimeError(f"Could not find a question column in {list(full.columns)}")

out_path = OUT_DIR / "batch_gibberish_words__responses.jsonl"
count = 0
with out_path.open("w", encoding="utf-8") as f:
    for i, q in enumerate(q_series):
        if not isinstance(q, str) or not q.strip():
            continue
        b = body(sys, user_text(ex_gibb, q))
        f.write(jsonl_line(f"gib-{i:07d}-{uuid.uuid4().hex[:6]}", API_URL, b) + "\n")
        count += 1

print(f"Wrote {count} requests → {out_path}")
