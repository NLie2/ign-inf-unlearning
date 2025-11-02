# join_outputs_to_csv.py
import json, re, csv
from pathlib import Path
from typing import Optional
import pandas as pd

# --------- EDIT THESE ---------
PATH_FULL = Path("../datasets/input_data/wmdp-bio_complete_dataset.csv")

PAIRS = [
    {
        "style": "gibberish",
        "input_path":  "batch_inputs/batch_gibberish_words__responses.jsonl",
        "results_path":"batch_outputs/gibberish.jsonl",
        "out_csv":     "batch_outputs/gibberish.csv",
        "index_re":    r"-(\d{7})-",
    },
    {
        "style": "real_words_sciency",
        "input_path":  "batch_inputs/batch_real_words_sciency__responses.jsonl",
        "results_path":"batch_outputs/real_words_sciency.jsonl",
        "out_csv":     "batch_outputs/real_words_sciencycsv",
        "index_re":    r"-(\d{7})-",
    },
    {
        "style": "nonsensical_biology",
        "input_path":  "batch_inputs/batch_nonsensical_biology__responses.jsonl",
        "results_path":"batch_outputs/nonsensical_biology.jsonl",
        "out_csv":     "batch_outputs/nonsensical_biology.csv",
        "index_re":    r"-(\d{7})-",
    },
]
# --------------------------------

OG_RE = re.compile(r'OG:\s*"(.+?)"', flags=re.S)

def load_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_user_content_from_body(body: dict) -> str:
    for key in ("input", "messages"):
        if key in body and isinstance(body[key], list):
            for msg in body[key]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    return content if isinstance(content, str) else ""
    return ""

def parse_inputs_for_original_and_index(input_jsonl_path: str, index_re: str):
    idx_rx = re.compile(index_re)
    cid_to_original = {}
    cid_to_index = {}

    for obj in load_jsonl(input_jsonl_path):
        cid = obj.get("custom_id")
        if not cid:
            continue

        m = idx_rx.search(cid)
        if m:
            try:
                cid_to_index[cid] = int(m.group(1))
            except Exception:
                pass

        body = obj.get("body", {})
        user = extract_user_content_from_body(body)
        if not user:
            continue

        matches = OG_RE.findall(user)
        if matches:
            cid_to_original[cid] = matches[-1].replace('\\"','"').strip()
        else:
            for line in reversed(user.splitlines()):
                line = line.strip()
                if line.startswith("OG:"):
                    q = line[3:].strip()
                    if len(q) >= 2 and q[0] == '"' and q[-1] == '"':
                        q = q[1:-1]
                    cid_to_original[cid] = q.replace('\\"','"').strip()
                    break

    return cid_to_original, cid_to_index

def extract_text_from_responses_body(body: dict) -> str:
    chunks = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    chunks.append(c.get("text", ""))
    return "".join(chunks).strip()

def parse_results_for_rewritten(results_jsonl_path: str, index_re: str):
    mapping = {}
    idx_rx = re.compile(index_re)
    for obj in load_jsonl(results_jsonl_path):
        cid = obj.get("custom_id")
        resp = obj.get("response", {})
        if resp.get("status_code") != 200:
            continue
        body = resp.get("body", {})
        text = extract_text_from_responses_body(body)
        if not text:
            continue
        try:
            data = json.loads(text)
            rewritten = data.get("rewritten", "").strip()
        except Exception:
            rewritten = text.strip()
        if cid and rewritten:
            # Extract index from custom_id
            m = idx_rx.search(cid)
            if m:
                try:
                    index = int(m.group(1))
                    mapping[index] = rewritten
                except Exception:
                    pass
    return mapping

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("/", "")
                  .str.replace(r"[()]", "", regex=True))
    return df

def detect_question_col(df: pd.DataFrame) -> str:
    for name in ["question","questions","og_question","text","prompt"]:
        if name in df.columns:
            return name
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        return obj_cols[0]
    raise RuntimeError("Could not detect a question column in the full dataset.")

def detect_option_cols(df: pd.DataFrame):
    cols = list(df.columns)
    preferred_single = ["answer_options","options","choices","answers"]
    for c in preferred_single:
        if c in df.columns:
            return [c]

    candidates = []
    for c in cols:
        name = c.lower()
        if (
            name in {"a","b","c","d","e","f"} or
            re.match(r"^(option|choice|answer)[ _\-]?[a-f]$", name) or
            name.startswith("option_") or
            name.startswith("choice_") or
            name.startswith("answer_")
        ):
            candidates.append(c)

    def sort_key(c):
        m = re.search(r"([a-f])$", c.lower())
        return (c.lower(), m.group(1) if m else "~")

    return sorted(set(candidates), key=sort_key)

def detect_correct_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["correct","label","answer","correct_answer","gold","target"]:
        if c in df.columns:
            return c
    return None

def join_options_for_row(row: pd.Series, option_cols: list) -> str:
    vals = []
    for c in option_cols:
        v = row.get(c, None)
        if pd.notna(v):
            v = str(v).strip()
            if v:
                vals.append(v)
    return " | ".join(vals)

def process_pair(style: str, input_path: str, results_path: str, out_csv: str,
                 index_re: str, df_full: pd.DataFrame, option_cols: list,
                 correct_col: Optional[str]):
    cid_to_original, cid_to_index = parse_inputs_for_original_and_index(input_path, index_re)
    index_to_rewritten = parse_results_for_rewritten(results_path, index_re)

    # Match by index instead of custom_id
    keys = sorted(set(cid_to_index.values()) & set(index_to_rewritten.keys()))
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["custom_id","style","index","original","rewritten","options"]
        if correct_col:
            header.append("correct")
        w.writerow(header)

        for idx in keys:
            # Find the custom_id for this index
            cid = None
            for cid_val, idx_val in cid_to_index.items():
                if idx_val == idx:
                    cid = cid_val
                    break
            
            options_str = ""
            correct_val = ""
            if 0 <= idx < len(df_full):
                row = df_full.iloc[idx]
                options_str = join_options_for_row(row, option_cols)
                if correct_col:
                    cv = row.get(correct_col, "")
                    correct_val = "" if pd.isna(cv) else str(cv)

            w.writerow([
                cid if cid else f"{style}-{idx:07d}",
                style,
                idx,
                cid_to_original.get(cid, "") if cid else "",
                index_to_rewritten[idx],
                options_str
            ] + ([correct_val] if correct_col else []))

    print(f"[{style}] wrote {out_csv} ({len(keys)} rows)")
    missing_in_results = set(cid_to_index.values()) - set(index_to_rewritten.keys())
    missing_in_inputs  = set(index_to_rewritten.keys()) - set(cid_to_index.values())
    if missing_in_results:
        print(f"  - {len(missing_in_results)} inputs had no result (see errors.jsonl).")
    if missing_in_inputs:
        print(f"  - {len(missing_in_inputs)} results had no matching input (custom_id mismatch?).")

if __name__ == "__main__":
    df_full = normalize_cols(pd.read_csv(PATH_FULL))
    _ = detect_question_col(df_full)  # sanity
    option_cols = detect_option_cols(df_full)
    correct_col = detect_correct_col(df_full)

    if not option_cols:
        print("Warning: no option columns detected. The 'options' field will be empty.")

    for p in PAIRS:
        process_pair(
            style=p["style"],
            input_path=p["input_path"],
            results_path=p["results_path"],
            out_csv=p["out_csv"],
            index_re=p["index_re"],
            df_full=df_full,
            option_cols=option_cols,
            correct_col=correct_col
        )
