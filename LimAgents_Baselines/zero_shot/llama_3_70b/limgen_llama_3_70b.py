import os
import sys
import time
import re
import json
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# =========================
# vLLM readiness check
# =========================
def wait_for_vllm(base_url: str, timeout_s: int = 600) -> bool:
    t0 = time.time()
    base_url = base_url.rstrip("/")
    models_url = base_url + "/models"
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(models_url, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

# =========================
# 1) Config
# =========================
INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/data/final_not_balanced_data/df_not_bal_prep_gt_final_mist_dec_from_gpt_gem_0_7k_rows.csv"
OUTPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/zero_shot/llama_3_70b_two_cl_not_bal_data/df_zs_llama31_70b_0_7k_rows.csv"

TEXT_COL = "input_text_without_lim"
MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# ---- Token budgets ----
MAX_CTX = 92_000
RESERVED_FOR_GEN = 2_000
RESERVED_FOR_PROMPT_OVERHEAD = 1_000 
# Maximum allowed for the input text itself
INPUT_BUDGET = MAX_CTX - RESERVED_FOR_GEN - RESERVED_FOR_PROMPT_OVERHEAD

TEMPERATURE = 0.2
TIMEOUT = 600
MAX_NEW_TOKENS = 1_500

SAVE_EVERY = 30
SLEEP_S = 0.5

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# =========================
# 2) Token helpers
# =========================
def tok_len(text: str) -> int:
    return len(tokenizer.encode(text or ""))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])

# =========================
# 3) vLLM chat call
# =========================
def call_vllm_chat(base_url: str, model: str, prompt_text: str, max_tokens: int, temperature: float, timeout_s: int):
    url = base_url.rstrip("/") + "/chat/completions"

    messages = [
        {"role": "user", "content": f"Generate limitations from this text.\n\nTEXT:\n{prompt_text}"},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    backoff = 2
    for attempt in range(6):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(30, backoff * 2)
                continue
            r.raise_for_status()
            j = r.json()
            text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            return (text or "").strip(), j
        except Exception:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(30, backoff * 2)
    return "", {}

# =========================
# 4) Main pipeline
# =========================
def run_pipeline_zero_shot():
    df_all = pd.read_csv(INPUT_CSV)
    
    # Define range (0 to 6957)
    idx_to_run = [i for i in range(0, 6958) if i < len(df_all)]
    
    # Work on the specific subset
    df = df_all.iloc[idx_to_run].copy()

    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    for k, orig_i in enumerate(tqdm(idx_to_run, desc="Processing LLM Decisions")):
        # Get raw text from the specified column
        raw_text = str(df_all.iloc[orig_i].get(TEXT_COL, "") or "").strip()

        # Skip if text is basically empty
        if len(raw_text) < 10:
            df.iloc[k, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_EMPTY_TEXT"
            continue

        # Truncate only if it exceeds the model's limits
        input_text = truncate_to_tokens(raw_text, INPUT_BUDGET)
        input_tokens = tok_len(input_text)

        try:
            output_text, raw_json = call_vllm_chat(
                base_url=VLLM_BASE_URL,
                model=VLLM_MODEL,
                prompt_text=input_text,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                timeout_s=TIMEOUT,
            )

            df.iloc[k, df.columns.get_loc("final_merged_limitations")] = output_text if output_text else "EMPTY_OUTPUT"
            df.iloc[k, df.columns.get_loc("full_chat_history")] = json.dumps(raw_json)[:100000]

        except Exception as e:
            df.iloc[k, df.columns.get_loc("final_merged_limitations")] = f"ERROR: {e}"

        # Periodic Save
        if k % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(SLEEP_S)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Final results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=600):
        raise RuntimeError(f"vLLM server not found at {VLLM_BASE_URL}")
    
    run_pipeline_zero_shot()



