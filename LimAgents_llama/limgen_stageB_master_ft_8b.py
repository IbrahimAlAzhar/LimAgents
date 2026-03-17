import os
import time
import json
import pandas as pd
import requests
from tqdm import tqdm

# =========================
# Config
# =========================

LEADER_COL = "leader_final_output"
OUT_COL = "final_merged_limitations"

# Your fine-tuned master vLLM endpoint
VLLM_BASE_URL_MASTER = os.environ.get("VLLM_BASE_URL_MASTER", "http://127.0.0.1:8001/v1").rstrip("/")
VLLM_MODEL_MASTER = os.environ.get("VLLM_MODEL_MASTER", "llama3-8b-master-ft")

MAX_TOKENS = 1500
TEMPERATURE = 0.0
TIMEOUT = 600

MASTER_SYSTEM_PROMPT = (
    "You are the **Master Agent**. You will receive the Leader Agent’s collected limitation analyses.\n"
    "Your job: produce ONE consolidated, non-redundant list of limitations.\n"
    "Rules:\n"
    "- Merge duplicates.\n"
    "- Keep specificity and evidence.\n"
    "- Do NOT invent new limitations beyond what is provided.\n"
    "Output format:\n"
    "Start with: \"Here is the consolidated list of key limitations identified in the paper:\"\n"
    "Then bullets grouped by category.\n"
)

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

def chat_completion(base_url: str, model: str, system: str, user: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def run():
    os.makedirs(os.path.dirname(OUTPUT_STAGEB), exist_ok=True)

    df = pd.read_csv(INPUT_STAGEA)
    if OUT_COL not in df.columns:
        df[OUT_COL] = "PENDING"

    for i in tqdm(range(len(df)), desc="StageB Master FT (8B)"):
        leader_text = str(df.at[df.index[i], LEADER_COL] or "").strip()

        if not leader_text or leader_text.startswith("ERROR") or leader_text in ["NO_OUTPUT_FROM_LEADER", "SKIPPED_SHORT_TEXT"]:
            df.at[df.index[i], OUT_COL] = f"SKIPPED_STAGEB: {leader_text[:200]}"
            continue

        user_msg = (
            "Leader Agent output is below.\n"
            "Please consolidate into a single final list, removing redundancy and grouping by category.\n\n"
            "=== LEADER OUTPUT START ===\n"
            f"{leader_text}\n"
            "=== LEADER OUTPUT END ===\n"
        )

        try:
            out = chat_completion(
                base_url=VLLM_BASE_URL_MASTER,
                model=VLLM_MODEL_MASTER,
                system=MASTER_SYSTEM_PROMPT,
                user=user_msg,
            )
            df.at[df.index[i], OUT_COL] = out.strip()

        except Exception as e:
            df.at[df.index[i], OUT_COL] = f"ERROR_STAGEB: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_STAGEB, index=False)

        time.sleep(0.2)

    df.to_csv(OUTPUT_STAGEB, index=False)
    print("✅ Saved StageB output:", OUTPUT_STAGEB)

if __name__ == "__main__":
    print("Using master vLLM:", VLLM_BASE_URL_MASTER, "| model:", VLLM_MODEL_MASTER)
    if not wait_for_vllm(VLLM_BASE_URL_MASTER, timeout_s=600):
        raise RuntimeError(f"Master vLLM not ready at {VLLM_BASE_URL_MASTER}")
    run()
