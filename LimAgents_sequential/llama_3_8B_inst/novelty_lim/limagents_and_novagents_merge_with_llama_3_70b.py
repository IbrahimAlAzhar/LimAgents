import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1️⃣ LOAD SINGLE DATAFRAME (UPDATED)
# ==========================================

df = pd.read_csv(INPUT_PATH)


# Output will be saved in the same directory you requested
OUT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_seq_7_agents/llama_3_8B_inst/novelty_lim"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(
    OUT_DIR,
    "df_llm_novelty_agents_limgen_llama3_8b_seq_100_199_merged_with_llama_3_70b.csv"
)

# ==========================================
# 2️⃣ DROP ROWS WHERE EITHER COLUMN IS EMPTY
# ==========================================

# Replace empty strings and whitespace-only strings with NaN
df["final_merged_limitations"] = df["final_merged_limitations"].replace(r"^\s*$", pd.NA, regex=True)
df["novelty_report"] = df["novelty_report"].replace(r"^\s*$", pd.NA, regex=True)

# Drop rows where either column is NaN
df = df.dropna(subset=["final_merged_limitations", "novelty_report"]).reset_index(drop=True)

print(f"Remaining rows after cleaning: {len(df)}")

# ==========================================
# 3️⃣ vLLM CONFIG
# ==========================================

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

MAX_NEW_TOKENS = 1200
TEMPERATURE = 0.2
TIMEOUT = 600

# ==========================================
# 4️⃣ MERGE PROMPT
# ==========================================

def get_merge_prompt(lim1: str, lim2: str) -> str:
    return f"""You are an Expert Scientific Reviewer. I have two sets of paper limitations that may overlap or describe similar issues in different ways.
    
**TASK:**
- Merge the following two lists into one consolidated, high-quality list.
- Group similar points into logical categories (e.g., Experimental Rigor, Novelty, Clarity).
- Remove redundancies while preserving the specific evidence and nuance from both sources.
- Ensure the output is concise, professional, and grounded in scientific terminology.

**INPUT LIST A:**
{lim1}

**INPUT LIST B:**
{lim2}

**OUTPUT FORMAT:**
- Start with: "Consolidated Limitations:"
- Use a bulleted list with bolded categories.
- Do not provide conversational filler or introductory fluff."""


# ==========================================
# 5️⃣ CALL vLLM (LLaMA-3-70B)
# ==========================================

def call_vllm_chat(prompt_text: str):
    url = VLLM_BASE_URL + "/chat/completions"

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise scientific editor. Never add new information."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
    }

    backoff = 2
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue

            r.raise_for_status()
            output = r.json()
            text = output["choices"][0]["message"]["content"]
            return text.strip()

        except Exception as e:
            if attempt == 4:
                print("Final failure:", e)
                return None
            time.sleep(backoff)
            backoff *= 2

    return None


# ==========================================
# 6️⃣ MERGE ROW-WISE USING LLAMA-3-70B
# ==========================================

if "merged_final_output" not in df.columns:
    df["merged_final_output"] = None

for i in tqdm(range(len(df)), desc="Merging with LLaMA-3-70B"):

    if pd.notna(df.loc[i, "merged_final_output"]):
        continue

    lim1 = str(df.loc[i, "final_merged_limitations"])
    lim2 = str(df.loc[i, "novelty_report"])

    prompt = get_merge_prompt(lim1, lim2)
    merged_text = call_vllm_chat(prompt)

    df.loc[i, "merged_final_output"] = merged_text if merged_text else "ERROR"

    # Save checkpoint every 5 rows
    if (i + 1) % 5 == 0:
        df.to_csv(SAVE_PATH, index=False)

    time.sleep(0.5)

df.to_csv(SAVE_PATH, index=False)

print("✅ LLaMA-3-70B Merging Complete (Cleaned Version)")
print(f"Saved to: {SAVE_PATH}")