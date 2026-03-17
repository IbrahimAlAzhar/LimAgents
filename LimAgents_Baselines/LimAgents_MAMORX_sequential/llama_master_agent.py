import pandas as pd
import os
import torch
import time
import signal
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ==========================================
# 1. Configuration & Model Loading
# ==========================================

# PATHS
# We read the file that already has the agent outputs
INPUT_CSV = "df.csv" 
# We save to the same folder. You can overwrite the file or create a _merged version.
OUTPUT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_agents_2"
# Saving as a new file first is safer. You can rename it later if you want to replace the original.
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MODEL CONFIG
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/llama3_8b_instruct"

MAX_NEW_TOKENS = 1024 
MAX_CONTEXT_TOKENS = 8000

print(f"Loading Llama 3 model from {MODEL_ID}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. Signal Handling (Graceful Exit)
# ==========================================
global_df = None
global_current_row = 0

def signal_handler(signum, frame):
    print(f"\n⚠️  Received signal {signum}. Saving progress...")
    if global_df is not None:
        save_path = os.path.join(OUTPUT_DIR, f"emergency_merge_save_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# 3. Inference Helper
# ==========================================

def truncate_prompt(prompt: str, max_len: int = 7300) -> str:
    """Truncates input text to leave room for generation."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_len:
        return tokenizer.decode(input_ids[:max_len], skip_special_tokens=True)
    return prompt

def run_llama(prompt: str, system_prompt: str = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Prepare input
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_input = truncate_prompt(text_input)
    
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,      
            temperature=0.3,     
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() 

def get_review_leader_prompt(clarity_out: str, impact_out: str, experiment_out: str) -> str:
    return f"""You are the **Review Leader** for a top-tier scientific conference (e.g., ICLR). You have just received detailed analysis reports from three specialized agents (Clarity, Impact, and Experiment) regarding a specific paper.

**Your Objective:**
Synthesize the three reports below into a single, final, comprehensive list of limitations.
1. **Merge redundancies:** If multiple agents identified the same issue (e.g., "missing hyperparameters"), combine them into one strong point.
2. **Filter noise:** Remove minor nitpicks that do not affect the paper's validity.
3. **Format professionally:** Output **ONLY** the final numbered list. Do not include plans, steps, or conversational filler.

**Input Reports:**

=== CLARITY AGENT REPORT ===
{clarity_out}

=== IMPACT AGENT REPORT ===
{impact_out}

=== EXPERIMENT AGENT REPORT ===
{experiment_out}

**Final Output:**
Provide the final consolidated list of limitations below.""" 

# ==========================================
# 4. Main Processing
# ==========================================

print("Loading CSV file...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from: {INPUT_CSV}")
except FileNotFoundError:
    print(f"CSV not found at: {INPUT_CSV}")
    sys.exit(1)

# Ensure the column exists
if "leader_merger_limitations" not in df.columns:
    df["leader_merger_limitations"] = ""

# Since you are loading the "100_199" file, we process ALL rows in this dataframe.
# (Indices will be 0 to 99 relative to this dataframe)
global_df = df 

print("Starting Review Leader (Merger) Generation...")

# Using tqdm to track progress
for i, row in tqdm(df.iterrows(), total=len(df)): # len(df)
    global_current_row = i
    
    # 1. Retrieve the existing outputs from the row
    # Using row.get() handles NaNs or missing keys gracefully
    clarity_res = str(row.get('clarity_agent_response', '')) 
    print('clarity_res:', clarity_res)
    impact_res = str(row.get('impact_agent_response', ''))
    print('impact_res:', impact_res)
    exp_res = str(row.get('experiment_agent_response', '')) 
    print('exp_res:', exp_res)

    # Skip if the agents didn't generate anything meaningful (optional check)
    if len(clarity_res) < 10 and len(impact_res) < 10 and len(exp_res) < 10:
        print(f"Skipping row {i}: No input data from agents.")
        continue

    try:
        # 2. Run Review Leader
        leader_prompt = get_review_leader_prompt(clarity_res, impact_res, exp_res)
        final_res = run_llama(leader_prompt, system_prompt="You are the Review Leader merging agent feedbacks.")
        print(f"Row {i} - Merged Limitations:\n{final_res}\n")
        
        # 3. Store result
        df.at[i, "leader_merger_limitations"] = final_res

    except Exception as e:
        print(f"Error at row {i}: {e}")
        continue

    # Intermediate save every 5 rows
    if i % 5 == 0:
        df.to_csv(OUTPUT_FILE, index=False)

# Final Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Final merged file saved to: {OUTPUT_FILE}")