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
INPUT_CSV = "df.csv" # Updated to your likely input source
OUTPUT_DIR = ""
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "df.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MODEL CONFIG
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/llama3_8b_instruct"

# INCREASED tokens because generation needs to be longer than "Yes/No"
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
        save_path = os.path.join(OUTPUT_DIR, f"emergency_save_row_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# 3. Inference Helper
# ==========================================

def truncate_prompt(prompt: str, max_len: int = 7500) -> str:
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
            do_sample=True,      # Enabled sampling for creative generation
            temperature=0.3,     # Low temp for focused but fluent output
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() 

# ==========================================
# 4. Agent Prompts (longer with no tool mentions)
# ==========================================
def get_clarity_agent_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents working with a scientific paper to identify and generate limitations. You are highly curious and have incredible attention to detail, and your job is to scrutinize the paper for limitations related to clarity, such as unclear explanations of methods, experimental settings, key concepts, or organization that could hinder understanding, reproducibility, or generalizability. Identify any missing details, ambiguities, or poorly organized sections that might limit the paper's value or lead to misinterpretation. 

The ’review_leader’ will ask questions regarding your feedback on clarity-related limitations; respond and ask follow-up questions if needed. Scrutinize the paper heavily for hidden ambiguities, unstated assumptions, or vague terms that could undermine reproducibility or comprehension. Think of limitations like undefined concepts, incomplete implementation details (e.g., hyperparameters, equipment specs), or disorganized flow that a reviewer might flag as barriers to replication. Ensure the paper provides all necessary background for methods; if not, note it as a limitation. If unsure about a term or concept, ask for clarification, as unexplained elements represent key limitations. 

When done discussing with the review_leader, inform them you are finished, and provide a summary list of clarity-related limitations, including any missing or misleading information, ambiguous statements, poorly organized points, or suggestions for improvement.

PAPER CONTENT:
{paper_content}"""

def get_impact_agent_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents working with a scientific paper to identify and generate limitations. You are highly curious and skeptical, focusing on limitations in the paper’s novelty, significance, and impact, such as overstated claims, unaddressed assumptions, or weak justifications that could undermine its contributions. Ensure the paper clearly explains motivations and findings without exaggeration; identify limitations like limited scope, lack of comparison to prior work, or potential biases that reduce real-world applicability. 

The ’review_leader’ will ask for your feedback on impact-related limitations; respond and ask follow-up questions if needed. Scrutinize for hidden issues like unsubstantiated goals, ignored alternatives, or overgeneralization that could limit the paper's significance. Consider multiple perspectives, such as how assumptions might fail in different contexts. Think of reviewer questions about poorly justified aspects or what might make claims less novel. Ensure you understand all terms; if unsure, ask, as unclear definitions are limitations. 

When finished, inform the review_leader and provide a summary list of impact-related limitations, including missing justifications, overstated contributions, unaddressed risks, or suggestions to strengthen the paper's claims.

PAPER CONTENT:
{paper_content}"""

def get_experiment_agent_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents working with a scientific paper to identify and generate limitations. You are an expert scientist who evaluates experiments, methodology, and analyses for limitations, such as flaws in design, incomplete ablations, biases, or gaps in analysis that could affect validity, reproducibility, or generalizability. 

When the review_leader asks for help judging existing experiments or suggesting improvements, focus on limitations like inadequate controls, small sample sizes, untested assumptions, or poor metric choices. Ensure you fully understand the paper's claims and goals before critiquing. Ask follow-up questions if needed to clarify. Be detailed in identifying limitations: explain the setup flaws, missing comparisons (e.g., baselines, settings), inappropriate metrics, or analysis gaps, and why they matter (e.g., leading to biased results or limited applicability). Compare the paper's approach to ideal experiments you would design, highlighting shortcomings like why an experiment is misleading or fails to support claims. 

When done, inform the review_leader you are finished, and provide a summary list of experiment-related limitations, including specific issues, why they are problematic, and suggestions for better designs or analyses.

PAPER CONTENT:
{paper_content}"""

def get_review_leader_prompt(clarity_out: str, impact_out: str, experiment_out: str, num_agents: int = 4) -> str:
    return f"""You are part of a group of agents working with a scientific paper to identify and generate limitations.
     You are the review_leader, responsible for compiling a final list of limitations by collaborating with other agents. 
     Delegate tasks involving their expertise (clarity, impact, experiments) and communicate to gather comprehensive limitations. 

To start, draft a high-level plan with a concise list of steps for approaching limitation generation. Then, execute the plan, noting 
the current step as you progress. Create sub-plans for complex steps if needed. Share plans with agents if it guides them. Use multiple 
rounds of communication to ensure thorough coverage; follow up if responses are inadequate, misunderstood, or incomplete, by clarifying 
or discussing further. After sending a message, note what you expect in the response (e.g., a detailed list of limitations). If it doesn't 
match, investigate for errors. 

Information about agents: There are {num_agents} agents, including yourself as review_leader. Others: clarity_agent, experiment_agent, 
impact_agent. 

VERY IMPORTANT: Draft the high-level plan only once at the beginning. Focus on compiling limitations like methodological flaws, clarity 
issues, impact overstatements, and experimental gaps into a cohesive final output.

Here are the reports provided by your agents:

=== CLARITY AGENT REPORT ===
{clarity_out}

=== IMPACT AGENT REPORT ===
{impact_out}

=== EXPERIMENT AGENT REPORT ===
{experiment_out}

Please proceed with compiling the final limitations list."""

==========

print("Loading CSV file...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("CSV not found.")
    sys.exit(1)

# Initialize columns
df["clarity_agent_response"] = ""
df["impact_agent_response"] = ""
df["experiment_agent_response"] = ""
df["leader_final_limitations"] = ""

# Slice if needed (e.g., first 100)
# df = df.iloc[0:100] 
START_INDEX = 0
END_INDEX = len(df) 

# Slice the dataframe for processing
df = df.iloc[START_INDEX:END_INDEX].copy()
global_df = df 

print("Starting Multi-Agent Generation...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    global_current_row = i
    
    # Get text (Assuming 'input_text_cleaned' exists, otherwise use 'pdf_text_without_gt' or similar)
    paper_text = str(row.get("input_text_cleaned", "")) 
    if len(paper_text) < 100:
        continue # Skip empty rows

    try:
        # 1. Run Clarity Agent
        clarity_prompt = get_clarity_agent_prompt(paper_text)
        clarity_res = run_llama(clarity_prompt, system_prompt="You are an expert reviewer focusing on clarity.")
        df.at[i, "clarity_agent_response"] = clarity_res

        # 2. Run Impact Agent
        impact_prompt = get_impact_agent_prompt(paper_text)
        impact_res = run_llama(impact_prompt, system_prompt="You are an expert reviewer focusing on impact.")
        df.at[i, "impact_agent_response"] = impact_res

        # 3. Run Experiment Agent
        exp_prompt = get_experiment_agent_prompt(paper_text)
        exp_res = run_llama(exp_prompt, system_prompt="You are an expert reviewer focusing on experiments.")
        df.at[i, "experiment_agent_response"] = exp_res

        # 4. Run Review Leader (Merge)
        leader_prompt = get_review_leader_prompt(clarity_res, impact_res, exp_res)
        final_res = run_llama(leader_prompt, system_prompt="You are the Review Leader merging agent feedbacks.")
        df.at[i, "leader_final_limitations"] = final_res

    except Exception as e:
        print(f"Error at row {i}: {e}")
        continue

    # Intermediate save every 10 rows
    if i % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False)

# Final Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Saved to {OUTPUT_FILE}")