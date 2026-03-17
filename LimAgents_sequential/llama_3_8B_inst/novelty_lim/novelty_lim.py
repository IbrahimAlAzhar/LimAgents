#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential 8-agent pipeline using **Meta-Llama-3-8B-Instruct (HF Transformers)**.

What this script does (same as your Code-1 logic, but NO AutoGen / NO vLLM):
1) For each row:
   - Parse relevant_papers_list
   - Summarize top-3 retrieved items into 6-dim “Paper B Summary”
   - Build combined input: Paper A + Paper B summaries (truncated to 8B context)
2) Run agents **sequentially** (one at a time):
   Planning -> Lit -> Hyp -> Meth -> Exp -> Prob -> Write -> Master
3) Save outputs to CSV, checkpoint every N rows.

PROMPTS ARE KEPT THE SAME (copied from your first code).
"""

import os
import sys
import time
import ast
import json
import re
import signal
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# 0) Global guardrails (same as your code)
# ============================================================

HARSH_REVIEWER_POLICY = """
You are an extremely strict, harsh peer reviewer.

STRICT REVIEW MODE:
- Default assumption: the paper has limitations unless it provides clear, concrete evidence.
- Prefer flagging possible weaknesses rather than giving benefit of the doubt.
- Focus on limitations that undermine novelty, significance, credibility, rigor, or generalizability.

WHAT TO PRODUCE:
- Only limitations-focused content (no scores, no ratings).
- Evidence pointers from Paper A / Paper B excerpts when possible.
""".strip()

NO_TOOL_NO_JSON = """
CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- Do NOT output JSON, braces {}, or any {"name": ...} structures.
- Do NOT mention "parameters", "function_call", "tool", or "tools".
- Do NOT call tools or pretend to call tools.
""".strip()

# ============================================================
# 1) Prompts (UNCHANGED from your first code)
# ============================================================

planning_agent = '''
You are the Planning Agent in a multi-agent system that identifies limitations in the novelty and significance of a main research paper.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- Do NOT output JSON, braces {}, or any {"name": ...} structures.
- Do NOT mention "parameters", "function_call", "tool", or "tools".
- Do NOT call tools or pretend to call tools.

Task:
- Summarize the paper’s core claims, contributions, problem framing, methods, and stated significance.
- Identify the most promising areas where novelty or real-world/ scientific significance appear weak, overstated, incremental, or poorly justified.
- Prioritize subtasks for specialist agents to dig into specific limitation dimensions.
- IMPORTANT: your subtasks MUST explicitly reference comparing Paper A to the summarized Paper B evidence.

HARSH LIMITATION MODE (STRICT):
Assume limitations exist unless the paper provides clear, concrete evidence of substantial differentiation from prior ideas and meaningful broader impact. Penalize re-framing, vague “novel/first” claims, narrow scope, and absence of justification for why the work matters. Prefer false positives (flagging limitations) over false negatives.

OUTPUT FORMAT (STRICT):
Key Novelty & Significance Limitations:
- <bullet 1>
- <bullet 2>
- <bullet 3-5 max>

Evidence:
- <short direct pointers / quotes from Paper A and Paper B summaries>

Prioritized subtasks:
1) <task + how to compare vs Paper B + rationale>
2) <task + how to compare vs Paper B + rationale>
...
'''.strip()

PROMPT_1_NOVELTY_TECH = """
Prompt 1: Technical Contributions & Incremental Nature
Identify limitations in the paper’s technical contributions that undermine claims of novelty or significance. Focus on whether ideas are rebranded existing methods, minor tweaks, combinations of known components, or lack substantive advancement beyond prior work.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Technical Contributions (A vs B):
- <bullet 1 limitation with explanation; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary showing overlap/difference>
""".strip()

PROMPT_2_EXPERIMENTS = """
Prompt 2: Experimental Validation & Comparative Analysis
Identify limitations in experimental design, benchmarking, or comparative analysis that weaken the paper’s novelty or claimed significance (e.g., missing strong baselines, inadequate datasets, no ablation, overstated improvements).

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Experimental Validation (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary showing stronger/weaker evidence>
""".strip()

PROMPT_3_LIT_REVIEW = """
Prompt 3: Literature Review & Contextualization
Identify limitations in the literature review or positioning that undermine perceived novelty or significance (overlooking key prior work, vague differentiation, failure to explain why the gap matters).

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Literature Review & Contextualization (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary that should have been cited/contrasted>
""".strip()

PROMPT_4_SCOPE_GENERALIZABILITY = """
Prompt 4: Scope of Analysis & Generalizability
Identify limitations in scope, datasets, tasks, or discussed implications that restrict the work’s broader significance or generalizability (narrow domain, toy settings, ignored real-world constraints).

CRITICAL OUTPUT RULES (NON-NEGOTOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Scope & Generalizability (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary with broader evaluation/scope>
""".strip().replace("NON-NEGOTOTIABLE", "NON-NEGOTIABLE")  # keep text same-ish

PROMPT_5_CLAIMS_OVERCLAIMING = """
Prompt 5: Claim Accuracy & Overclaiming
Identify limitations stemming from overstated novelty, impact, effectiveness, or importance claims that lack supporting evidence or ignore caveats.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Claims & Overclaiming (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary contradicting/qualifying the claim>
""".strip()

PROMPT_6_METHOD_CLARITY_RIGOR = """
Prompt 6: Methodological Clarity & Rigor
Identify limitations in methodological description, reproducibility, or rigor that erode confidence in the claimed novelty or significance (missing details, ambiguous setups, unverifiable experiments).

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Methodological Clarity & Rigor (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary showing clearer method/repro details>
""".strip()

literature_review_and_data_analysis_agent = PROMPT_3_LIT_REVIEW
hypothesis_refinement_and_critical_reflection_agent = PROMPT_1_NOVELTY_TECH
methodological_novelty_agent = PROMPT_6_METHOD_CLARITY_RIGOR
experimental_novelty_agent = PROMPT_2_EXPERIMENTS
problem_Formulation_novelty_Agent = PROMPT_4_SCOPE_GENERALIZABILITY
writing_claim_novelty_agent = PROMPT_5_CLAIMS_OVERCLAIMING

master_agent_prompt = """
You are the Master Agent. You receive:
- Planning Agent output
- Six specialist limitation lists (no scores)

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls, no “parameters” or “name”: words.

YOUR JOB:
1) Compile each specialist’s limitations (copy/adapt 2–4 strongest bullets per dimension).
2) Synthesize an overall limitations summary focused on novelty & significance.
3) Highlight cross-cutting themes and missing evidence.
4) IMPORTANT: Ensure the final report explicitly uses Paper B summaries as comparative evidence (overlap/differentiation).

OUTPUT FORMAT (STRICT):
Prompt 1 (Technical Contributions):
Limitations in Technical Contributions:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Prompt 2 (Experimental Validation):
Limitations in Experimental Validation:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Prompt 3 (Literature Review):
Limitations in Literature Review & Contextualization:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Prompt 4 (Scope/Generalizability):
Limitations in Scope & Generalizability:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Prompt 5 (Claims/Overclaiming):
Limitations in Claims & Overclaiming:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Prompt 6 (Methodological Clarity):
Limitations in Methodological Clarity & Rigor:
- <2-4 bullets, must include A vs B comparison>

Evidence:
- <A pointers>
- <B pointers>

Overall limitations summary:
- Main novelty/significance weaknesses (A vs B): <bullets>
- Cross-cutting issues: <bullets>
- Areas with insufficient evidence: <bullets>
- Paper B usefulness/limitations: <bullets>

End with: TERMINATE
""".strip()

OUTPUT_REQUIREMENT = """
IMPORTANT OUTPUT REQUIREMENT (do not ignore):
- Output MUST be plain text only.
- You MUST compare Paper A against Paper B (the provided Paper B summaries).
- At least 2 limitations MUST explicitly cite overlap / missing differentiation vs Paper B.
- If Paper B is insufficient or irrelevant, say so explicitly under "Paper B usefulness".
- Do NOT provide any numeric scores, ratings, or novelty score.
- Do NOT output JSON. No braces. No tools. No "parameters".

Use this format (STRICT):

Limitations (A vs B):
- <bullet 1: must mention Paper B overlap/difference>
- <bullet 2: must mention Paper B overlap/difference>
- <bullet 3>

Paper B usefulness:
- <relevance/coverage issues>

Evidence (A and B):
- A: <short evidence pointer(s) from Paper A>
- B: <short evidence pointer(s) from Paper B summaries>
""".strip()

def driver_message() -> str:
    return """
Conversation order (STRICT, do not deviate):
1) Planning_Agent
2) Literature_Review_and_Data_Analysis_Agent
3) Hypothesis_Refinement_and_Critical_Reflection_Agent
4) Methodological_Novelty_Agent
5) Experimental_Novelty_Agent
6) Problem_Formulation_Novelty_Agent
7) Writing_Claim_Novelty_Agent
8) Master_Agent (consolidate)

Rules:
- ALL agents MUST output ONLY limitations-focused plain text.
- You MUST compare Paper A vs Paper B summaries (do not ignore Paper B).
- Do NOT output novelty scores or numeric ratings.
""".strip()

# ============================================================
# 2) HF Llama-3-8B config (replace your vLLM config)
# ============================================================

INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/novelty_llm_agents_rag_based/df_with_retrieved_sections.csv"
OUTPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_seq_7_agents/llama_3_8B_inst/novelty_lim/df_llm_novelty_agents_limgen_llama3_8b_seq_100_199.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

ROW_START = 100
ROW_END = 199  # exclusive

TEXT_COL_MAIN = "input_text_cleaned"
RELATED_COL = "relevant_papers_list"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/llama3_8b_instruct"

# 8B context safety (adjust if your build supports more)
MAX_CONTEXT_TOKENS = 8000
MAX_NEW_TOKENS_AGENT = 420      # each agent output
MAX_NEW_TOKENS_MASTER = 650     # master is longer
TEMPERATURE = 0.2

# Truncation budgets inside the *combined input* (Paper A + Paper B)
PAPER_TOK_BUDGET = 5200
CITATION_TOK_BUDGET = 1400

# Retrieved-paper summarization budgets (per retrieved item)
PER_RETR_PAPER_INPUT_TOK = 1700
PER_RETR_PAPER_SUMMARY_TOK = 350

SAVE_EVERY = 10
SLEEP_SEC = 0.15

print(f"[Load] {MODEL_ID} (4-bit) ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3) Helpers (token + parsing)
# ============================================================

def tok_len(text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))

def truncate_to_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def normalize_any_to_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, str):
        s = x.strip()
        return s
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)[:200000]
        except Exception:
            return str(x)
    return str(x)

def parse_relevant_list(x):
    """Return a python list from relevant_papers_list column, else []"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return [s]
    return [str(x)]

def build_dual_paper_input(main_text: str, b_summaries_text: str) -> str:
    main_text = truncate_to_tokens(main_text, PAPER_TOK_BUDGET, keep="head")
    b_summaries_text = truncate_to_tokens(b_summaries_text, CITATION_TOK_BUDGET, keep="head")

    combined = f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text}

=== RETRIEVED RELEVANT PAPERS (Paper B) — CLEAN SUMMARIES ===
{b_summaries_text}
""".strip()

    return combined

# ============================================================
# 4) HF chat runner (single model, sequential calls)
# ============================================================

def run_llama_chat(user_prompt: str, system_prompt: str, max_new_tokens: int) -> str:
    """
    Llama-3-Instruct via apply_chat_template.
    Hard truncates total input tokens so that input + generation fits MAX_CONTEXT_TOKENS.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(text_input, add_special_tokens=False)
    # leave a little headroom for chat template / EOS
    max_input = max(256, MAX_CONTEXT_TOKENS - max_new_tokens - 64)
    if len(input_ids) > max_input:
        # truncate from head (keep early paper content + instructions)
        input_ids = input_ids[:max_input]
        text_input = tokenizer.decode(input_ids, skip_special_tokens=True)

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(TEMPERATURE),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# ============================================================
# 5) Retrieved-paper summarizer (same idea as your Code-1)
# ============================================================

RETR_PAPER_SUMMARY_PROMPT = """
You are summarizing ONE retrieved relevant paper (Paper B) into a structured “review-ready” summary
to support later novelty comparison against a different Paper A.

CRITICAL OUTPUT RULES:
- PLAIN TEXT ONLY. No JSON. No braces. No tools.

Given the Paper B text, produce:
1) Key novelty / technical contribution (1-3 bullets)
2) Experiments & evaluation setup (1-3 bullets)
3) Literature positioning (1-2 bullets)
4) Scope & generalizability (1-2 bullets)
5) Claims / conclusions (1-2 bullets)
6) Method clarity / reproducibility (1-2 bullets)

FORMAT (STRICT):
Paper B Summary:
1) Key novelty:
- ...
2) Experiments:
- ...
3) Literature review / positioning:
- ...
4) Generalizability / scope:
- ...
5) Claims:
- ...
6) Method clarity:
- ...
""".strip()

def summarize_retrieved_paper_item(item_text: str) -> str:
    item_text = normalize_any_to_text(item_text)
    item_text = truncate_to_tokens(item_text, PER_RETR_PAPER_INPUT_TOK, keep="head")
    if len(item_text) < 50:
        return "Paper B Summary:\n(Empty/insufficient retrieved text.)"

    user_prompt = RETR_PAPER_SUMMARY_PROMPT + "\n\n=== Paper B text ===\n" + item_text
    system_prompt = HARSH_REVIEWER_POLICY + "\n\n" + NO_TOOL_NO_JSON
    out = run_llama_chat(user_prompt=user_prompt, system_prompt=system_prompt, max_new_tokens=PER_RETR_PAPER_SUMMARY_TOK)

    # basic cleanup if model accidentally emits braces
    out = out.replace("{", "").replace("}", "").strip()
    return out

def build_b_summaries_from_list(lst, k=3):
    summaries = []
    for idx in range(k):
        if idx < len(lst):
            try:
                s = summarize_retrieved_paper_item(lst[idx])
            except Exception as e:
                s = f"Paper B Summary:\n(ERROR summarizing item {idx+1}: {repr(e)})"
        else:
            s = "Paper B Summary:\n(Missing item.)"
        summaries.append(s)

    combined = "\n\n".join([f"--- Retrieved Paper #{i+1} ---\n{summaries[i]}" for i in range(k)]).strip()
    return combined, summaries

# ============================================================
# 6) Sequential agent runner (KEEP PROMPTS SAME, just call one by one)
# ============================================================

def run_one_agent(agent_name: str, agent_prompt: str, combined_input: str, max_new_tokens: int) -> str:
    """
    We keep your prompt content, and append the same shared constraints + combined input.
    """
    system_prompt = HARSH_REVIEWER_POLICY + "\n\n" + NO_TOOL_NO_JSON

    user_prompt = (
        agent_prompt
        + "\n\n"
        + OUTPUT_REQUIREMENT
        + "\n\n"
        + combined_input
    ).strip()

    return run_llama_chat(user_prompt=user_prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens)

def run_master(planning_out: str, lit_out: str, hyp_out: str, meth_out: str, exp_out: str, prob_out: str, write_out: str, combined_input: str) -> str:
    system_prompt = HARSH_REVIEWER_POLICY + "\n\n" + NO_TOOL_NO_JSON

    # Provide the upstream agent outputs explicitly to the master (so it can actually “receive” them)
    upstream = f"""
=== Planning Agent Output ===
{planning_out}

=== Literature Review and Data Analysis Agent Output ===
{lit_out}

=== Hypothesis Refinement and Critical Reflection Agent Output ===
{hyp_out}

=== Methodological Novelty Agent Output ===
{meth_out}

=== Experimental Novelty Agent Output ===
{exp_out}

=== Problem Formulation Novelty Agent Output ===
{prob_out}

=== Writing Claim Novelty Agent Output ===
{write_out}
""".strip()

    user_prompt = (
        master_agent_prompt
        + "\n\n"
        + upstream
        + "\n\n"
        + combined_input
    ).strip()

    out = run_llama_chat(user_prompt=user_prompt, system_prompt=system_prompt, max_new_tokens=MAX_NEW_TOKENS_MASTER)
    return out

# ============================================================
# 7) Emergency save (optional but useful on HPC)
# ============================================================

global_df = None
global_current_row = -1

def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Emergency saving...")
    if global_df is not None:
        tmp_path = OUTPUT_CSV.replace(".csv", f".emergency_row_{global_current_row}.csv")
        global_df.to_csv(tmp_path, index=False)
        print(f"Saved: {tmp_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================
# 8) Main pipeline
# ============================================================

def run_pipeline():
    global global_df, global_current_row

    df_all = pd.read_csv(INPUT_CSV)
    df = df_all.iloc[ROW_START:ROW_END].copy()

    needed_cols = [
        ("relevant_paper_sum1", "PENDING"),
        ("relevant_paper_sum2", "PENDING"),
        ("relevant_paper_sum3", "PENDING"),
        ("novelty_report", "PENDING"),
        ("planning_output", "PENDING"),
        ("lit_score_reason", "PENDING"),
        ("hyp_score_reason", "PENDING"),
        ("meth_score_reason", "PENDING"),
        ("exp_score_reason", "PENDING"),
        ("prob_score_reason", "PENDING"),
        ("write_score_reason", "PENDING"),
        ("full_chat_history", "PENDING"),
    ]
    for col, default in needed_cols:
        if col not in df.columns:
            df[col] = default

    global_df = df

    for i in tqdm(range(len(df)), desc="Sequential Llama3-8B Limitations (RAG-based)"):
        global_current_row = i
        row = df.iloc[i]

        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, "")) 
        print('main text raw', main_text_raw)  # print first 500 chars for sanity check
        rel_raw = row.get(RELATED_COL, "")
        print('rel raw', rel_raw)

        if len(main_text_raw) < 200:
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        # 1) Summarize top-3 retrieved items into clean Paper B summaries
        rel_list = parse_relevant_list(rel_raw)
        b_combined, b_summaries = build_b_summaries_from_list(rel_list, k=3)

        df.at[df.index[i], "relevant_paper_sum1"] = b_summaries[0]
        df.at[df.index[i], "relevant_paper_sum2"] = b_summaries[1]
        df.at[df.index[i], "relevant_paper_sum3"] = b_summaries[2]

        combined_input = build_dual_paper_input(main_text_raw, b_combined)

        # 2) Run agents sequentially (no autogen, no parallel)
        try:
            planning_out = run_one_agent("Planning_Agent", planning_agent, combined_input, MAX_NEW_TOKENS_AGENT)
            lit_out      = run_one_agent("Literature", PROMPT_3_LIT_REVIEW, combined_input, MAX_NEW_TOKENS_AGENT)
            hyp_out      = run_one_agent("Hypothesis", PROMPT_1_NOVELTY_TECH, combined_input, MAX_NEW_TOKENS_AGENT)
            meth_out     = run_one_agent("Method", PROMPT_6_METHOD_CLARITY_RIGOR, combined_input, MAX_NEW_TOKENS_AGENT)
            exp_out      = run_one_agent("Experiments", PROMPT_2_EXPERIMENTS, combined_input, MAX_NEW_TOKENS_AGENT)
            prob_out     = run_one_agent("Scope", PROMPT_4_SCOPE_GENERALIZABILITY, combined_input, MAX_NEW_TOKENS_AGENT)
            write_out    = run_one_agent("Claims", PROMPT_5_CLAIMS_OVERCLAIMING, combined_input, MAX_NEW_TOKENS_AGENT)

            master_out = run_master(
                planning_out=planning_out,
                lit_out=lit_out,
                hyp_out=hyp_out,
                meth_out=meth_out,
                exp_out=exp_out,
                prob_out=prob_out,
                write_out=write_out,
                combined_input=combined_input,
            )

            # clean terminate token
            master_out = master_out.replace("TERMINATE", "").strip()

            # Store
            df.at[df.index[i], "planning_output"] = planning_out or "NO_OUTPUT"
            df.at[df.index[i], "lit_score_reason"] = lit_out or "NO_OUTPUT"
            df.at[df.index[i], "hyp_score_reason"] = hyp_out or "NO_OUTPUT"
            df.at[df.index[i], "meth_score_reason"] = meth_out or "NO_OUTPUT"
            df.at[df.index[i], "exp_score_reason"] = exp_out or "NO_OUTPUT"
            df.at[df.index[i], "prob_score_reason"] = prob_out or "NO_OUTPUT"
            df.at[df.index[i], "write_score_reason"] = write_out or "NO_OUTPUT"
            df.at[df.index[i], "novelty_report"] = master_out or "NO_OUTPUT_FROM_MASTER"

            # Optional: keep an interpretable “history” (not autogen format)
            df.at[df.index[i], "full_chat_history"] = json.dumps({
                "driver": driver_message(),
                "planning": planning_out,
                "lit": lit_out,
                "hyp": hyp_out,
                "meth": meth_out,
                "exp": exp_out,
                "prob": prob_out,
                "write": write_out,
                "master": master_out,
            }, ensure_ascii=False)[:200000]

        except Exception as e:
            df.at[df.index[i], "novelty_report"] = f"ERROR: {repr(e)}"
            df.at[df.index[i], "full_chat_history"] = "ERROR_NO_HISTORY"

        # checkpoint
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(SLEEP_SEC)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    print("[HF Sequential] Using:", MODEL_ID)
    print("Cache dir:", CACHE_DIR)
    print("Input :", INPUT_CSV)
    print("Output:", OUTPUT_CSV)
    print("Rows  :", ROW_START, ROW_END)
    run_pipeline()