import os
import sys
import time
import ast
import json
import re
import signal
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 0) INPUT / OUTPUT / MODEL CONFIG
# ============================================================


# Required columns
NOVELTY_INPUT_COL = "input_text_for_novelty"
RELEVANT_LIST_COL = "relevant_papers_list"
RELEVANT_SUM_COL = "relevant_papers_sum"

# Mistral model
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/mistral_7b_v3_instruct"

# Generation config
TEMPERATURE = 0.3
DO_SAMPLE = True

MAX_CONTEXT_TOKENS = 32000
MAX_NEW_TOKENS = 900
MAX_PROMPT_TOKENS = MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS - 512

MAIN_PAPER_TOKEN_BUDGET = 24000
RELATED_SUM_TOKEN_BUDGET = 5000

SAVE_EVERY = 5
SLEEP_SEC = 1

# ============================================================
# 1) LOAD MISTRAL
# ============================================================

print("Loading Mistral model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2) GLOBALS FOR SAFE EMERGENCY SAVE
# ============================================================

global_df = None
global_current_row = -1

def signal_handler(signum, frame):
    print(f"\n⚠️ Received signal {signum}. Saving progress...")
    if global_df is not None:
        save_path = os.path.join(OUTPUT_DIR, f"emergency_save_row_{global_current_row}.csv")
        global_df.to_csv(save_path, index=False)
        print(f"Saved emergency progress to: {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================
# 3) TEXT / TOKEN HELPERS
# ============================================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "... [TRUNCATED]"

def _messages_token_len(messages: List[Dict[str, str]]) -> int:
    tmp = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return int(tmp.shape[-1])

def truncate_user_prompt_for_context(system_prompt: Optional[str], user_prompt: str) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})

    cur_len = _messages_token_len(msgs)
    if cur_len <= MAX_PROMPT_TOKENS:
        return user_prompt

    user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    low = 512
    high = len(user_ids)
    best = low

    while low <= high:
        mid = (low + high) // 2
        trial_prompt = tokenizer.decode(user_ids[:mid], skip_special_tokens=True)

        trial_msgs = []
        if system_prompt:
            trial_msgs.append({"role": "system", "content": system_prompt})
        trial_msgs.append({"role": "user", "content": trial_prompt})

        if _messages_token_len(trial_msgs) <= MAX_PROMPT_TOKENS:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    truncated = tokenizer.decode(user_ids[:best], skip_special_tokens=True)
    return truncated + "\n\n... [TRUNCATED TO FIT CONTEXT]"

def normalize_any_to_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, dict)):
                return json.dumps(parsed, ensure_ascii=False, indent=2)[:200000]
            return s
        except Exception:
            return s
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)[:200000]
        except Exception:
            return str(x)
    return str(x)

# ============================================================
# 4) RELEVANT PAPERS PARSING + SUMMARIZATION
# ============================================================

def parse_relevant_papers_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i) for i in parsed if str(i).strip()]
            return [str(parsed)]
        except Exception:
            return [s]

    return [str(x)]

def summarization_prompt_for_relevant_paper(item_text: str, idx: int) -> str:
    return f"""You will summarize one relevant paper for novelty comparison. Produce a compact, structured summary focusing ONLY on the following dimensions:

1) Literature analysis
2) Data analysis
3) Hypothesis refinement and critical reflection
4) Methodological novelty
5) Experimental novelty
6) Problem formulation novelty
7) Writing/claim novelty

Rules:
- Use plain text only.
- Keep it concise and information-dense.
- If a dimension is not stated in the text, write "Not stated".
- Do not invent details.

Relevant paper #{idx} text:
{item_text}

Output format (STRICT):
Relevant Paper #{idx} Summary:
- Literature analysis: ...
- Data analysis: ...
- Hypothesis refinement and critical reflection: ...
- Methodological novelty: ...
- Experimental novelty: ...
- Problem formulation novelty: ...
- Writing/claim novelty: ...
""".strip()

def summarize_relevant_papers_list(relevant_list: List[str], max_items: int = 3) -> str:
    if not relevant_list:
        return ""

    take = relevant_list[:max_items]
    summaries = []

    for j, item in enumerate(take, start=1):
        item_clean = clean_text_detailed(item)
        item_tr = truncate_to_tokens(item_clean, max_tokens=6000)

        prompt = summarization_prompt_for_relevant_paper(item_tr, j)
        out = mistral_generate(
            user_prompt=prompt,
            system_prompt="You are an expert scientific summarizer.",
            max_new_tokens=650
        )
        summaries.append(out.strip())

    return "\n\n".join(summaries).strip()

# ============================================================
# 5) GENERATION HELPER
# ============================================================

def mistral_generate(user_prompt: str, system_prompt: Optional[str] = None, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    user_prompt = truncate_user_prompt_for_context(system_prompt, user_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return text.strip()

# ============================================================
# 6) STRICT PROMPTS FROM CODE 2
# ============================================================

planning_agent = '''
You are the Planning Agent in a multi-agent system designed to measure the novelty of a main research paper 
compared to a specific related paper.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- DO NOT output JSON.
- DO NOT output anything like {"name": ..., "parameters": ...} or "parameters:" blocks.
- DO NOT call tools or pretend to call tools.

Task:
- Synthesize the core contributions, problem statements, methods, and claims of both papers.
- Prioritize novelty assessment criteria based on the papers' domain.
- Assign prioritized subtasks to other agents and forward relevant excerpts.

OUTPUT FORMAT (STRICT):
Score: <float 0..1>   (planning confidence that Paper A is novel overall, based ONLY on given excerpts; harsh default low)
Reason:
- <2-5 bullets explaining what seems overlapping vs potentially distinct>
Evidence:
- <short pointers from Paper A / Paper B excerpts>
Prioritized subtasks:
1) <task + rationale>
2) <task + rationale>
...
'''.strip()

NO_TOOL_NO_JSON = """
CRITICAL (HARD BLOCKLIST):
- Output MUST be plain text only.
- Do NOT output JSON (NO braces: do not output '{' or '}').
- Do NOT output any schema such as {"name": "...", "parameters": {...}}.
- Do NOT use the words "parameters", "function_call", "tool", "tools", or 'name":'.
- Do NOT call tools / functions (even hypothetically).

If you accidentally start to output JSON or any braces, STOP and rewrite the entire answer in the required plain-text format.
""".strip()

HARSH_REVIEWER_POLICY = """
HARSH REVIEWER MODE (STRICT):
You are an extremely skeptical reviewer. Assume low novelty unless proven otherwise with direct evidence.

Scoring rule (score in [0,1]):
- Start from 0.0 by default.
- Increase score only if you can point to CLEAR, SPECIFIC differences between Paper A and Paper B in your criterion.
- If there is ANY meaningful overlap in your criterion (same task framing, same method family, same experiment setup, same claims, same dataset/analysis idea), your score MUST be <= 0.30.
- If the Paper A contribution is a known variant / incremental tweak / combination of known components already present in Paper B, score MUST be <= 0.20.
- Score > 0.50 is allowed ONLY if Paper A introduces a new capability or formulation that Paper B does not cover AND you cite explicit evidence from the excerpts.
- If evidence is missing or ambiguous, score MUST be <= 0.15 and you must say "insufficient evidence".

Hard constraints:
- Do NOT reward novelty because Paper A mentions something Paper B does not; absence of mention is NOT novelty unless the excerpt proves a real new contribution.
- Penalize vague novelty claims ("novel", "first", "unique") unless backed by concrete described differences.
- Treat re-framing/rewording as NOT novel.
- Prefer false negatives over false positives.

Output must remain plain text and follow the required output format.
""".strip()

PROMPT_1_NOVELTY_TECH = """
Prompt 1: Novelty and Technical Contributions
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on assessing 
whether the paper introduces significant new concepts or insights, or if it primarily offers incremental 
improvements, weak technical contributions, or simplistic adaptations of existing methods without substantial 
advancements. Consider if the proposed methods closely resemble prior works or rely heavily on established 
techniques.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

PROMPT_2_EXPERIMENTS = """
Prompt 2: Experimental Validation and Comparative Analysis
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on whether the paper 
provides comprehensive experimental validation, including adequate comparisons with benchmarks and state-of-the-art 
techniques, or if it lacks sufficient benchmarking, fails to demonstrate significant performance improvements, 
and raises concerns about reliability and applicability.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

PROMPT_3_LIT_REVIEW = """
Prompt 3: Literature Review and Contextualization
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on the thoroughness of 
the literature review and how well the paper contextualizes its contributions within existing research, or if 
it overlooks prior studies, fails to clarify differences or build upon them, leading to ambiguity about originality 
and significance.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

PROMPT_4_SCOPE_GENERALIZABILITY = """
Prompt 4: Scope of Analysis and Generalizability
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on the breadth of the 
analysis, including whether the paper explores diverse datasets and broader implications, or if it is limited to 
narrow tasks, restricting generalizability and the applicability of findings in varied contexts.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

PROMPT_5_CLAIMS_OVERCLAIMING = """
Prompt 5: Claim Accuracy and Overclaiming
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on whether the paper's 
claims about its contributions are substantiated by results, or if it exaggerates novelty, effectiveness, or 
impact without adequate evidence, leading to skepticism about the true relevance and advancements.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

PROMPT_6_METHOD_CLARITY_RIGOR = """
Prompt 6: Methodological Clarity and Rigor
Evaluate the novelty of the input paper compared to the provided relevant papers. Focus on the clarity and rigor 
of the methodology, including whether the paper articulates experimental setups in detail for reproducibility and 
validity assessment, or if it lacks sufficient detail, undermining the perceived quality and reliability of the 
contributions.

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT call tools or pretend to call tools.
- If you start outputting JSON/braces, STOP and rewrite.

Output format (STRICT):
Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short pointers from the input paper and relevant papers>
""".strip()

master_agent_prompt = """
You are the Master Agent. You will receive:
- Planning Agent output
- Six specialist novelty assessments (each with Novelty Score/Reasons/Evidence)

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY.
- ABSOLUTELY FORBIDDEN: JSON, braces '{' '}', or anything like {"name": ..., "parameters": ...}.
- Do NOT include the words "parameters" or 'name":' anywhere.
- Do NOT call tools or pretend to call tools.
- If any specialist output looks like JSON/tool-call, IGNORE it and proceed using remaining evidence.
- If you are about to output braces/JSON, STOP and rewrite.

YOUR JOB:
1) Produce a consolidated novelty report.
2) Include each dimension's Novelty Score and a short rationale.
3) Compute Overall novelty score in [0,1] (simple average of the six scores; if any missing, average available scores).
4) Provide a final summary.

OUTPUT FORMAT (STRICT):
Prompt 1 (Technical Contributions):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Prompt 2 (Experimental Validation):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Prompt 3 (Literature Review):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Prompt 4 (Scope/Generalizability):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Prompt 5 (Claims/Overclaiming):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Prompt 6 (Methodological Clarity):
Novelty Score: <float 0..1>
Reasons:
- <2-4 bullets>
Evidence:
- <pointers>

Overall novelty score: <float 0..1>
Final summary:
- Main novelty drivers: <bullets>
- Main overlaps / not-novel aspects: <bullets>
- Missing evidence: <bullets>
End with: TERMINATE
""".strip()

OUTPUT_REQUIREMENT = """
IMPORTANT OUTPUT REQUIREMENT (do not ignore):
- Provide a novelty score in [0, 1], where 0 means not novel and 1 means highly novel.
- Provide detailed reasons referencing specific aspects from the input paper and relevant papers.
- Output MUST be plain text only. No JSON. No braces. No tools. No "parameters".

If you accidentally output anything containing '{' or '}', or a structure like {"name":..., "parameters":...},
you MUST immediately rewrite the output as plain text in the format below.

Use this format:

Novelty Score: <float 0..1>
Reasons:
- <bullet 1>
- <bullet 2>
- <bullet 3>
Evidence:
- <short evidence pointer(s) from Paper A / Paper B excerpts>
""".strip()

# ============================================================
# 7) INPUT BUILDING
# ============================================================

def build_dual_paper_input(main_text: str, related_sum_text: str) -> str:
    main_text = truncate_to_tokens(main_text, MAIN_PAPER_TOKEN_BUDGET)
    related_sum_text = truncate_to_tokens(related_sum_text, RELATED_SUM_TOKEN_BUDGET)

    return f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text}

=== RETRIEVED RELEVANT PAPERS / CONTEXT (Paper B) ===
{related_sum_text}
""".strip()

# ============================================================
# 8) AGENT RUNNERS (SEQUENTIAL, NO AUTOGEN)
# ============================================================

def build_agent_input(base_prompt: str, combined_input: str) -> str:
    return (
        HARSH_REVIEWER_POLICY
        + "\n\n"
        + base_prompt
        + "\n\n"
        + NO_TOOL_NO_JSON
        + "\n\n"
        + OUTPUT_REQUIREMENT
        + "\n\n"
        + combined_input
    )

def build_planning_input(combined_input: str) -> str:
    return (
        HARSH_REVIEWER_POLICY
        + "\n\n"
        + planning_agent
        + "\n\n"
        + NO_TOOL_NO_JSON
        + "\n\n"
        + combined_input
    )

def build_master_input(combined_input: str, outputs: Dict[str, str]) -> str:
    specialist_block = f"""
Planning Agent Output:
{outputs.get("planning_output", "")}

Prompt 3 - Literature Review and Data Analysis:
{outputs.get("lit_score_reason", "")}

Prompt 1 - Hypothesis Refinement and Critical Reflection:
{outputs.get("hyp_score_reason", "")}

Prompt 6 - Methodological Novelty:
{outputs.get("meth_score_reason", "")}

Prompt 2 - Experimental Novelty:
{outputs.get("exp_score_reason", "")}

Prompt 4 - Problem Formulation Novelty:
{outputs.get("prob_score_reason", "")}

Prompt 5 - Writing / Claim Novelty:
{outputs.get("write_score_reason", "")}
""".strip()

    return (
        HARSH_REVIEWER_POLICY
        + "\n\n"
        + master_agent_prompt
        + "\n\n"
        + NO_TOOL_NO_JSON
        + "\n\n"
        + combined_input
        + "\n\n"
        + "SPECIALIST OUTPUTS:\n"
        + specialist_block
    )

def sanitize_plaintext_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text.replace("TERMINATE", "").strip()

def force_rewrite_if_invalid(raw_output: str, original_prompt: str, system_prompt: str, must_contain: Optional[List[str]] = None) -> str:
    out = raw_output or ""
    bad_markers = ["{", "}", '"parameters"', "parameters", '"name"', 'name":', '"tool"', '"tools"']
    invalid = any(m in out for m in bad_markers)

    if must_contain:
        for m in must_contain:
            if m not in out:
                invalid = True
                break

    if not invalid:
        return sanitize_plaintext_output(out)

    rewrite_prompt = f"""Your previous answer was invalid because it included forbidden formatting or missed required headings.

Rewrite the answer now.

STRICT RULES:
- Plain text only
- No JSON
- No braces
- No tools
- Do not use the words parameters or name":
- Follow the required format exactly

Original task:
{original_prompt}
"""
    repaired = mistral_generate(
        user_prompt=rewrite_prompt,
        system_prompt=system_prompt,
        max_new_tokens=MAX_NEW_TOKENS
    )
    return sanitize_plaintext_output(repaired)

# ============================================================
# 9) MAIN PIPELINE
# ============================================================

def run_pipeline():
    global global_df, global_current_row

    print("Loading CSV file...")
    df_all = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_all)} rows total.")

    df = df_all.iloc[START_ROW:END_ROW].copy()
    df = df.reset_index(drop=False).rename(columns={"index": "orig_index"})
    global_df = df

    required_out_cols = [
        "novelty_report",
        "planning_output",
        "lit_score_reason",
        "hyp_score_reason",
        "meth_score_reason",
        "exp_score_reason",
        "prob_score_reason",
        "write_score_reason",
        "full_chat_history",
        RELEVANT_SUM_COL,
    ]
    for c in required_out_cols:
        if c not in df.columns:
            df[c] = ""

    print(f"Starting sequential novelty pipeline for rows {START_ROW}:{END_ROW} ({len(df)} rows)...")

    for r in tqdm(range(len(df))):
        global_current_row = r
        row = df.iloc[r]

        main_text_raw = normalize_any_to_text(row.get(NOVELTY_INPUT_COL, ""))
        relevant_raw = row.get(RELEVANT_LIST_COL, "")

        if len(main_text_raw.strip()) < 200:
            df.at[r, "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        try:
            rel_list = parse_relevant_papers_list(relevant_raw)
            rel_sum = summarize_relevant_papers_list(rel_list, max_items=3)
            df.at[r, RELEVANT_SUM_COL] = rel_sum

            if len(rel_sum.strip()) < 50:
                rel_sum = "INSUFFICIENT RETRIEVED CONTEXT PROVIDED."

            main_text_tr = truncate_to_tokens(main_text_raw, MAIN_PAPER_TOKEN_BUDGET)
            rel_sum_tr = truncate_to_tokens(rel_sum, RELATED_SUM_TOKEN_BUDGET)
            combined_input = build_dual_paper_input(main_text_tr, rel_sum_tr)

            outputs = {}
            chat_history_log = []

            # 1) Planning agent
            planning_input = build_planning_input(combined_input)
            planning_raw = mistral_generate(
                user_prompt=planning_input,
                system_prompt="You are the Planning Agent."
            )
            planning_out = force_rewrite_if_invalid(
                raw_output=planning_raw,
                original_prompt=planning_input,
                system_prompt="You are the Planning Agent.",
                must_contain=["Score:", "Reason:", "Evidence:"]
            )
            outputs["planning_output"] = planning_out
            df.at[r, "planning_output"] = planning_out
            chat_history_log.append({"agent": "Planning_Agent", "output": planning_out})

            # 2) Literature Review and Data Analysis
            lit_input = build_agent_input(PROMPT_3_LIT_REVIEW, combined_input)
            lit_raw = mistral_generate(
                user_prompt=lit_input,
                system_prompt="You are the Literature_Review_and_Data_Analysis_Agent."
            )
            lit_out = force_rewrite_if_invalid(
                raw_output=lit_raw,
                original_prompt=lit_input,
                system_prompt="You are the Literature_Review_and_Data_Analysis_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["lit_score_reason"] = lit_out
            df.at[r, "lit_score_reason"] = lit_out
            chat_history_log.append({"agent": "Literature_Review_and_Data_Analysis_Agent", "output": lit_out})

            # 3) Hypothesis Refinement and Critical Reflection
            hyp_input = build_agent_input(PROMPT_1_NOVELTY_TECH, combined_input)
            hyp_raw = mistral_generate(
                user_prompt=hyp_input,
                system_prompt="You are the Hypothesis_Refinement_and_Critical_Reflection_Agent."
            )
            hyp_out = force_rewrite_if_invalid(
                raw_output=hyp_raw,
                original_prompt=hyp_input,
                system_prompt="You are the Hypothesis_Refinement_and_Critical_Reflection_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["hyp_score_reason"] = hyp_out
            df.at[r, "hyp_score_reason"] = hyp_out
            chat_history_log.append({"agent": "Hypothesis_Refinement_and_Critical_Reflection_Agent", "output": hyp_out})

            # 4) Methodological Novelty
            meth_input = build_agent_input(PROMPT_6_METHOD_CLARITY_RIGOR, combined_input)
            meth_raw = mistral_generate(
                user_prompt=meth_input,
                system_prompt="You are the Methodological_Novelty_Agent."
            )
            meth_out = force_rewrite_if_invalid(
                raw_output=meth_raw,
                original_prompt=meth_input,
                system_prompt="You are the Methodological_Novelty_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["meth_score_reason"] = meth_out
            df.at[r, "meth_score_reason"] = meth_out
            chat_history_log.append({"agent": "Methodological_Novelty_Agent", "output": meth_out})

            # 5) Experimental Novelty
            exp_input = build_agent_input(PROMPT_2_EXPERIMENTS, combined_input)
            exp_raw = mistral_generate(
                user_prompt=exp_input,
                system_prompt="You are the Experimental_Novelty_Agent."
            )
            exp_out = force_rewrite_if_invalid(
                raw_output=exp_raw,
                original_prompt=exp_input,
                system_prompt="You are the Experimental_Novelty_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["exp_score_reason"] = exp_out
            df.at[r, "exp_score_reason"] = exp_out
            chat_history_log.append({"agent": "Experimental_Novelty_Agent", "output": exp_out})

            # 6) Problem Formulation Novelty
            prob_input = build_agent_input(PROMPT_4_SCOPE_GENERALIZABILITY, combined_input)
            prob_raw = mistral_generate(
                user_prompt=prob_input,
                system_prompt="You are the Problem_Formulation_Novelty_Agent."
            )
            prob_out = force_rewrite_if_invalid(
                raw_output=prob_raw,
                original_prompt=prob_input,
                system_prompt="You are the Problem_Formulation_Novelty_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["prob_score_reason"] = prob_out
            df.at[r, "prob_score_reason"] = prob_out
            chat_history_log.append({"agent": "Problem_Formulation_Novelty_Agent", "output": prob_out})

            # 7) Writing Claim Novelty
            write_input = build_agent_input(PROMPT_5_CLAIMS_OVERCLAIMING, combined_input)
            write_raw = mistral_generate(
                user_prompt=write_input,
                system_prompt="You are the Writing_Claim_Novelty_Agent."
            )
            write_out = force_rewrite_if_invalid(
                raw_output=write_raw,
                original_prompt=write_input,
                system_prompt="You are the Writing_Claim_Novelty_Agent.",
                must_contain=["Novelty Score:", "Reasons:", "Evidence:"]
            )
            outputs["write_score_reason"] = write_out
            df.at[r, "write_score_reason"] = write_out
            chat_history_log.append({"agent": "Writing_Claim_Novelty_Agent", "output": write_out})

            # 8) Master Agent
            master_input = build_master_input(combined_input, outputs)
            master_raw = mistral_generate(
                user_prompt=master_input,
                system_prompt="You are the Master Agent."
            )
            master_out = force_rewrite_if_invalid(
                raw_output=master_raw,
                original_prompt=master_input,
                system_prompt="You are the Master Agent.",
                must_contain=["Overall novelty score:", "Final summary:"]
            )

            df.at[r, "novelty_report"] = master_out
            chat_history_log.append({"agent": "Master_Agent", "output": master_out})

            df.at[r, "full_chat_history"] = json.dumps(chat_history_log, ensure_ascii=False)

        except Exception as e:
            err = f"ERROR: {repr(e)}"
            df.at[r, "novelty_report"] = err
            df.at[r, "full_chat_history"] = err
            print(f"Error on row r={r}, orig_index={row.get('orig_index')}: {e}")

        if r % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_FILE, index=False)

        time.sleep(SLEEP_SEC)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved to {OUTPUT_FILE}")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    run_pipeline()