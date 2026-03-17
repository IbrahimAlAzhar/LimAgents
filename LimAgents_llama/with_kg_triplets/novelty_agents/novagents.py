#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import ast
import json
import pandas as pd
from tqdm import tqdm

import autogen
from transformers import AutoTokenizer
import requests


# ============================================================
# 0) Global guardrails (MUST exist; referenced later)
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
# 1) Prompts (kept long; limitations-only; no scoring)
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

CRITICAL OUTPUT RULES (NON-NEGOTIABLE):
- Output MUST be PLAIN TEXT ONLY. No JSON, no braces {}, no tool calls.

Output format (STRICT):
Limitations in Scope & Generalizability (A vs B):
- <bullet 1; MUST compare against Paper B summaries>
- <bullet 2; MUST compare against Paper B summaries>
- <bullet 3>

Evidence (A and B):
- A: <short quote/pointer from Paper A>
- B: <short pointer from Paper B summary with broader evaluation/scope>
""".strip()

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


# ============================================================
# 2) Master Agent prompt (strict + final consolidated report)
# ============================================================

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


# ============================================================
# 3) vLLM readiness check
# ============================================================

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

def get_vllm_cli_args(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    candidates = [
        base_url.replace("/v1", "") + "/version",
        base_url + "/models",
        base_url.replace("/v1", "") + "/openapi.json",
    ]
    out = {"base_url": base_url, "queried": [], "responses": []}
    for url in candidates:
        try:
            r = requests.get(url, timeout=5)
            out["queried"].append(url)
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" in ct:
                out["responses"].append({"url": url, "status": r.status_code, "json": r.json()})
            else:
                out["responses"].append({"url": url, "status": r.status_code, "text_snippet": r.text[:500]})
        except Exception as e:
            out["responses"].append({"url": url, "error": str(e)})
    return out


# ============================================================
# 4) Config (UPDATED token budgets)
# ============================================================


TEXT_COL_MAIN = "input_text_cleaned" # "input_text_for_novelty"
RELATED_COL = "relevant_papers_list"

MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# --- Updated settings you gave ---
VLLM_MAX_MODEL_LEN = 100000  # matches your server --max-model-len
MAX_CTX = max(92_000, VLLM_MAX_MODEL_LEN - 1_000)

RESERVED_FOR_CONVO = min(30_000, MAX_CTX // 2)  # chat history reserve
RESERVED_FOR_GEN = min(2_000, MAX_CTX // 20)    # generation reserve

AVAILABLE_FOR_INPUT = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
AVAILABLE_FOR_INPUT = max(4_000, AVAILABLE_FOR_INPUT)

PAPER_TOK_BUDGET = min(50_000, max(2_000, int(AVAILABLE_FOR_INPUT * 0.82)))
CITATION_TOK_BUDGET = min(6_000, max(1_000, AVAILABLE_FOR_INPUT - PAPER_TOK_BUDGET))

MAX_TOKENS_PER_REPLY = 400
MAX_ROUND = 10
TEMPERATURE = 0.2
TIMEOUT = 600

# Summarization budgets (for each retrieved paper item)
# Keep these small-ish because you're doing up to 3 per row.
PER_RETR_PAPER_INPUT_TOK = 2500
PER_RETR_PAPER_SUMMARY_TOK = 700

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)


# ============================================================
# 5) LLM config for AutoGen  (tool calls disabled)
# ============================================================

llm_config = {
    "config_list": [{
        "api_type": "openai",
        "model": VLLM_MODEL,
        "api_key": "EMPTY",
        "base_url": VLLM_BASE_URL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_PER_REPLY,
        "extra_body": {
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
    }],
    "timeout": TIMEOUT,
    "cache_seed": None,
    "extra_body": {
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    },
}

def is_terminate_msg(msg) -> bool:
    c = (msg.get("content") or "").strip()
    return c == "TERMINATE" or c.endswith("\nTERMINATE") or c.endswith("TERMINATE")


# ============================================================
# 6) Helpers
# ============================================================

def tok_len(text: str) -> int:
    return len(tokenizer.encode(text or ""))

def truncate_to_tokens(text: str, max_tokens: int, keep: str = "head") -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
    return tokenizer.decode(ids)

def normalize_any_to_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""
        # don't JSON-dump huge nested stuff; keep as-is if it's long
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
            # if it's just a long string blob, treat as single item
            return [s]
    return [str(x)]

def build_dual_paper_input(main_text: str, b_summaries_text: str) -> str:
    # Truncate Paper A and Paper B summaries using new budgets
    main_text = truncate_to_tokens(main_text, PAPER_TOK_BUDGET, keep="head")
    b_summaries_text = truncate_to_tokens(b_summaries_text, CITATION_TOK_BUDGET, keep="head")

    combined = f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text}

=== RETRIEVED RELEVANT PAPERS (Paper B) — CLEAN SUMMARIES ===
{b_summaries_text}
""".strip()

    max_input_budget = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
    if tok_len(combined) > max_input_budget:
        # emergency shrink Paper A a bit
        main_budget = max(10_000, PAPER_TOK_BUDGET - 10_000)
        main_text2 = truncate_to_tokens(main_text, main_budget, keep="head")
        combined = f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text2}

=== RETRIEVED RELEVANT PAPERS (Paper B) — CLEAN SUMMARIES ===
{b_summaries_text}
""".strip()

    return combined


# ============================================================
# 7) Output-format wrapper (limitations-only)
# ============================================================

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


# ============================================================
# 8) Driver message (NO scores)
# ============================================================

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
# 9) NEW: summarize each retrieved paper item into a clean 6-dim summary
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

def vllm_chat_completion(content: str, max_tokens: int, temperature: float = 0.2, timeout: int = 60) -> str:
    """Direct REST call to vLLM /chat/completions"""
    url = VLLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "tool_choice": "none",
        "tools": [],
    }
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"vLLM chat/completions failed: {r.status_code} | {r.text[:300]}")
    j = r.json()
    return (j["choices"][0]["message"]["content"] or "").strip()

def summarize_retrieved_paper_item(item_text: str) -> str:
    item_text = normalize_any_to_text(item_text)
    item_text = truncate_to_tokens(item_text, PER_RETR_PAPER_INPUT_TOK, keep="head")
    if len(item_text) < 50:
        return "Paper B Summary:\n(Empty/insufficient retrieved text.)"
    prompt = RETR_PAPER_SUMMARY_PROMPT + "\n\n=== Paper B text ===\n" + item_text
    out = vllm_chat_completion(prompt, max_tokens=PER_RETR_PAPER_SUMMARY_TOK, temperature=TEMPERATURE, timeout=TIMEOUT)
    # enforce plain text (basic cleanup)
    out = out.replace("{", "").replace("}", "").strip()
    return out


def build_b_summaries_from_list(lst, k=3) -> tuple[str, list[str]]:
    """Return combined text + list of per-item summaries length k."""
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

    combined = "\n\n".join(
        [f"--- Retrieved Paper #{i+1} ---\n{summaries[i]}" for i in range(k)]
    ).strip()
    return combined, summaries


# ============================================================
# 10) Main pipeline (now: create B summaries cols first, then run agents)
# ============================================================

def run_pipeline():
    df_all = pd.read_csv(INPUT_CSV)
    df = df_all.iloc[ROW_START:ROW_END].copy()

    # Ensure required columns exist
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

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=999,
        code_execution_config=False,
        default_auto_reply="Continue."
    )

    for i in tqdm(range(len(df)), desc="AutoGen Limitations (RAG-based)"):
        row = df.iloc[i]

        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, ""))
        rel_raw = row.get(RELATED_COL, "")

        if len(main_text_raw) < 200:
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        # --- NEW: parse and summarize top-3 retrieved items ---
        rel_list = parse_relevant_list(rel_raw)
        b_combined, b_summaries = build_b_summaries_from_list(rel_list, k=3)

        df.at[df.index[i], "relevant_paper_sum1"] = b_summaries[0]
        df.at[df.index[i], "relevant_paper_sum2"] = b_summaries[1]
        df.at[df.index[i], "relevant_paper_sum3"] = b_summaries[2]

        # Use combined Paper B summaries as "related"
        combined_input = build_dual_paper_input(main_text_raw, b_combined)
        combined_tokens = tok_len(combined_input)

        # Agents
        planning = autogen.AssistantAgent(
            name="Planning_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + planning_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        lit = autogen.AssistantAgent(
            name="Literature_Review_and_Data_Analysis_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + literature_review_and_data_analysis_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        hyp = autogen.AssistantAgent(
            name="Hypothesis_Refinement_and_Critical_Reflection_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + hypothesis_refinement_and_critical_reflection_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        meth = autogen.AssistantAgent(
            name="Methodological_Novelty_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + methodological_novelty_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        exp = autogen.AssistantAgent(
            name="Experimental_Novelty_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + experimental_novelty_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        prob = autogen.AssistantAgent(
            name="Problem_Formulation_Novelty_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + problem_Formulation_novelty_Agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        write = autogen.AssistantAgent(
            name="Writing_Claim_Novelty_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + writing_claim_novelty_agent
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + OUTPUT_REQUIREMENT + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        master = autogen.AssistantAgent(
            name="Master_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + master_agent_prompt
                           + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + combined_input,
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        agents_list = [user_proxy, planning, lit, hyp, meth, exp, prob, write, master]

        # You asked: with round_robin, max_round ~ number agents so each speaks once
        max_round = min(MAX_ROUND, len(agents_list))

        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=max_round,
            speaker_selection_method="round_robin",
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        task_msg = driver_message()

        try:
            print(f"\n[Row {ROW_START + i}] tokens={combined_tokens} | max_round={max_round} | reply_max_tokens={MAX_TOKENS_PER_REPLY}")
            chat_result = user_proxy.initiate_chat(manager, message=task_msg, clear_history=True)

            def last_msg(name: str) -> str:
                for m in reversed(chat_result.chat_history):
                    if m.get("name") == name and m.get("content"):
                        return (m.get("content") or "").strip()
                return ""

            plan_out = last_msg("Planning_Agent")
            lit_out = last_msg("Literature_Review_and_Data_Analysis_Agent")
            hyp_out = last_msg("Hypothesis_Refinement_and_Critical_Reflection_Agent")
            meth_out = last_msg("Methodological_Novelty_Agent")
            exp_out = last_msg("Experimental_Novelty_Agent")
            prob_out = last_msg("Problem_Formulation_Novelty_Agent")
            write_out = last_msg("Writing_Claim_Novelty_Agent")
            master_out = (last_msg("Master_Agent") or "").strip()

            # --- HARD FILTER: require Paper B usage + headings ---
            bad_markers = ["{", "}", '"parameters"', '"tool"', '"function_call"', 'tool_calls', 'function_call']
            required_headers = [
                "Prompt 1 (Technical Contributions):",
                "Prompt 2 (Experimental Validation):",
                "Prompt 3 (Literature Review):",
                "Prompt 4 (Scope/Generalizability):",
                "Prompt 5 (Claims/Overclaiming):",
                "Prompt 6 (Methodological Clarity):",
                "Overall limitations summary:",
            ]
            # Require explicit Paper B evidence presence
            needs_b = master_out.lower().count("paper b") < 3 and master_out.lower().count("b:") < 3

            invalid = any(m in master_out for m in bad_markers) \
                      or any(h not in master_out for h in required_headers) \
                      or needs_b

            if invalid:
                rewrite_msg = (
                    "REWRITE ONLY. Your previous output was INVALID (missing required headings and/or not using Paper B). "
                    "Now output ONLY plain text using the EXACT required format in the Master Agent prompt. "
                    "You MUST include Evidence lines for both A and B in every section. "
                    "Do NOT include braces, JSON, tools, or any numeric scores. "
                    "End with TERMINATE."
                )
                gc2 = autogen.GroupChat(
                    agents=[user_proxy, master],
                    messages=[],
                    max_round=2,
                    speaker_selection_method="round_robin",
                )
                mgr2 = autogen.GroupChatManager(groupchat=gc2, llm_config=llm_config)
                chat2 = user_proxy.initiate_chat(mgr2, message=rewrite_msg, clear_history=True)

                for m in reversed(chat2.chat_history):
                    if m.get("name") == "Master_Agent" and m.get("content"):
                        master_out = (m.get("content") or "").strip()
                        break

            master_out = master_out.replace("TERMINATE", "").strip()

            df.at[df.index[i], "planning_output"] = plan_out or "NO_OUTPUT"
            df.at[df.index[i], "lit_score_reason"] = lit_out or "NO_OUTPUT"
            df.at[df.index[i], "hyp_score_reason"] = hyp_out or "NO_OUTPUT"
            df.at[df.index[i], "meth_score_reason"] = meth_out or "NO_OUTPUT"
            df.at[df.index[i], "exp_score_reason"] = exp_out or "NO_OUTPUT"
            df.at[df.index[i], "prob_score_reason"] = prob_out or "NO_OUTPUT"
            df.at[df.index[i], "write_score_reason"] = write_out or "NO_OUTPUT"
            df.at[df.index[i], "novelty_report"] = master_out or "NO_OUTPUT_FROM_MASTER"
            df.at[df.index[i], "full_chat_history"] = str(chat_result.chat_history)

        except Exception as e:
            df.at[df.index[i], "novelty_report"] = f"ERROR: {repr(e)}"
            df.at[df.index[i], "full_chat_history"] = "ERROR_NO_CHAT_HISTORY"

        # checkpoint every 10 processed rows (not at i=0)
        if (i + 1) % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    print("[CLI args]", sys.argv)
    print(f"Using vLLM base_url: {VLLM_BASE_URL}")
    print(f"Using model name  : {VLLM_MODEL}")

    print("\n[Env snapshot]")
    for k in ["VLLM_BASE_URL", "VLLM_MODEL", "RAY_ADDRESS", "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG"]:
        print(f"{k}={os.environ.get(k)}")

    print("\n[Best-effort vLLM server info]")
    info = get_vllm_cli_args(VLLM_BASE_URL)
    try:
        print(json.dumps(info, indent=2)[:5000])
    except Exception:
        print(str(info)[:5000])

    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=TIMEOUT):
        raise RuntimeError(f"vLLM not ready at {VLLM_BASE_URL}")

    # One-shot sanity test
    print("\n[Sanity test] /chat/completions ...")
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "temperature": 0,
        "tool_choice": "none",
        "tools": [],
    }
    r = requests.post(VLLM_BASE_URL + "/chat/completions", json=payload, timeout=30)
    print("Sanity status:", r.status_code)
    print("Sanity body  :", r.text[:300])
    if r.status_code != 200:
        raise RuntimeError("Sanity test failed; vLLM endpoint is rejecting requests. See output above.")

    run_pipeline()
