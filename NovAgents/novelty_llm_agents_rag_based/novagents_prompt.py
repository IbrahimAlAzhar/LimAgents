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
# 0) Planning + Guards
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

# Stronger: forbid braces + forbid the usual toolcall tokens
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


# ============================================================
# 1) Your exact 6 prompts (with minimal safety additions)
# ============================================================

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


# Map prompts to your existing agent names (keep rest of pipeline unchanged)
literature_review_and_data_analysis_agent = PROMPT_3_LIT_REVIEW
hypothesis_refinement_and_critical_reflection_agent = PROMPT_1_NOVELTY_TECH
methodological_novelty_agent = PROMPT_6_METHOD_CLARITY_RIGOR
experimental_novelty_agent = PROMPT_2_EXPERIMENTS
problem_Formulation_novelty_Agent = PROMPT_4_SCOPE_GENERALIZABILITY
writing_claim_novelty_agent = PROMPT_5_CLAIMS_OVERCLAIMING


# ============================================================
# 2) Master Agent prompt (strict + outputs final consolidated report)
# ============================================================

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
# 4) Config
# ============================================================


TEXT_COL_MAIN = "input_text_for_novelty"
RELATED_COL = "relevant_papers_list"

MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# token budgets
MAX_CTX = 92_000
RESERVED_FOR_CONVO = 25_000
RESERVED_FOR_GEN = 2_500
MAIN_TOK_BUDGET = 40_000
RELATED_TOK_BUDGET = 20_000

MAX_TOKENS_PER_REPLY = 500
TEMPERATURE = 0.2
TIMEOUT = 600

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
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = ast.literal_eval(s)
                return json.dumps(parsed, ensure_ascii=False, indent=2)[:200000]
            except Exception:
                return s
        return s
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)[:200000]
        except Exception:
            return str(x)
    return str(x)

def build_dual_paper_input(main_text: str, related_text: str) -> str:
    main_text = truncate_to_tokens(main_text, MAIN_TOK_BUDGET, keep="head")
    related_text = truncate_to_tokens(related_text, RELATED_TOK_BUDGET, keep="head")

    combined = f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text}

=== RETRIEVED RELEVANT PAPERS / CONTEXT (Paper B) ===
{related_text}
""".strip()

    max_input_budget = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
    if tok_len(combined) > max_input_budget:
        main_budget = max(10_000, MAIN_TOK_BUDGET - 10_000)
        main_text2 = truncate_to_tokens(main_text, main_budget, keep="head")
        combined = f"""
=== MAIN PAPER / QUERY (Paper A) ===
{main_text2}

=== RETRIEVED RELEVANT PAPERS / CONTEXT (Paper B) ===
{related_text}
""".strip()

    return combined


# ============================================================
# 7) Output-format wrapper + self-repair instruction
# ============================================================

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
# 8) Driver message
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
8) Master_Agent (consolidate + overall score)

Rules:
- Specialists MUST output Novelty Score/Reasons/Evidence in the required format.
- Master_Agent MUST compute overall novelty score in [0,1] and summarize.
""".strip()


# ============================================================
# 9) Main pipeline (includes one-shot rewrite if JSON/toolcall leaks)
# ============================================================

def run_pipeline():
    df_all = pd.read_csv(INPUT_CSV)
    df = df_all.iloc[ROW_START:ROW_END].copy()

    # Ensure required columns exist
    for col, default in [
        ("novelty_report", "PENDING"),
        ("planning_output", "PENDING"),
        ("lit_score_reason", "PENDING"),
        ("hyp_score_reason", "PENDING"),
        ("meth_score_reason", "PENDING"),
        ("exp_score_reason", "PENDING"),
        ("prob_score_reason", "PENDING"),
        ("write_score_reason", "PENDING"),
        ("full_chat_history", "PENDING"),
    ]:
        if col not in df.columns:
            df[col] = default

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=999,
        code_execution_config=False,
        default_auto_reply="Continue."
    )

    for i in tqdm(range(len(df)), desc="AutoGen Novelty (RAG-based)"):
        row = df.iloc[i]

        main_text_raw = normalize_any_to_text(row.get(TEXT_COL_MAIN, ""))
        related_raw = normalize_any_to_text(row.get(RELATED_COL, ""))

        if len(main_text_raw) < 200:
            df.at[df.index[i], "novelty_report"] = "SKIPPED_SHORT_MAIN_TEXT"
            continue

        if len(related_raw) < 50:
            related_raw = "INSUFFICIENT RETRIEVED CONTEXT PROVIDED."

        combined_input = build_dual_paper_input(main_text_raw, related_raw)
        combined_tokens = tok_len(combined_input)

        # Agents
        planning = autogen.AssistantAgent(
            name="Planning_Agent",
            system_message=HARSH_REVIEWER_POLICY + "\n\n" + planning_agent + "\n\n" + NO_TOOL_NO_JSON + "\n\n" + combined_input,
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

        agents_list = [
            user_proxy,
            planning,
            lit,
            hyp,
            meth,
            exp,
            prob,
            write,
            master,
        ]

        max_round = len(agents_list)

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

            master_raw = last_msg("Master_Agent")
            master_out = (master_raw or "").strip()

            # --- HARD FILTER: if tool-call/JSON leaked, ask Master to rewrite once ---
            bad_markers = ["{", "}", '"parameters"', "parameters", '"name"', 'name":']
            if any(m in master_out for m in bad_markers) or ("Overall novelty score:" not in master_out):
                rewrite_msg = (
                    "REWRITE ONLY. Your previous output was INVALID (contained JSON/tool-call or missing the required headings). "
                    "Now output ONLY plain text using the EXACT required format. "
                    "Do NOT include braces, JSON, or the words parameters/name. "
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

                # get rewritten output
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
    print(json.dumps(info, indent=2)[:5000])

    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=600):
        raise RuntimeError(f"vLLM not ready at {VLLM_BASE_URL}")

    # One-shot sanity test (tool_choice=none + tools=[])
    print("\n[Sanity test] /chat/completions with tool_choice=none ...")
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
