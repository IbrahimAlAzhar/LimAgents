import os
import sys
import time
import re
import ast
import json
import pandas as pd
from tqdm import tqdm

import autogen
from transformers import AutoTokenizer
import requests


# =========================
# vLLM readiness check
# =========================
def wait_for_vllm(base_url: str, timeout_s: int = 600) -> bool:
    """
    Poll /v1/models until the vLLM OpenAI server is ready.
    base_url should look like: http://<host>:8000/v1
    """
    t0 = time.time()
    base_url = base_url.rstrip("/")
    models_url = base_url + "/models"  # GET /v1/models
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
# Best-effort server info dump
# =========================
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


# =========================
# 1) Config

# ROW_START = 0
# ROW_END = 1  # exclusive

# TEXT_COL = "input_text_cleaned"
# CHUNKS_COL = "gpt_ranked_chunks"
# KG_COL = "kg_triplets"

# MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"

# VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
# VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# # ---- Token budgets ----

ROW_START = 0
ROW_END = 1  # exclusive

TEXT_COL = "input_text_cleaned"
CHUNKS_COL = "gpt_ranked_chunks"
KG_COL = "kg_triplets"

MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# IMPORTANT: match this to vLLM --max-model-len (PBS sets this env var)
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "60000"))

# ---- Token budgets (adaptive) ----
# We keep slack for:
# - system prompts + multi-agent chat history
# - generation
# MAX_CTX = 92_000
# RESERVED_FOR_CONVO = 30_000
# RESERVED_FOR_GEN = 2_000
# PAPER_TOK_BUDGET = 50_000
# CITATION_TOK_BUDGET = 6_000
# MAX_TOKENS_PER_REPLY = 400
# MAX_ROUND = 10
# TEMPERATURE = 0.2
# TIMEOUT = 600

# The values below are conservative for stability. 
MAX_CTX = max(92_000, VLLM_MAX_MODEL_LEN - 1_000)

RESERVED_FOR_CONVO = min(30_000, MAX_CTX // 2)    # chat history reserve
RESERVED_FOR_GEN = min(2_000, MAX_CTX // 20)      # generation reserve

# Allocate remaining to paper + citations.
# Keep a safety margin so we never hit the hard limit.
AVAILABLE_FOR_INPUT = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
AVAILABLE_FOR_INPUT = max(4_000, AVAILABLE_FOR_INPUT)

# Split paper vs citation
PAPER_TOK_BUDGET = min(50_000, max(2_000, int(AVAILABLE_FOR_INPUT * 0.82)))
CITATION_TOK_BUDGET = min(6_000, max(1_000, AVAILABLE_FOR_INPUT - PAPER_TOK_BUDGET))

# With round_robin, set max_round roughly to number of agents so each speaks once.
MAX_TOKENS_PER_REPLY = 400
MAX_ROUND = 10

TEMPERATURE = 0.2
TIMEOUT = 600

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)


# =========================
# 2) LLM config for AutoGen
# =========================
llm_config = {
    "config_list": [{
        "api_type": "openai",
        "model": VLLM_MODEL,
        "api_key": "EMPTY",
        "base_url": VLLM_BASE_URL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_PER_REPLY,
        "extra_body": {"parallel_tool_calls": False},
    }],
    "timeout": TIMEOUT,
    "cache_seed": None,
}


# =========================
# 3) Termination detection
# =========================
def is_terminate_msg(msg) -> bool:
    c = (msg.get("content") or "").strip()
    return c == "TERMINATE" or c.endswith("\nTERMINATE") or c.endswith("TERMINATE")


# =========================
# 4) Prompts
# =========================
def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
When finished, provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
When done, deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
When finished, provide a bullet list of experimental evaluation-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings, is practical in resources, and addresses deployment constraints.
When finished, provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations and missing details needed to reproduce results.
When finished, provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations.
When finished, provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    return f"""You are the **Leader Agent**.
You are coordinating a group of specialist agents to produce a comprehensive limitation list.
Available specialist agents: {agent_names}

In your first message:
- Tell each specialist what to do (briefly).
- Ask them to output bullet limitations with evidence.
- Then instruct the Master Agent to consolidate (after specialists speak).

PAPER CONTENT:
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent**.
You will receive limitation analyses from specialist agents and produce ONE consolidated list.
Rules:
- Integrate specialist outputs.
- Remove redundancy (merge similar limitations).
- Keep specificity and evidence.
- Do NOT invent new limitations beyond what specialists raised.

Output format:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then bullets, grouped by category.

PAPER CONTENT (for context):
{paper_content}"""


# =========================
# 5) Token helpers
# =========================
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

def build_combined_content(paper_text: str, citation_text: str) -> str:
    paper_text = truncate_to_tokens(paper_text, PAPER_TOK_BUDGET, keep="head")
    citation_text = truncate_to_tokens(citation_text, CITATION_TOK_BUDGET, keep="head")

    combined = f"""
=== MAIN PAPER CONTENT ===
{paper_text}

=== CITED PAPERS CONTEXT ===
{citation_text}
""".strip()

    # Ensure we fit within available input budget
    max_input_budget = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
    if tok_len(combined) > max_input_budget:
        # shrink paper first
        shrink_budget = max_input_budget - tok_len(citation_text) - 1500
        shrink_budget = max(2000, shrink_budget)
        paper_text2 = truncate_to_tokens(paper_text, shrink_budget, keep="head")
        combined = f"""
=== MAIN PAPER CONTENT ===
{paper_text2}

=== CITED PAPERS CONTEXT ===
{citation_text}
""".strip()

    return combined


# =========================
# 6) Main pipeline
# =========================
def run_pipeline():
    df_all = pd.read_csv(INPUT_CSV)
    df = df_all.iloc[ROW_START:ROW_END].copy()

    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=999,
        code_execution_config=False,
        default_auto_reply="Continue. Do not ask me questions; proceed with the analysis."
    )

    specialist_config = [
        ("Novelty_Significance", get_novelty_significance_prompt),
        ("Theoretical_Methodological", get_theoretical_methodological_prompt),
        ("Experimental_Evaluation", get_experimental_evaluation_prompt),
        ("Generalization_Robustness_Efficiency", get_generalization_robustness_efficiency_prompt),
        ("Clarity_Interpretability_Reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("Data_Ethics", get_data_ethics_prompt),
    ]

    all_agent_names = [name + "_Agent" for name, _ in specialist_config] + ["Citation_Agent"]
    agent_names_str = ", ".join(all_agent_names)

    for i in tqdm(range(len(df)), desc="AutoGen Llama31"):
        row = df.iloc[i]

        paper_text_raw = str(row.get(TEXT_COL, "") or "")
        kg_triplets_raw = str(row.get(KG_COL, "") or "")

        kg_block = ""
        if kg_triplets_raw.strip():
            kg_block = f"\n\n=== KG TRIPLETS (from paper) ===\n{kg_triplets_raw}".strip()

        paper_plus_kg_raw = (paper_text_raw + kg_block).strip()

        raw_ranked_chunks = row.get(CHUNKS_COL, "[]")
        citation_chunks_str = ""

        try:
            if isinstance(raw_ranked_chunks, str):
                chunks_list = ast.literal_eval(raw_ranked_chunks)
            elif isinstance(raw_ranked_chunks, list):
                chunks_list = raw_ranked_chunks
            else:
                chunks_list = []

            top_3_chunks = chunks_list[:3] if isinstance(chunks_list, list) else []

            formatted_list = []
            for idx, item in enumerate(top_3_chunks):
                if isinstance(item, list) and len(item) > 0:
                    chunk_text = item[0]
                    formatted_list.append(f"[Chunk {idx+1}]: {chunk_text}")
                elif isinstance(item, dict):
                    chunk_text = item.get("chunk") or item.get("text") or item.get("content") or str(item)
                    formatted_list.append(f"[Chunk {idx+1}]: {chunk_text}")
                elif isinstance(item, str):
                    formatted_list.append(f"[Chunk {idx+1}]: {item}")
                else:
                    formatted_list.append(f"[Chunk {idx+1}]: {str(item)}")

            citation_chunks_str = "\n\n".join(formatted_list) if formatted_list else "No citation chunks available."
        except Exception:
            citation_chunks_str = "No citation chunks available."

        if len(paper_plus_kg_raw) < 100:
            df.at[df.index[i], "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            continue

        paper_text_for_agents = truncate_to_tokens(paper_plus_kg_raw, PAPER_TOK_BUDGET, keep="head")
        citation_text_for_agents = truncate_to_tokens(citation_chunks_str, CITATION_TOK_BUDGET, keep="head")

        combined_content = build_combined_content(paper_text_for_agents, citation_text_for_agents)
        combined_tokens = tok_len(combined_content)

        leader_agent = autogen.AssistantAgent(
            name="Leader_Agent",
            system_message=get_leader_agent_prompt(combined_content, agent_names=agent_names_str),
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        citation_agent = autogen.AssistantAgent(
            name="Citation_Agent",
            system_message=get_citation_agent_prompt(paper_text_for_agents, citation_text_for_agents),
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        specialists = []
        for agent_name_base, prompt_func in specialist_config:
            specialists.append(
                autogen.AssistantAgent(
                    name=f"{agent_name_base}_Agent",
                    system_message=prompt_func(paper_text_for_agents),
                    llm_config=llm_config,
                    is_termination_msg=is_terminate_msg,
                )
            )

        master_agent = autogen.AssistantAgent(
            name="Master_Agent",
            system_message=get_master_agent_prompt(combined_content),
            llm_config=llm_config,
            is_termination_msg=is_terminate_msg,
        )

        agents_list = [user_proxy, leader_agent, citation_agent] + specialists + [master_agent]

        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=MAX_ROUND,
            speaker_selection_method="round_robin",
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        task_msg = (
            "Leader_Agent: Please instruct each specialist to provide bullet-point limitations with evidence. "
            "After specialists respond, Master_Agent should consolidate into a final grouped list."
        )

        try:
            print(f"\n[Row {ROW_START + i}] combined_content_tokens={combined_tokens} | VLLM_MAX_MODEL_LEN={VLLM_MAX_MODEL_LEN}")

            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True,
            )

            master_messages = []
            for msg in chat_result.chat_history:
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip().replace("TERMINATE", "").strip()
                    if len(content) > 50:
                        master_messages.append(content)

            final_output = master_messages[-1] if master_messages else "NO_OUTPUT_FROM_MASTER"
            df.at[df.index[i], "final_merged_limitations"] = final_output
            df.at[df.index[i], "full_chat_history"] = str(chat_result.chat_history)

        except Exception as e:
            df.at[df.index[i], "final_merged_limitations"] = f"ERROR: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    print("[CLI args]", sys.argv)
    print(f"Using vLLM base_url: {VLLM_BASE_URL}")
    print(f"Using model name  : {VLLM_MODEL}")
    print(f"Using max len     : {VLLM_MAX_MODEL_LEN}")
    print(f"MAX_CTX={MAX_CTX} RESERVED_FOR_CONVO={RESERVED_FOR_CONVO} RESERVED_FOR_GEN={RESERVED_FOR_GEN}")
    print(f"PAPER_TOK_BUDGET={PAPER_TOK_BUDGET} CITATION_TOK_BUDGET={CITATION_TOK_BUDGET}")

    print("\n[Env snapshot]")
    for k in ["VLLM_BASE_URL", "VLLM_MODEL", "VLLM_MAX_MODEL_LEN", "RAY_ADDRESS", "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG"]:
        print(f"{k}={os.environ.get(k)}")

    print("\n[Best-effort vLLM server info]")
    info = get_vllm_cli_args(VLLM_BASE_URL)
    print(json.dumps(info, indent=2)[:5000])

    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=600):
        raise RuntimeError(f"vLLM not ready at {VLLM_BASE_URL}")

    run_pipeline()
