import os
import sys
import time
import re
import ast
import json
import pandas as pd
from tqdm import tqdm
import requests
import autogen
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



TEXT_COL = "input_text_cleaned"
CITED_COL = "cited_in"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/mistral_7b_v3_instruct"

# Use the snapshot dir (you confirmed tokenizer files exist there)
MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/mistral_7b_v3_instruct/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "mistral-7b-instruct")  # must match --served-model-name

# ---- Token budgets ----
MAX_CTX = 32_768
RESERVED_FOR_CONVO = 8_000
RESERVED_FOR_GEN = 1_500
PAPER_TOK_BUDGET = 18_000
CITATION_TOK_BUDGET = 4_000

MAX_TOKENS_PER_REPLY = 400
TEMPERATURE = 0.2
TIMEOUT = 600

# group chat turns: (user + leader + citation + 6 specialists + master) ≈ 9 assistants + user
# give a bit extra for “continue” turns
MAX_ROUND = 24

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print("\n[Tokenizer]")
print(f"MODEL_ID    : {MODEL_ID}")
print(f"CACHE_DIR   : {CACHE_DIR}")
print(f"MODEL_PATH  : {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True
)


# =========================
# 2) LLM config for AutoGen (vLLM OpenAI-compatible)
# =========================
llm_config = {
    "config_list": [{
        "api_type": "openai",
        "model": VLLM_MODEL,
        "api_key": "EMPTY",
        "base_url": VLLM_BASE_URL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS_PER_REPLY,
        # important for some servers that don't like parallel tool call schema
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
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on novelty & significance issues (incremental contribution, overstated claims, weak motivation, niche impact).
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on theoretical/methodological soundness, assumptions, proofs, ablations/component analysis.
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on experimental evaluation (baselines, rigor, stats, metrics, comparisons, validation).
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on generalization, robustness, efficiency, practicality/deployment constraints.
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on clarity, interpretability, and reproducibility (missing details, ambiguity, missing settings).
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent identifying limitations in a scientific paper.
Focus ONLY on data, bias, fairness, ethics, privacy risks, dataset issues.
Return bullet points with brief evidence from the paper text.
PAPER CONTENT:
{paper_content}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the Citation Agent.
Compare MAIN ARTICLE vs CITED PAPERS INFO:
- missing related work coverage
- misinterpretation / selective citation
Return bullet points: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}"""

def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    return f"""You are the Leader Agent coordinating specialists: {agent_names}

STRICT RULES:
- Do NOT produce limitations yourself.
- Only delegate: assign each specialist what to focus on and ask for bullet limitations with evidence.
- Keep delegation short.

PAPER CONTENT:
{paper_content}"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the Master Agent.
You will read the specialists' outputs in this group chat and produce ONE consolidated limitation list.

Rules:
- Merge duplicates, keep specificity
- Do NOT invent new limitations not supported by specialist messages
- Group by category
- End your message with: TERMINATE

PAPER CONTENT (context):
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
    return tokenizer.decode(ids, skip_special_tokens=True)

def clean_text_detailed(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_intro_and_abstract(cited_entry) -> str:
    if pd.isna(cited_entry):
        return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""

    processed = []
    for idx, (_, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict):
            continue

        intro = ""
        for sec in data.get("sections", []):
            if isinstance(sec, dict) and "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break

        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)

        if t_clean or a_clean or i_clean:
            processed.append(
                f"'Paper{idx}_Title: {t_clean}', 'Paper{idx}_Abstract': '{a_clean}', 'Paper{idx}_Introduction': '{i_clean}'."
            )
    return "\n".join(processed)

def build_combined_content(paper_text: str, citation_text: str) -> str:
    paper_text = truncate_to_tokens(paper_text, PAPER_TOK_BUDGET, keep="head")
    citation_text = truncate_to_tokens(citation_text, CITATION_TOK_BUDGET, keep="head")

    combined = f"""
=== MAIN PAPER CONTENT ===
{paper_text}

=== CITED PAPERS CONTEXT ===
{citation_text}
""".strip()

    max_input_budget = MAX_CTX - RESERVED_FOR_CONVO - RESERVED_FOR_GEN
    if tok_len(combined) > max_input_budget:
        shrink_budget = max_input_budget - tok_len(citation_text) - 1500
        shrink_budget = max(8000, shrink_budget)
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
        default_auto_reply="Continue."
    )

    specialist_config = [
        ("Novelty_Significance", get_novelty_significance_prompt),
        ("Theoretical_Methodological", get_theoretical_methodological_prompt),
        ("Experimental_Evaluation", get_experimental_evaluation_prompt),
        ("Generalization_Robustness_Efficiency", get_generalization_robustness_efficiency_prompt),
        ("Clarity_Interpretability_Reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("Data_Ethics", get_data_ethics_prompt),
    ]

    specialist_names = [f"{name}_Agent" for name, _ in specialist_config]
    all_agent_names = ["Citation_Agent"] + specialist_names
    agent_names_str = ", ".join(all_agent_names)

    for i in tqdm(range(len(df)), desc="AutoGen Mistral"):
        row = df.iloc[i]
        paper_text_raw = str(row.get(TEXT_COL, "") or "")
        citation_text_raw = extract_intro_and_abstract(row.get(CITED_COL, ""))

        if len(paper_text_raw) < 100:
            df.at[df.index[i], "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            continue

        paper_text_for_agents = truncate_to_tokens(paper_text_raw, PAPER_TOK_BUDGET, keep="head")
        citation_text_for_agents = truncate_to_tokens(citation_text_raw, CITATION_TOK_BUDGET, keep="head")
        combined_content = build_combined_content(paper_text_for_agents, citation_text_for_agents)

        # Agents
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

        specialists = [
            autogen.AssistantAgent(
                name=f"{agent_name_base}_Agent",
                system_message=prompt_func(paper_text_for_agents),
                llm_config=llm_config,
                is_termination_msg=is_terminate_msg,
            )
            for agent_name_base, prompt_func in specialist_config
        ]

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

        task_msg = f"""
We will run a structured limitation-finding discussion.

Leader_Agent: delegate tasks ONLY (no limitations).
Citation_Agent + Specialists: output bullet-point limitations with evidence.
Master_Agent: consolidate into a single non-redundant list grouped by category and end with TERMINATE.

Begin now.
""".strip()

        try:
            print(f"\n[Row {ROW_START + i}] combined_tokens={tok_len(combined_content)} | max_round={MAX_ROUND} | reply_max_tokens={MAX_TOKENS_PER_REPLY}")

            result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True,
            )

            # Extract Master output
            master_out = "NO_OUTPUT_FROM_MASTER"
            for msg in reversed(result.chat_history):
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    master_out = msg.get("content", "").replace("TERMINATE", "").strip()
                    break

            df.at[df.index[i], "final_merged_limitations"] = master_out
            df.at[df.index[i], "full_chat_history"] = str(result.chat_history)

        except Exception as e:
            df.at[df.index[i], "final_merged_limitations"] = f"ERROR: {e}"
            df.at[df.index[i], "full_chat_history"] = "ERROR"

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

    print("\n[Env snapshot]")
    for k in ["VLLM_BASE_URL", "VLLM_MODEL", "RAY_ADDRESS", "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG"]:
        print(f"{k}={os.environ.get(k)}")

    print("\n[Best-effort vLLM server info]")
    info = get_vllm_cli_args(VLLM_BASE_URL)
    print(json.dumps(info, indent=2)[:5000])

    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=600):
        raise RuntimeError(f"vLLM not ready at {VLLM_BASE_URL}")

    run_pipeline()
