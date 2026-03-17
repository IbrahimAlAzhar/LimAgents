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

MODEL_PATH = "/lstr/sahara/datalab-ml/ibrahim/llama3_1_70b_awq"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "llama31-70b-awq")

# ---- Token budgets ----
MAX_CTX = 92_000
RESERVED_FOR_CONVO = 30_000
RESERVED_FOR_GEN = 2_000
PAPER_TOK_BUDGET = 50_000
CITATION_TOK_BUDGET = 6_000

MAX_TOKENS_PER_REPLY = 450
MAX_ROUND = 9  # user + leader + citation + 6 specialists = 9 turns-ish with round_robin

TEMPERATURE = 0.2
TIMEOUT = 600

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# =========================
# 2) LLM config for AutoGen (70B)
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
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance.
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
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis.
When done, deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics.
When finished, provide a bullet list of experimental evaluation-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability.
When finished, provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility.
When finished, provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations.
When finished, provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations with evidence.
PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    # ✅ Leader must ALSO summarize at the end (since no master)
    return f"""You are the **Leader Agent**.
You coordinate specialist agents to produce a comprehensive limitation list.
Available specialist agents: {agent_names}

Protocol:
1) In your FIRST message, instruct each specialist briefly what to do.
2) After specialists respond, produce a FINAL OUTPUT with:
   - A short section per agent (1–5 bullets)
   - Then a final section called: "LEADER_FINAL_MERGED_LIMITATIONS:" with a grouped bullet list (non-redundant)

Important:
- Keep limitations specific and grounded.
- Do NOT invent new limitations beyond specialist outputs.

PAPER CONTENT:
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
        shrink_budget = max_input_budget - tok_len(citation_text) - 2000
        shrink_budget = max(10_000, shrink_budget)
        paper_text2 = truncate_to_tokens(paper_text, shrink_budget, keep="head")
        combined = f"""
=== MAIN PAPER CONTENT ===
{paper_text2}

=== CITED PAPERS CONTEXT ===
{citation_text}
""".strip()

    return combined

# =========================
# 6) Main pipeline (Leader-only)
# =========================
def run_pipeline():
    df_all = pd.read_csv(INPUT_CSV)
    df = df_all.iloc[ROW_START:ROW_END].copy()

    # ✅ new columns
    if "leader_final_output" not in df.columns:
        df["leader_final_output"] = "PENDING"
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

    for i in tqdm(range(len(df)), desc="StageA Leader-only (70B)"):
        row = df.iloc[i]

        paper_text_raw = str(row.get(TEXT_COL, "") or "")
        citation_text_raw = extract_intro_and_abstract(row.get(CITED_COL, ""))

        if len(paper_text_raw) < 100:
            df.at[df.index[i], "leader_final_output"] = "SKIPPED_SHORT_TEXT"
            continue

        paper_text_for_agents = truncate_to_tokens(paper_text_raw, PAPER_TOK_BUDGET, keep="head")
        citation_text_for_agents = truncate_to_tokens(citation_text_raw, CITATION_TOK_BUDGET, keep="head")
        combined_content = build_combined_content(paper_text_for_agents, citation_text_for_agents)

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

        # ✅ NO MASTER AGENT
        agents_list = [user_proxy, leader_agent, citation_agent] + specialists

        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=MAX_ROUND,
            speaker_selection_method="round_robin",
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        task_msg = (
            "Leader_Agent: Instruct each specialist to provide bullet-point limitations with evidence. "
            "After everyone responds, you (Leader_Agent) must output your final merged list under the header "
            "'LEADER_FINAL_MERGED_LIMITATIONS:'."
        )

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True,
            )

            # ✅ capture the LAST Leader message (final)
            leader_msgs = []
            for msg in chat_result.chat_history:
                if msg.get("name") == "Leader_Agent" and msg.get("content"):
                    leader_msgs.append(msg.get("content").strip())

            final_leader = leader_msgs[-1] if leader_msgs else "NO_OUTPUT_FROM_LEADER"
            df.at[df.index[i], "leader_final_output"] = final_leader
            df.at[df.index[i], "full_chat_history"] = str(chat_result.chat_history)

        except Exception as e:
            df.at[df.index[i], "leader_final_output"] = f"ERROR: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved StageA output: {OUTPUT_CSV}")

if __name__ == "__main__":
    print("[CLI args]", sys.argv)
    print(f"Using vLLM base_url: {VLLM_BASE_URL}")
    print(f"Using model name  : {VLLM_MODEL}")

    print("\n[Best-effort vLLM server info]")
    info = get_vllm_cli_args(VLLM_BASE_URL)
    print(json.dumps(info, indent=2)[:5000])

    if not wait_for_vllm(VLLM_BASE_URL, timeout_s=600):
        raise RuntimeError(f"vLLM not ready at {VLLM_BASE_URL}")

    run_pipeline()
