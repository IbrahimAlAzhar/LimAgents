

import os
import pandas as pd
import autogen
import sys
import time
import ast
import re
from tqdm import tqdm
import tiktoken
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION
# ==========================================

# ✅ Do NOT hardcode keys in code. Set it in your shell:
# export OPENAI_API_KEY="..." 
os.environ['OPENAI_API_KEY'] = ''
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"
# Safe limit for text truncation helper
SAFE_INPUT_LIMIT = 48000

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2,
    "timeout": 120,
    "cache_seed": None
}

client = OpenAI(api_key=api_key)

MASTER_MODEL_ID = "gpt-4o-mini"
MASTER_TEMPERATURE = 0.0
MASTER_MAX_TOKENS = 1800

def extract_master_handoff(chat_history: list) -> str:
    """
    Finds the Leader_Agent message that contains the MASTER_HANDOFF block.
    """
    for msg in reversed(chat_history):
        if msg.get("name") == "Leader_Agent" and msg.get("content"):
            c = msg["content"]
            if "=== MASTER_HANDOFF_START ===" in c and "=== MASTER_HANDOFF_END ===" in c:
                start = c.index("=== MASTER_HANDOFF_START ===")
                end = c.index("=== MASTER_HANDOFF_END ===") + len("=== MASTER_HANDOFF_END ===")
                return c[start:end]
    return ""

def run_master_outside(paper_and_citations: str, master_handoff: str) -> str:
    """
    Separate call: Master merges specialist limitations into final limitations.
    """
    system_msg = (
        "You are the **Master Agent**. Your job is to merge specialist limitations into one final, non-redundant list.\n"
        "Rules:\n"
        "- Use ONLY what appears in the handoff (do not invent new limitations).\n"
        "- Merge duplicates, keep specificity/evidence.\n"
        "- Output format:\n"
        "Here is the consolidated list of key limitations identified in the paper:\n"
        "- **Category:** ...\n"
    )

    user_msg = (
        "Leader Agent, here are the limitation analyses from the team.\n"
        "Please synthesize them into a single consolidated list, grouped by category.\n\n"
        f"{master_handoff}"
    )

    resp = client.chat.completions.create(
        model=MASTER_MODEL_ID,
        temperature=MASTER_TEMPERATURE,
        max_tokens=MASTER_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()

# Paths (UPDATED INPUT; output kept in same folder)

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. PROMPT DEFINITIONS (EXACTLY AS PROVIDED, only small update: include kg_triplets)
# ==========================================

def get_novelty_significance_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
The review_leader will ask for your feedback; respond thoroughly and ask clarifying questions if needed. When finished, inform the review_leader and provide list of novelty- and significance-related limitations with explanations and evidence from the paper. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

def get_citation_agent_prompt(paper_content: str, citation_content: str, kg_triplets: str) -> str:
    return f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Use KG_TRIPLETS (extracted from the paper) as extra evidence when helpful.
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}
=== KG_TRIPLETS ===
{kg_triplets}"""

def get_theoretical_methodological_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
Identify issues like unstated or overly strong assumptions, incomplete theoretical analysis, errors in derivations, methods that only work under restricted conditions not clearly acknowledged, missing ablations, lack of isolation of individual contributions, or ablations that do not convincingly attribute performance gains.
The review_leader will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the review_leader and deliver a list of theoretical, methodological, and ablation-related limitations with supporting evidence. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

def get_experimental_evaluation_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
Highlight issues like small-scale experiments, missing error bars or confidence intervals, unreported failed experiments, outdated or weak baselines, missing key competitors, unfair hyperparameter tuning, reliance on misleading metrics, missing standard metrics, or overemphasis on minor gains without practical or statistical significance.
The review_leader will interact with you; respond critically and seek clarification if needed. When finished, inform the review_leader and provide a list of experimental evaluation-related limitations, including validation, comparisons, baselines, and metrics. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings (e.g., different datasets, domains, noise, adversarial conditions), is practical in terms of resources (time, memory, hardware, scalability), and addresses genuine deployment needs without ignoring real-world constraints.
Point out limitations like overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, poor performance under shifts, excessive training/inference demands, high resource needs restricting deployment, reliance on synthetic data, ignoring constraints like cost or latency, lack of user studies or field tests, or over-optimistic assumptions about environments.
The review_leader will seek your input; respond thoroughly and clarify ambiguities. When finished, inform the review_leader and provide a list of generalization-, robustness-, efficiency-, and applicability-related limitations. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations of methods, settings, concepts, or organization hindering understanding; lack of explainability or insights into decisions; and insufficient details for replication, such as code, data, hyperparameters, or protocols.
Identify issues like ambiguities, unstated assumptions, vague terms undermining comprehension, black-box behavior without explanations, missing feature importance or mechanistic understanding, poorly organized sections, missing code/data release, unreported seeds, ambiguous procedures, or lack of open science practices.
The review_leader will ask questions; respond and ask follow-up questions if needed. When done, inform the review_leader and provide a list of clarity-, interpretability-, and reproducibility-related limitations, including suggestions for improvement where relevant. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

def get_data_ethics_prompt(paper_content: str, kg_triplets: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for issues in collection, labeling, cleaning, representativeness, or documentation; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Point out limitations such as small or non-diverse data, labeling errors, undocumented preprocessing, data leakage, reliance on flawed datasets without validation, biased outcomes leading to discrimination, lack of fairness metrics, unreported subgroup performance, ethical oversights, or failure to discuss misuse potential.
The review_leader will consult you; provide evidence-based critique and ask clarifying questions. When done, inform the review_leader and provide a list of data integrity-, bias-, fairness-, and ethics-related limitations. You also have access to KG_TRIPLETS extracted from the paper; use them as additional evidence when helpful.
PAPER CONTENT:
{paper_content}

KG_TRIPLETS:
{kg_triplets}"""

# ✅ Literature Agent Prompt (uses rag_top3_concatenate_lim_peer + kg_triplets)
def get_literature_agent_prompt(paper_content: str, rag_related_limitations: str, kg_triplets: str) -> str:
    return f"""You are the **Literature Agent**.
You are part of a team identifying limitations in a scientific paper. You are given:
(1) the target PAPER CONTENT,
(2) KG_TRIPLETS extracted from the target paper, and
(3) LIMITATIONS extracted from RELATED PAPERS (retrieved via RAG).

Your job:
- Treat the related-paper limitations as *literature signals* (what similar papers were criticized for).
- Cross-check these signals against the target paper and its KG_TRIPLETS and propose *only* limitations that are plausibly applicable to the target paper.
- If a limitation from related papers is NOT supported by evidence in the target paper or KG_TRIPLETS, do NOT assert it as a definite limitation. Instead, either:
  (a) omit it, or
  (b) include it as a **Potential limitation** and explicitly say what evidence is missing.
- Be specific and grounded. Whenever possible, cite short evidence phrases from the target paper (and/or cite relevant KG triplets).
- Output a list in the style:
  - [Limitation]: Explanation + evidence from target paper; optionally reference related-paper signal.

=== TARGET PAPER CONTENT ===
{paper_content}

=== KG_TRIPLETS (from target paper) ===
{kg_triplets}

=== RELATED PAPERS LIMITATIONS (RAG: top-3 concatenated) ===
{rag_related_limitations}
"""

def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    return f"""You are the **Leader Agent**. Coordinate specialist agents to produce strong, evidence-based limitations.

Available specialist agents: {agent_names}

PROTOCOL (follow strictly):
1) Ask each specialist agent to analyze the paper (and Citation_Agent for citation-based issues).
2) If any agent output is vague, ask follow-up questions until it is specific and grounded.
3) When you are satisfied, PRODUCE A FINAL HANDOFF in this exact format:

=== MASTER_HANDOFF_START ===
[Novelty_Significance_Agent]
<final list from that agent>

[Theoretical_Methodological_Agent]
<final list from that agent>

[Experimental_Evaluation_Agent]
<final list from that agent>

[Generalization_Robustness_Efficiency_Agent]
<final list from that agent>

[Clarity_Interpretability_Reproducibility_Agent]
<final list from that agent>

[Data_Ethics_Agent]
<final list from that agent>

[Literature_Agent]
<final list from that agent>

[Citation_Agent]
<final list from that agent>
=== MASTER_HANDOFF_END ===

4) After printing the handoff block, respond only with: TERMINATE

PAPER CONTENT:
{paper_content}
"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent**."""  # Placeholder as we use run_master_outside

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def truncate_text_to_tokens(text: str, max_tokens: int = SAFE_INPUT_LIMIT) -> str:
    if not text:
        return ""
    try:
        import tiktoken
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        print(f"  ⚠️ Truncating input: {len(tokens)} tokens -> {max_tokens} tokens.")
        return encoding.decode(tokens[:max_tokens]) + "... [TRUNCATED]"

    except Exception as e:
        approx_chars = max_tokens * 4
        if len(text) <= approx_chars:
            return text
        print(f"  ⚠️ tiktoken unavailable ({e}); using char-based truncation.")
        return text[:approx_chars] + "... [TRUNCATED]"

def parse_master_handoff_sections(master_handoff: str) -> dict:
    """
    Parses the Leader's MASTER_HANDOFF block into {SectionName: section_text}.
    Example keys: "Novelty_Significance_Agent", "Citation_Agent", ...
    """
    if not isinstance(master_handoff, str) or not master_handoff.strip():
        return {}

    pattern = r"\[([^\]]+)\]\s*\n(.*?)(?=\n\s*\[[^\]]+\]\s*\n|\n\s*=== MASTER_HANDOFF_END ===)"
    matches = re.findall(pattern, master_handoff, flags=re.DOTALL)

    out = {}
    for name, content in matches:
        out[name.strip()] = content.strip()
    return out

def extract_last_agent_message(chat_history: list, agent_name: str) -> str:
    """
    Fallback: returns last message content from a specific agent in chat_history.
    """
    if not isinstance(chat_history, list):
        return ""
    for msg in reversed(chat_history):
        if msg.get("name") == agent_name and msg.get("content"):
            return str(msg["content"]).strip()
    return ""

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df = pd.read_csv(INPUT_CSV) 

        df = df[df['kg_triplets'].astype(str) != 'SKIPPED_NO_TEXT']
        df = df.reset_index(drop=True)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Initialize Columns
    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    # ✅ NEW: Store each agent's final output in its own column
    agent_output_cols = {
        "Novelty_Significance_Agent": "out_novelty_significance_agent",
        "Theoretical_Methodological_Agent": "out_theoretical_methodological_agent",
        "Experimental_Evaluation_Agent": "out_experimental_evaluation_agent",
        "Generalization_Robustness_Efficiency_Agent": "out_generalization_robustness_efficiency_agent",
        "Clarity_Interpretability_Reproducibility_Agent": "out_clarity_interpretability_reproducibility_agent",
        "Data_Ethics_Agent": "out_data_ethics_agent",
        "Literature_Agent": "out_literature_agent",
        "Citation_Agent": "out_citation_agent",
    }
    for col in agent_output_cols.values():
        if col not in df.columns:
            df[col] = ""

    # User Proxy (Constant)
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # --- LIST OF STANDARD SPECIALISTS ---
    standard_specialist_config = [
        ("Novelty_Significance", get_novelty_significance_prompt),
        ("Theoretical_Methodological", get_theoretical_methodological_prompt),
        ("Experimental_Evaluation", get_experimental_evaluation_prompt),
        ("Generalization_Robustness_Efficiency", get_generalization_robustness_efficiency_prompt),
        ("Clarity_Interpretability_Reproducibility", get_clarity_interpretability_reproducibility_prompt),
        ("Data_Ethics", get_data_ethics_prompt),
    ]

    all_agent_names = [name + "_Agent" for name, _ in standard_specialist_config]
    # ✅ Add Literature_Agent + Citation_Agent
    all_agent_names.append("Literature_Agent")
    all_agent_names.append("Citation_Agent")
    agent_names_str = ", ".join(all_agent_names)

    for i in tqdm(range(0, len(df))):
        row = df.iloc[i]

        # 1. Main Paper Text (USE input_text_cleaned)
        paper_text = str(row.get("input_text_cleaned", ""))

        # ✅ NEW: KG triplets (from column kg_triplets)
        kg_triplets = row.get("kg_triplets", "")
        if pd.isna(kg_triplets) or kg_triplets is None:
            kg_triplets = ""
        kg_triplets = str(kg_triplets).strip()
        if not kg_triplets:
            kg_triplets = "No KG triplets available."

        # ✅ Literature RAG limitations (from related papers)
        rag_lim_peer = row.get("rag_top3_concatenate_lim_peer", "")
        if pd.isna(rag_lim_peer) or rag_lim_peer is None:
            rag_lim_peer = ""
        rag_lim_peer = str(rag_lim_peer).strip()
        if not rag_lim_peer:
            rag_lim_peer = "No related-paper limitations available."

        # 2. CITATION DATA PREP (FROM gpt_ranked_chunks: top 3)
        raw_ranked_chunks = row.get("gpt_ranked_chunks", "[]")
        print('raw_ranked_chunks', raw_ranked_chunks)
        citation_chunks_str = ""

        try:
            # Convert string -> list (or accept list)
            if isinstance(raw_ranked_chunks, str):
                chunks_list = ast.literal_eval(raw_ranked_chunks)
            elif isinstance(raw_ranked_chunks, list):
                chunks_list = raw_ranked_chunks
            else:
                chunks_list = []

            # Take top 3 items
            top_3_chunks = chunks_list[:3] if isinstance(chunks_list, list) else []

            formatted_list = []
            for idx, item in enumerate(top_3_chunks):
                # Most common: item = [chunk_text, score] or [chunk_text, ...]
                if isinstance(item, list) and len(item) > 0:
                    chunk_text = item[0]
                    formatted_list.append(f"[Chunk {idx+1}]: {chunk_text}")
                # Sometimes: item is a dict
                elif isinstance(item, dict):
                    # Try common keys; fallback to full dict string
                    chunk_text = item.get("chunk") or item.get("text") or item.get("content") or str(item)
                    formatted_list.append(f"[Chunk {idx+1}]: {chunk_text}")
                # Sometimes: item is already a string
                elif isinstance(item, str):
                    formatted_list.append(f"[Chunk {idx+1}]: {item}")
                else:
                    formatted_list.append(f"[Chunk {idx+1}]: {str(item)}")

            citation_chunks_str = "\n\n".join(formatted_list) if formatted_list else "No citation chunks available."

        except Exception as e:
            print(f"Error parsing gpt_ranked_chunks at row {i}: {e}")
            citation_chunks_str = "No citation chunks available."

        # === TOKEN SAFEGUARDS ===
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=40000)
        citation_chunks_str = truncate_text_to_tokens(citation_chunks_str, max_tokens=10000)
        rag_lim_peer = truncate_text_to_tokens(rag_lim_peer, max_tokens=10000)
        kg_triplets = truncate_text_to_tokens(kg_triplets, max_tokens=8000)

        print('citation_chunks_str', citation_chunks_str)

        # Skip short text
        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue

        combined_content = f"""
        === MAIN PAPER CONTENT ===
        {paper_text}

        === CITED PAPERS CONTEXT (Top Ranked) ===
        {citation_chunks_str}

        === KG_TRIPLETS (from target paper) ===
        {kg_triplets}
        """

        # ==========================================
        # DYNAMIC AGENT CREATION
        # ==========================================

        agents_list = [user_proxy]

        # 1. Leader Agent
        leader_prompt = get_leader_agent_prompt(combined_content, agent_names=agent_names_str)
        if i < 5:
            print("\n" + "=" * 120)
            print(f"ROW {i} - Leader_Agent INPUT (system_message)")
            print(leader_prompt)
            print("=" * 120 + "\n")
        leader_agent = autogen.AssistantAgent(
            name="Leader_Agent",
            system_message=leader_prompt,
            llm_config=llm_config
        )
        agents_list.append(leader_agent)

        # 2. Citation Agent (Uses paper + top 3 ranked chunks + kg_triplets)
        citation_sys_msg = get_citation_agent_prompt(paper_text, citation_chunks_str, kg_triplets)
        if i < 5:
            print("\n" + "=" * 120)
            print(f"ROW {i} - Citation_Agent INPUT (system_message)")
            print(citation_sys_msg)
            print("=" * 120 + "\n")
        citation_agent = autogen.AssistantAgent(
            name="Citation_Agent",
            system_message=citation_sys_msg,
            llm_config=llm_config
        )
        agents_list.append(citation_agent)

        # 3. Literature Agent (Uses paper + rag_top3_concatenate_lim_peer + kg_triplets)
        literature_sys_msg = get_literature_agent_prompt(paper_text, rag_lim_peer, kg_triplets)
        if i < 5:
            print("\n" + "=" * 120)
            print(f"ROW {i} - Literature_Agent INPUT (system_message)")
            print(literature_sys_msg)
            print("=" * 120 + "\n")
        literature_agent = autogen.AssistantAgent(
            name="Literature_Agent",
            system_message=literature_sys_msg,
            llm_config=llm_config
        )
        agents_list.append(literature_agent)

        # 4. Standard Specialists (paper + kg_triplets)
        for agent_name_base, prompt_func in standard_specialist_config:
            sys_msg = prompt_func(paper_text, kg_triplets)
            if i < 5:
                print("\n" + "=" * 120)
                print(f"ROW {i} - {agent_name_base}_Agent INPUT (system_message)")
                print(sys_msg)
                print("=" * 120 + "\n")
            specialist = autogen.AssistantAgent(
                name=f"{agent_name_base}_Agent",
                system_message=sys_msg,
                llm_config=llm_config
            )
            agents_list.append(specialist)

        # 5. Group Chat
        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=45,
            speaker_selection_method="auto"
        )

        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # 6. Initiate
        task_msg = """Leader_Agent, please start the limitation analysis process for the provided paper content."""

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True
            )
            chat_hist = chat_result.chat_history

            # 1) extract handoff from Leader
            handoff = extract_master_handoff(chat_hist)

            if not handoff:
                final_output = "NO_MASTER_HANDOFF_FROM_LEADER"
            else:
                # 2) master is OUTSIDE autogen
                final_output = run_master_outside(combined_content, handoff)

            # ✅ NEW: Extract and store each agent's final response in new columns
            sections = parse_master_handoff_sections(handoff) if handoff else {}
            for agent_name, col_name in agent_output_cols.items():
                if handoff and agent_name in sections and sections[agent_name].strip():
                    df.at[df.index[i], col_name] = sections[agent_name]
                else:
                    df.at[df.index[i], col_name] = extract_last_agent_message(chat_hist, agent_name)

            df.at[df.index[i], "final_merged_limitations"] = final_output
            df.at[df.index[i], "full_chat_history"] = str(chat_hist)

        except Exception as e:
            print(f"Error on row {i}: {e}")
            df.at[df.index[i], "final_merged_limitations"] = f"ERROR: {e}"
            df.at[df.index[i], "full_chat_history"] = ""

        if i % 5 == 0:
            df.to_csv(OUTPUT_SLICE, index=False)
        time.sleep(3)

    df.to_csv(OUTPUT_SLICE, index=False)

if __name__ == "__main__":
    run_pipeline()
