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

os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4.1-mini"
# Safe limit for text truncation helper
SAFE_INPUT_LIMIT = 30000 # 48000

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2,
    "timeout": 120,
    "cache_seed": None
}

client = OpenAI(api_key=api_key)

MASTER_MODEL_ID = "gpt-4.1-mini" #  "gpt-4o-mini"
MASTER_TEMPERATURE = 0.0
MASTER_MAX_TOKENS = 1500

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
    Separate call: Master merges specialist outputs into final limitations.
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


os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. PROMPT DEFINITIONS (EXACTLY AS PROVIDED)
# ==========================================

def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
The review_leader will ask for your feedback; respond thoroughly and ask clarifying questions if needed. When finished, inform the review_leader and provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
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
Identify issues like unstated or overly strong assumptions, incomplete theoretical analysis, errors in derivations, methods that only work under restricted conditions not clearly acknowledged, missing ablations, lack of isolation of individual contributions, or ablations that do not convincingly attribute performance gains.
The review_leader will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the review_leader and deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""

def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
Highlight issues like small-scale experiments, missing error bars or confidence intervals, unreported failed experiments, outdated or weak baselines, missing key competitors, unfair hyperparameter tuning, reliance on misleading metrics, missing standard metrics, or overemphasis on minor gains without practical or statistical significance.
The review_leader will interact with you; respond critically and seek clarification if needed. When finished, inform the review_leader and provide a bullet list of experimental evaluation-related limitations, including validation, comparisons, baselines, and metrics.
PAPER CONTENT:
{paper_content}"""

def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings (e.g., different datasets, domains, noise, adversarial conditions), is practical in terms of resources (time, memory, hardware, scalability), and addresses genuine deployment needs without ignoring real-world constraints.
Point out limitations like overfitting to benchmarks, networkx lack of out-of-distribution testing, sensitivity to hyperparameters, poor performance under shifts, excessive training/inference demands, high resource needs restricting deployment, reliance on synthetic data, ignoring constraints like cost or latency, lack of user studies or field tests, or over-optimistic assumptions about environments.
The review_leader will seek your input; respond thoroughly and clarify ambiguities. When finished, inform the review_leader and provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations of methods, settings, concepts, or organization hindering understanding; lack of explainability or insights into decisions; and insufficient details for replication, such as code, data, hyperparameters, or protocols.
Identify issues like ambiguities, unstated assumptions, vague terms undermining comprehension, black-box behavior without explanations, missing feature importance or mechanistic understanding, poorly organized sections, missing code/data release, unreported seeds, ambiguous procedures, or lack of open science practices.
The review_leader will ask questions; respond and ask follow-up questions if needed. When done, inform the review_leader and provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, including suggestions for improvement where relevant.
PAPER CONTENT:
{paper_content}"""

def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for issues in collection, labeling, cleaning, representativeness, or documentation; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Point out limitations such as small or non-diverse data, labeling errors, undocumented preprocessing, data leakage, reliance on flawed datasets without validation, biased outcomes leading to discrimination, pink-box behavior without explanations, lack of fairness metrics, unreported subgroup performance, ethical oversights, or failure to discuss misuse potential.
The review_leader will consult you; provide evidence-based critique and ask clarifying questions. When done, inform the review_leader and provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""

def get_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    return f"""You are the **Leader Agent**. Coordinate specialist agents to produce strong, evidence-based limitations.

Available specialist agents: {agent_names}

PROTOCOL (follow strictly):
1) Ask each specialist agent to analyze the paper (and Citation_Agent for citation-based issues).
2) If any agent output is vague, ask follow-up questions until it is specific and grounded.
3) When you are satisfied, PRODUCE A FINAL HANDOFF in this exact format:

=== MASTER_HANDOFF_START ===
[Novelty_Significance_Agent]
<final bullet list from that agent>

[Theoretical_Methodological_Agent]
<final bullet list from that agent>

[Experimental_Evaluation_Agent]
<final bullet list from that agent>

[Generalization_Robustness_Efficiency_Agent]
<final bullet list from that agent>

[Clarity_Interpretability_Reproducibility_Agent]
<final bullet list from that agent>

[Data_Ethics_Agent]
<final bullet list from that agent>

[Citation_Agent]
<final bullet list from that agent>
=== MASTER_HANDOFF_END ===

4) After printing the handoff block, respond only with: TERMINATE

PAPER CONTENT:
{paper_content}
"""

def get_master_agent_prompt(paper_content: str) -> str:
    return f"""You are the **Master Agent**."""  

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

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df = pd.read_csv(INPUT_CSV) 

        # Filter the dataframe to keep rows where 'kg_triplets' is NOT 'SKIPPED_NO_TEXT'
        df = df[df['kg_triplets'] != 'SKIPPED_NO_TEXT']

        # Optional: Reset the index after dropping rows
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
    all_agent_names.append("Citation_Agent")
    agent_names_str = ", ".join(all_agent_names)

    for i in tqdm(range(0, len(df))):

        row = df.iloc[i]

        # 1. Main Paper Text
        paper_text = str(row.get("input_text_cleaned", ""))
        
        # --- NEW COLUMN INJECTION ---
        # Assuming you created 'combined_triplets' as requested previously
        kg_context_string = str(row.get("combined_triplets", "No additional KG context available.")) 
        print('kg context string',kg_context_string)

        # 2. CITATION DATA PREP (FROM gpt_ranked_chunks: top 3)
        raw_ranked_chunks = row.get("gpt_ranked_chunks", "[]")
        citation_chunks_str = ""

        try:
            if isinstance(raw_ranked_chunks, str):
                chunks_list = ast.literal_eval(raw_ranked_chunks)
            elif isinstance(raw_ranked_chunks, list):
                chunks_list = raw_ranked_chunks
            else:
                chunks_list = []

            top_3_chunks = chunks_list[:3] if isinstance(chunks_list, list) else [] 
            print('top 3 chunks',top_3_chunks)

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
            print('citation chunks str',citation_chunks_str)

        except Exception as e:
            citation_chunks_str = "No citation chunks available."

        # === TOKEN SAFEGUARDS ===
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=30000)
        citation_chunks_str = truncate_text_to_tokens(citation_chunks_str, max_tokens=6000) # 10,000
        
        # Inject the context and the requested sentence
        kg_data_injection = f"We are sending knowledge graph triplets from the paper and relevant papers.\n\n=== KG TRIPLETS ===\n{kg_context_string}"
        content_with_kg = f"{paper_text}\n\n{kg_data_injection}"

        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue

        combined_content = f"""
        === MAIN PAPER CONTENT ===
        {content_with_kg}

        === CITED PAPERS CONTEXT (Top Ranked) ===
        {citation_chunks_str}
        """

        # ==========================================
        # DYNAMIC AGENT CREATION
        # ==========================================

        agents_list = [user_proxy]

        leader_prompt = get_leader_agent_prompt(combined_content, agent_names=agent_names_str)
        leader_agent = autogen.AssistantAgent(
            name="Leader_Agent",
            system_message=leader_prompt,
            llm_config=llm_config
        )
        agents_list.append(leader_agent)

        citation_sys_msg = get_citation_agent_prompt(content_with_kg, citation_chunks_str)
        citation_agent = autogen.AssistantAgent(
            name="Citation_Agent",
            system_message=citation_sys_msg,
            llm_config=llm_config
        )
        agents_list.append(citation_agent)

        for agent_name_base, prompt_func in standard_specialist_config:
            sys_msg = prompt_func(content_with_kg)
            specialist = autogen.AssistantAgent(
                name=f"{agent_name_base}_Agent",
                system_message=sys_msg,
                llm_config=llm_config
            )
            agents_list.append(specialist)

        groupchat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=30, # 45 
            speaker_selection_method="auto"
        )

        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        task_msg = """Leader_Agent, please start the limitation analysis process for the provided paper content."""

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True
            )
            chat_hist = chat_result.chat_history
            handoff = extract_master_handoff(chat_hist)

            if not handoff:
                final_output = "NO_MASTER_HANDOFF_FROM_LEADER"
            else:
                final_output = run_master_outside(combined_content, handoff)

            df.at[df.index[i], "final_merged_limitations"] = final_output
            df.at[df.index[i], "full_chat_history"] = str(chat_hist)

        except Exception as e:
            df.at[df.index[i], "final_merged_limitations"] = f"ERROR: {e}"
            df.at[df.index[i], "full_chat_history"] = ""

        if i % 5 == 0:
            df.to_csv(OUTPUT_SLICE, index=False)
        time.sleep(3)

    df.to_csv(OUTPUT_SLICE, index=False)

if __name__ == "__main__":
    run_pipeline()