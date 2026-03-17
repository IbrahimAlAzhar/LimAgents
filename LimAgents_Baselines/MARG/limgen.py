import os
import pandas as pd
import autogen
import sys
import time
import ast
import re
from tqdm import tqdm
import tiktoken

# ==========================================
# 1. CONFIGURATION
# ==========================================

# NOTE: Ensure you replace this with your actual key or secure method
os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"
SAFE_INPUT_LIMIT = 48000 

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2, 
    "timeout": 120, 
    "cache_seed": None 
}

# Paths
INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/data/final_balanced_kde/df_updated_with_retrieval.csv"
OUTPUT_SLICE = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/MARG/df_autogen_marg_agents_100_199.csv"

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. MARG PROMPT DEFINITIONS
# ==========================================

def get_marg_impact_agent_prompt(paper_content: str, citation_content: str) -> str:
    """
    Adapted from MARG 'Impact' Expert Prompt.
    Focuses on: Novelty, Significance, Hidden Assumptions, Scope.
    """
    return f"""You are the **Impact & Novelty Expert** (MARG-Impact).
ROLE: You are highly skeptical of the paper's claims. Your goal is to identify limitations regarding the **significance, novelty, and hidden assumptions** of the work.

STRATEGY:
1. **Analyze Motivation:** Does the paper clearly justify its goals? Are the motivating problems real or contrived?
2. **Check Assumptions:** Identify "hidden assumptions" (e.g., assuming a robot is omnidirectional, assuming clean data). If the paper fails to justify these, it is a significant limitation.
3. **Verify Novelty:** Compare the paper's contribution against the provided 'CITED PAPERS INFO'. Does it merely rebrand existing work? Is the "gap" in literature real?
4. **Scope:** Does the method only work in narrow settings?

OUTPUT INSTRUCTIONS:
- Provide a bullet list of **Implicit Limitations** related to scope, novelty, and assumptions.
- Be specific. Do not say "The scope is limited." Say "The method is limited because it assumes X, which is rare in real-world settings."

=== MAIN PAPER CONTENT ===
{paper_content}

=== CITED PAPERS INFO ===
{citation_content}"""

def get_marg_experiments_agent_prompt(paper_content: str) -> str:
    """
    Adapted from MARG 'Experiments' Expert Prompt.
    Focuses on: Methodological Flaws, Missing Baselines, Ablation Studies.
    """
    return f"""You are the **Methodology & Experiments Expert** (MARG-Experiments).
ROLE: You are an expert scientist who designs high-quality evaluations. Your goal is to identify **methodological flaws and missing experiments**.

STRATEGY:
1. **Hypothesize Ideal Experiments:** Before judging, imagine what experiments *should* be run to rigorously prove the paper's claims (e.g., specific baselines, ablations, statistical tests).
2. **Gap Analysis:** Compare your "Ideal Experiments" to the "Actual Experiments" in the text.
   - Missing Baselines?
   - Missing Ablation Studies (component analysis)?
   - Weak Metrics?
3. **Verify Support:** Do the results actually support the strong claims made? Look for over-claiming based on weak evidence.

OUTPUT INSTRUCTIONS:
- List **Methodological Limitations**.
- Example: "The study is limited by the lack of an ablation study on component X, making it impossible to attribute the performance gains."

PAPER CONTENT:
{paper_content}"""

def get_marg_clarity_agent_prompt(paper_content: str) -> str:
    """
    Adapted from MARG 'Clarity' Expert Prompt.
    Focuses on: Reproducibility, Hyperparameters, Vague Definitions.
    """
    return f"""You are the **Clarity & Reproducibility Expert** (MARG-Clarity).
ROLE: You have extreme attention to detail. Your goal is to ensure the paper is reproducible and unambiguous.

STRATEGY:
1. **Reproducibility Check:** Look for missing implementation details: hyperparameters, seed numbers, hardware specs, or data filtering steps.
2. **Concept Definitions:** Identify vague terms or "black box" explanations. If a term is used without definition, it is a clarity limitation.
3. **Inconsistencies:** Check for contradictions in the text (e.g., Figure 1 shows X, but text says Y).

OUTPUT INSTRUCTIONS:
- List **Reproducibility & Clarity Limitations**.
- Example: "The method is not reproducible because the authors fail to specify the hyperparameters for the baseline models."

PAPER CONTENT:
{paper_content}"""

def get_marg_leader_agent_prompt(paper_content: str, agent_names: str) -> str:
    """
    Adapted from MARG Leader Agent.
    Orchestrates the specific experts.
    """
    return f"""You are the **Leader Agent**. You are coordinating a MARG (Multi-Agent Review Generation) team.
Available Experts: {agent_names}

PROTOCOL:
1. **Plan:** Instruct your experts to analyze the paper using their specific strategies (Impact, Experiments, Clarity).
2. **Delegate:** - Ask **Impact_Agent** to check for hidden assumptions and novelty gaps.
   - Ask **Experiments_Agent** to check for missing baselines and ablation studies.
   - Ask **Clarity_Agent** to check for reproducibility issues (hyperparameters, code details).
3. **Synthesize:** As experts respond, ensure their limitations are **specific** and **actionable**. If an agent gives a generic response (e.g., "Add more details"), ask them to specify *which* details.
4. **Handoff:** Once you have gathered strong critiques from all experts, instruct the **Master_Agent** to perform the final "Refinement" step.
   - Say: "Master_Agent, here are the raw limitations from the team. Please refine and categorize them."

PAPER CONTENT:
{paper_content}"""

def get_marg_master_refinement_prompt(paper_content: str) -> str:
    """
    Adapted from MARG Refinement Prompt.
    Filters and Categorizes limitations.
    """
    return f"""You are the **Master Agent** (Refinement Stage).
ROLE: Your job is to take the raw limitations identified by the team and **refine** them into a final, high-quality list.

TASK:
1. **Prune Invalid Comments:** Remove any limitation that is factually incorrect (e.g., if the team says "Missing Baseline X" but Baseline X is actually in the Appendix, remove it) or trivial (e.g., grammar issues).
2. **Categorize:** Group the limitations into two categories:
   - **Explicit Limitations:** Weaknesses explicitly admitted by the authors (e.g., in the Discussion/Conclusion).
   - **Implicit/Methodological Limitations:** Flaws identified by the agents that the authors did not admit (e.g., missing comparisons, hidden assumptions).
3. **Sharpen Specificity:** Ensure every bullet point is detailed.
   - Bad: "The evaluation is weak."
   - Good: "The evaluation is limited by the exclusion of standard datasets (e.g., ImageNet), testing only on synthetic data."

OUTPUT FORMAT:
Start with: "Here is the consolidated, refined list of research limitations:"
[Bulleted list categorized by Explicit vs Implicit]
Then end with: "TERMINATE"

PAPER CONTENT:
{paper_content}"""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def clean_text_detailed(text):
    if pd.isna(text) or text is None: return ""
    text = str(text).replace('\n', ' ')
    text = re.sub(r'\S+\s+et\s+al\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_intro_and_abstract(cited_entry):
    if pd.isna(cited_entry): return ""
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except: return ""
    if not isinstance(parsed, dict): return ""

    processed = []
    for idx, (pid, data) in enumerate(parsed.items(), 1):
        if not isinstance(data, dict): continue
        # Intro scan
        intro = ""
        for sec in data.get("sections", []):
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)

        if t_clean or a_clean or i_clean:
            processed.append(f"'Paper{idx}_Title: {t_clean}', 'Paper{idx}_Abstract': '{a_clean}', 'Paper{idx}_Introduction': '{i_clean}'.")
    return "\n".join(processed)

def truncate_text_to_tokens(text: str, max_tokens: int = SAFE_INPUT_LIMIT) -> str:
    if not text:
        return ""
    try:
        encoding = tiktoken.get_encoding("o200k_base")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    print(f"  ⚠️ Truncating input: {len(tokens)} tokens -> {max_tokens} tokens.")
    return encoding.decode(tokens[:max_tokens]) + "... [TRUNCATED]"

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df1 = pd.read_csv(INPUT_CSV) 
        # Adjust the slice as needed (e.g., 140:199)
        df = df1.iloc[100:200].copy() 
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

    # Agent names for the Leader to know
    # We define the core MARG agents + Master
    all_agent_names = ["Impact_Agent", "Experiments_Agent", "Clarity_Agent", "Master_Agent"]
    agent_names_str = ", ".join(all_agent_names)

    # Configuration for standard specialists (excluding Impact which needs citations)
    standard_specialist_config = [
        ("Experiments", get_marg_experiments_agent_prompt),
        ("Clarity", get_marg_clarity_agent_prompt)
    ]

    for i in tqdm(range(0, len(df))):
        row = df.iloc[i]
        
        # Data Prep
        paper_text = str(row.get("input_text_cleaned", "")) 
        citation_text = extract_intro_and_abstract(row.get("cited_in", ""))
        
        # Token Safeguards
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=40000) 
        citation_text = truncate_text_to_tokens(citation_text, max_tokens=10000)

        # Skip short text
        if len(paper_text) < 100:
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = "SKIPPED_SHORT_TEXT"
            continue
            
        # Context for Leader/Master
        combined_content = f"""=== MAIN PAPER CONTENT ===\n{paper_text}\n\n=== CITED PAPERS CONTEXT ===\n{citation_text}"""

        # ==========================================
        # DYNAMIC AGENT CREATION (MARG TEAM)
        # ==========================================
        
        agents_list = [user_proxy]

        # 1. Leader Agent (Needs Combined Content)
        leader_prompt = get_marg_leader_agent_prompt(combined_content, agent_names=agent_names_str)
        leader_agent = autogen.AssistantAgent(
            name="Leader_Agent",
            system_message=leader_prompt,
            llm_config=llm_config
        )
        agents_list.append(leader_agent)

        # 2. Master Agent (Needs Combined Content for Refinement)
        master_prompt = get_marg_master_refinement_prompt(combined_content)
        master_agent = autogen.AssistantAgent(
            name="Master_Agent",
            system_message=master_prompt,
            llm_config=llm_config
        )
        agents_list.append(master_agent)

        # 3. Impact Agent (Needs Paper + Citation Content)
        impact_prompt = get_marg_impact_agent_prompt(paper_text, citation_text)
        impact_agent = autogen.AssistantAgent(
            name="Impact_Agent",
            system_message=impact_prompt,
            llm_config=llm_config
        )
        agents_list.append(impact_agent)

        # 4. Standard Specialists (Experiments, Clarity) - Need Paper Content Only
        for agent_name_base, prompt_func in standard_specialist_config:
            sys_msg = prompt_func(paper_text)
            
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
            max_round=35, # MARG is concise, but give enough rounds for back-and-forth
            speaker_selection_method="auto" 
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # 6. Initiate
        task_msg = "Leader_Agent, please coordinate the MARG team to generate a refined list of research limitations."

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            # Capture Output from Master Agent
            master_messages = []
            for msg in chat_result.chat_history:
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    clean_content = content.replace("TERMINATE", "").strip()
                    
                    if len(clean_content) > 50:
                        master_messages.append(clean_content)
            
            if master_messages:
                final_output = master_messages[-1] 
            else:
                final_output = "NO_OUTPUT_FROM_MASTER"

            col_idx_lim = df.columns.get_loc("final_merged_limitations")
            col_idx_hist = df.columns.get_loc("full_chat_history")
            
            df.iat[i, col_idx_lim] = final_output
            df.iat[i, col_idx_hist] = str(chat_result.chat_history)
            
        except Exception as e:
            print(f"Error on row {i}: {e}")
            df.iat[i, df.columns.get_loc("final_merged_limitations")] = f"ERROR: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_SLICE, index=False)
        time.sleep(3)

    df.to_csv(OUTPUT_SLICE, index=False)

if __name__ == "__main__":
    run_pipeline()