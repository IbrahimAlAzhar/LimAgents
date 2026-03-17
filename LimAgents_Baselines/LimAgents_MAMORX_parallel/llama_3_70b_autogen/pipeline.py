import os
import pandas as pd
import time
from tqdm import tqdm

# --- IMPORT AUTOGEN CORRECTLY ---
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Use the endpoint provided by the PBS script (or fallback to localhost)
base_url = os.environ.get("MODEL_ENDPOINT", "http://localhost:8000/v1")
print(f"🔗 Python Script Connecting to: {base_url}")

# Define the config list (Correct Format for AutoGen)
config_list = [
    {
        "model": "llama3-70b", # Matches the --served-model-name in PBS
        "base_url": base_url,
        "api_key": "EMPTY",
    }
]

# Create the LLM config dictionary
llm_config = {
    "config_list": config_list,
    "temperature": 0.3,
    "timeout": 600,
    "cache_seed": None 
}

# 2. Input/Output Paths

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

GLOBAL_CONTEXT_NOTE = """
[SYSTEM NOTE]: The FULL CONTENT of the scientific paper is provided in the chat history above. 
You do not need to use external tools to read the paper; simply analyze the text provided by the Leader.
"""

def get_agent_prompts():
    return {
        "Clarity": f"""You are the **Clarity Agent**. Scrutinize for clarity-related limitations.
- Identify missing details, ambiguities, or poor structure.
- Output Format: "- [Description]: Explanation; Section reference."
{GLOBAL_CONTEXT_NOTE}""",

        "Impact": f"""You are the **Impact Agent**. Focus on limitations in novelty and significance.
- Look for overstated claims or weak motivations.
- Output Format: "- [Description]: Explanation; Impact on field."
{GLOBAL_CONTEXT_NOTE}""",

        "Experiment": f"""You are the **Experiment Agent**. Evaluate experimental limitations.
- Look for flaws in design, missing ablations, or weak baselines.
- Output Format: "- [Description]: Why problematic; Suggestion for improvement."
{GLOBAL_CONTEXT_NOTE}""",

        "Master": f"""You are the **Master Agent**. Synthesize limitations into a cohesive list.
- Remove redundancy.
- Output MUST be a clean, Numbered List: "1. [Statement]: Justification [Sources]."
- NO conversational filler.
{GLOBAL_CONTEXT_NOTE}""",

        "Leader": f"""You are the **Leader Agent**. Coordinate the team.
1. Ask Clarity, Impact, Experiment agents to analyze.
2. If output is weak, ask for revision.
3. Once satisfied, ask Master Agent to merge.
4. Reply "TERMINATE" after the list is generated.
{GLOBAL_CONTEXT_NOTE}"""
    }

def create_swarm():
    prompts = get_agent_prompts()
    
    # Initialize Agents with the corrected classes
    leader = AssistantAgent(name="Leader_Agent", system_message=prompts["Leader"], llm_config=llm_config)
    master = AssistantAgent(name="Master_Agent", system_message=prompts["Master"], llm_config=llm_config)
    
    specialists = []
    for name in ["Clarity", "Impact", "Experiment"]:
        agent = AssistantAgent(
            name=f"{name}_Agent",
            system_message=prompts[name],
            llm_config=llm_config
        )
        specialists.append(agent)

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    all_agents = [user_proxy, leader, master] + specialists
    
    groupchat = GroupChat(
        agents=all_agents,
        messages=[],
        max_round=20, 
        speaker_selection_method="auto", 
        allow_repeat_speaker=True
    )
    
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    return user_proxy, manager

def run_pipeline():
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} rows.") 
        
        # --- ADJUST RANGE HERE ---
        START_INDEX = 100
        END_INDEX = 101 # Adjust as needed
        # -------------------------
        
        df = df.iloc[START_INDEX:END_INDEX]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    # Initialize Agents
    user_proxy, manager = create_swarm()

    print(f"Processing rows {START_INDEX} to {END_INDEX}...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        paper_text = str(row.get("input_text_cleaned", ""))
        
        if len(paper_text) < 100:
            df.at[i, "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            continue

        task_msg = f"""
        Here is the scientific paper content to analyze:
        === PAPER BEGIN ===
        {paper_text[:25000]} 
        === PAPER END ===
        Leader Agent, please coordinate your team to identify limitations.
        """
        # Note: Truncated text to 25k chars to ensure it fits in 4k/8k context limits if necessary

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            master_messages = []
            history_str = str(chat_result.chat_history)
            
            for msg in chat_result.chat_history:
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    if content != "TERMINATE":
                        master_messages.append(content)
            
            if master_messages:
                final_output = "\n\n".join(master_messages)
            else:
                final_output = "NO_OUTPUT_FROM_MASTER"
            
            df.at[i, "final_merged_limitations"] = final_output
            df.at[i, "full_chat_history"] = history_str
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            df.at[i, "final_merged_limitations"] = f"ERROR: {e}"

        if (i - START_INDEX) % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False) 

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Pipeline Complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()