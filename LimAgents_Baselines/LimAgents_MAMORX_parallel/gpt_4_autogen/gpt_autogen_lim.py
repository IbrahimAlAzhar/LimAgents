import os
import pandas as pd
import autogen
from typing import Dict, List
from tqdm import tqdm
import sys
import time 

# ==========================================
# 1. CONFIGURATION
# ==========================================

# 1. Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# 2. Model Configuration
MODEL_ID = "gpt-4o-mini"

llm_config = {
    "config_list": [
        {
            "model": MODEL_ID,
            "api_key": api_key,
        }
    ],
    "temperature": 0.3, 
    "timeout": 120, 
    "cache_seed": None 
}


# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

GLOBAL_CONTEXT_NOTE = """
[SYSTEM NOTE]: The FULL CONTENT of the scientific paper is provided in the chat history above. 
You do not need to use external tools to read the paper; simply analyze the text provided by the Leader.
"""

def get_agent_prompts():
    return {
        "Clarity": f"""You are the **Clarity Agent**. Scrutinize for clarity-related limitations: unclear methods, experimental settings, concepts, organization, or reproducibility issues.
- Identify missing details, ambiguities, or poor structure that hinder understanding or replication.
- Output Format: "- [Description]: Explanation (e.g., hinders reproducibility); Section reference."
{GLOBAL_CONTEXT_NOTE}""",

        "Impact": f"""You are the **Impact Agent**. Focus on limitations in novelty, significance, and impact.
- Look for overstated claims, unaddressed assumptions, weak motivations, or limited applicability.
- Scrutinize for hidden issues undermining contributions.
- Output Format: "- [Description]: Explanation (e.g., reduces significance); Impact on field."
{GLOBAL_CONTEXT_NOTE}""",

        "Experiment": f"""You are the **Experiment Agent**. Evaluate experimental limitations.
- Look for flaws in design, missing ablations, poor metrics, weak baselines, or reproducibility gaps.
- Suggest ideals and compare to the paper's actual approach.
- Output Format: "- [Description]: Why problematic; Suggestion for improvement."
{GLOBAL_CONTEXT_NOTE}""",

        "Master": f"""You are the **Master Agent**. You synthesize limitations from all other agents into a cohesive, non-redundant list.
- Prioritize critical limitations (e.g., validity impacts).
- Resolve overlaps by combining points from different agents.
- Your Output MUST be a clean, Numbered List: "1. [Statement]: Justification [Sources]."
- Do not include conversational filler. Just the list.
{GLOBAL_CONTEXT_NOTE}""",

        "Leader": f"""You are the **Leader Agent**. You coordinate the team.
Your Goal: Produce a high-quality list of limitations for the scientific paper provided by the User.
        
PROTOCOL:
1. **Instruct**: Ask the Agents (**Clarity, Impact, Experiment**) to analyze the paper.
2. **Evaluate**: When an agent replies, check their work. Is it specific? Is it grounded?
3. **Refine**: If an output is weak, ask that specific Agent to revise it.
4. **Finalize**: Once you have good outputs from the team, instruct the **Master Agent** to merge everything.
5. **Terminate**: Once the Master Agent provides the final list, reply with "TERMINATE".
{GLOBAL_CONTEXT_NOTE}"""
    }

# ==========================================
# 3. AGENT INITIALIZATION
# ==========================================

def create_swarm():
    prompts = get_agent_prompts()
    
    # Define Core Agents
    leader = autogen.AssistantAgent(name="Leader_Agent", system_message=prompts["Leader"], llm_config=llm_config)
    master = autogen.AssistantAgent(name="Master_Agent", system_message=prompts["Master"], llm_config=llm_config)
    
    specialists = []
    # Only creating the agents we actually use
    for name in ["Clarity", "Impact", "Experiment"]:
        agent = autogen.AssistantAgent(
            name=f"{name}_Agent",
            system_message=prompts[name],
            llm_config=llm_config
        )
        specialists.append(agent)

    # User Proxy (Triggers the chat)
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Add all agents to the group
    all_agents = [user_proxy, leader, master] + specialists
    
    groupchat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=20, 
        speaker_selection_method="auto", 
        allow_repeat_speaker=True
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    return user_proxy, manager

# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================

def run_pipeline():
    # Load Data
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} rows.") 
         # 2. Define Slice - UPDATED TO MATCH YOUR FILENAME (100-199)
        START_INDEX = 100
        END_INDEX = 199 
        
        # We slice for iteration, but we write back to 'df' using the index
        df = df.iloc[START_INDEX:END_INDEX]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 1. Initialize with "PENDING" to prevent NaN issues
    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"
    if "full_chat_history" not in df.columns:
        df["full_chat_history"] = "PENDING"

    # Initialize Agents (Done once here because context is refreshed via User Message)
    user_proxy, manager = create_swarm()

  

    print(f"Processing rows {START_INDEX} to {END_INDEX}...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        
        # Get Paper Content
        paper_text = str(row.get("input_text_cleaned", ""))
        cited_info = str(row.get("cited_in_ret", "")) 
        
        # 3. Explicitly mark skipped rows
        if len(paper_text) < 100:
            df.at[i, "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            df.at[i, "full_chat_history"] = "SKIPPED"
            continue

        # Construct Initial Message
        task_msg = f"""
        Here is the scientific paper content to analyze:
        
        === PAPER BEGIN ===
        {paper_text}
        === PAPER END ===

        Leader Agent, please coordinate your team (Clarity, Impact, Experiment) to identify limitations. 
        Ensure the Master Agent merges them into a final list.
        """

        try:
            # Start Chat (Clear history ensures agents don't see previous papers)
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            # Extract Results from Master Agent
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
            
            # Save to DataFrame
            df.at[i, "final_merged_limitations"] = final_output
            df.at[i, "full_chat_history"] = history_str
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            df.at[i, "final_merged_limitations"] = f"ERROR: {e}"

        # Save periodically
        if i % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False) 
        time.sleep(1)

    # Final Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Pipeline Complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()