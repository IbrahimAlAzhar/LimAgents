import os
import pandas as pd
import autogen
from tqdm import tqdm
import time
import sys

# 1. IMPORT YOUR PROMPTS
from prompts_config import get_agent_prompts

# 2. IMPORT DPO UTILS
from dpo_utils import RAGMemory, grade_solutions

# ==========================================
# 1. CONFIGURATION
# ==========================================
os.environ['OPENAI_API_KEY'] = 

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.4, 
    "max_tokens": 4000, 
    "cache_seed": None
}

INPUT_CSV = ""
OUTPUT_CSV = ""
DB_PATH = ""

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Initialize DPO Memory
rag = RAGMemory(db_path=DB_PATH)

# ==========================================
# 2. SWARM INITIALIZATION
# ==========================================

def create_swarm_for_row(paper_text, cited_info, gold_standard=""):
    """
    Creates agents using the new simplified prompts + DPO Injection.
    """
    
    # Load the base prompts dictionary
    base_prompts = get_agent_prompts()

    # --- INJECT DPO CONTEXT INTO LEADER ---
    leader_prompt = base_prompts["Leader"] + f"\n\n[DPO CONTEXT]:\n{gold_standard}"
    
    # --- INJECT DPO TASK INTO MASTER ---
    master_base = base_prompts["Master"]
    master_prompt = f"""{master_base}

    [DPO TASK OVERRIDE]:
    Ignore the instruction to produce a single list. You must generate **TWO DISTINCT** lists:
    1. **Solution A (Standard)**: A balanced, safe list.
    2. **Solution B (Critical)**: A rigorous, harsh list focusing on novelty/technical flaws.

    [GOLD STANDARD EXAMPLE]:
    {gold_standard}

    **REQUIRED OUTPUT FORMAT:**
    === SOLUTION A ===
    [List A]

    === SOLUTION B ===
    [List B]
    """

    # 1. Define Specialized Agents
    clarity = autogen.AssistantAgent("Clarity_Agent", system_message=base_prompts["Clarity"], llm_config=llm_config)
    impact = autogen.AssistantAgent("Impact_Agent", system_message=base_prompts["Impact"], llm_config=llm_config)
    experiment = autogen.AssistantAgent("Experiment_Agent", system_message=base_prompts["Experiment"], llm_config=llm_config)
    
    # 2. Define Orchestrators
    leader = autogen.AssistantAgent("Leader_Agent", system_message=leader_prompt, llm_config=llm_config)
    master = autogen.AssistantAgent("Master_Agent", system_message=master_prompt, llm_config=llm_config)

    # 3. User Proxy
    user_proxy = autogen.UserProxyAgent("User_Proxy", human_input_mode="NEVER", code_execution_config=False)

    # 4. Group Chat
    all_agents = [user_proxy, leader, master, clarity, impact, experiment]
    
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
# 3. MAIN PROCESSING LOOP
# ==========================================

def run_pipeline():
    try:
        df1 = pd.read_csv(INPUT_CSV) 
        # Adjust slice as needed
        df = df1.iloc[100:142].copy()
        print(f"Loaded {len(df)} rows to process.")
    except Exception as e:
        print(f"Error loading CSV: {e}"); return

    # Init Columns
    new_cols = ["winner_solution", "loser_solution", "winner_score", "full_chat_history", "final_generated_limitations"]
    for col in new_cols:
        if col not in df.columns: df[col] = ""

    df_slice = df 

    for i, row in tqdm(df_slice.iterrows(), total=len(df_slice)):
        
        # 1. Safety wrapper for the WHOLE ROW
        try:
            paper_text = str(row.get("input_text_cleaned", ""))
            cited_info = str(row.get("cited_in_ret", "")) 
            
            if len(paper_text) < 50: 
                print(f"Skipping row {i} (text too short)")
                continue

            # --- STEP 1: RETRIEVE GOLD STANDARD ---
            gold_example = rag.get_gold_standard(paper_text)
            
            # --- STEP 2: CREATE SWARM (Now protected by try/except) ---
            user_proxy, manager = create_swarm_for_row(paper_text, cited_info, gold_example)

            # --- INTEGRATED CITED INFO HERE ---
            task_msg = f"""
            Here is the scientific paper content to analyze:
            
            === PAPER BEGIN ===
            {paper_text}
            === PAPER END ===
            
            === CITED PAPERS CONTEXT ===
            {cited_info}
            === END CONTEXT ===

            Leader Agent, coordinate your team (Clarity, Impact, Experiment).
            Master Agent, remember to generate TWO lists (Solution A and Solution B) as per your DPO instructions.
            """

            # Run Chat
            chat_result = user_proxy.initiate_chat(manager, message=task_msg, clear_history=True)

            # --- STEP 3: ROBUST EXTRACTION ---
            final_content = ""
            for msg in reversed(chat_result.chat_history):
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    if len(content) > 20: 
                        final_content = content
                        break
            
            # --- STEP 4: PARSE OUTPUT ---
            sol_a = ""
            sol_b = ""
            
            if "=== SOLUTION A ===" in final_content and "=== SOLUTION B ===" in final_content:
                parts = final_content.split("=== SOLUTION B ===")
                sol_a = parts[0].replace("=== SOLUTION A ===", "").strip()
                sol_b = parts[1].strip()
                dpo_status = "SUCCESS"
            else:
                sol_a = final_content
                sol_b = "MISSING_DPO_FORMAT"
                dpo_status = "FALLBACK"

            # --- STEP 5: CONDITIONAL GRADING ---
            if dpo_status == "SUCCESS":
                print(f"  [DPO] Grading Row {i}...")
                scores = grade_solutions(paper_text, sol_a, sol_b)
                winner = sol_a if scores.get("Winner") == "A" else sol_b
                loser = sol_b if scores.get("Winner") == "A" else sol_a
                win_score = scores.get("A_total", 0) if scores.get("Winner") == "A" else scores.get("B_total", 0)
                
                rag.add_winner(paper_text, winner, win_score)
                df.at[i, "winner_solution"] = winner
                df.at[i, "loser_solution"] = loser
                df.at[i, "winner_score"] = win_score
            else:
                # If fallback, save what we have
                df.at[i, "winner_solution"] = sol_a
                df.at[i, "winner_score"] = "N/A (Fallback)"

            # Save results to memory
            df.at[i, "final_generated_limitations"] = sol_a 
            df.at[i, "full_chat_history"] = str(chat_result.chat_history)
            
        except Exception as e:
            # This catches crashes in Agent Creation, Chat, or Grading
            print(f"Error processing row {i}: {e}")
            df.at[i, "final_generated_limitations"] = f"Error: {e}"

        # --- SAFE SAVE ---
        # We put saving in a try block so it doesn't kill the loop if file is open
        if i % 5 == 0:
            try:
                df.to_csv(OUTPUT_CSV, index=False)
            except Exception as e:
                print(f"Warning: Could not save CSV at row {i} (File might be open): {e}")
                
        time.sleep(2) 

    # Final Save
    try:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Pipeline Complete. Saved to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Final save failed: {e}")

if __name__ == "__main__":
    run_pipeline()