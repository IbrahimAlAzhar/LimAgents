import pandas as pd 
df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_with_sft/df.csv") 

# split train test using score from each agents output 
# splitts train and test data (take top 50 samples)
import pandas as pd
import re

df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_with_sft/df.csv")
# 1. Define the function to extract the score
def extract_total_sum(text):
    """
    Extracts the score from patterns like:
    - "**Total Sum: 51/60**"
    - "Total Sum: 49/60"
    - "Total Sum 53/60" (handles missing colon)
    """
    if not isinstance(text, str):
        return 0
    
    # Updated Regex Explanation:
    # Total\s+Sum  -> Matches "Total Sum" (case insensitive)
    # \s*[:]*\s* -> Matches optional spaces and optional colon (handles "Total Sum:" and "Total Sum 50")
    # (\d+)        -> Captures the digits
    match = re.search(r"Total\s+Sum\s*[:]*\s*(\d+)", text, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    return 0

# 2. List of your specific columns
# 2. DEFINE YOUR COLUMNS
agent_columns = [
    'Novelty_Significance_Agent_response', 
    'Citation_Agent_response', 
    'Theoretical_Methodological_Agent_response', 
    'Experimental_Evaluation_Agent_response', 
    'Generalization_Robustness_Efficiency_Agent_response', 
    'Clarity_Interpretability_Reproducibility_Agent_response', 
    'Data_Ethics_Agent_response'
]

# 3. Apply the extraction to these columns
# This creates a temporary DataFrame where all text is replaced by the extracted numbers
scores_only = df[agent_columns].applymap(extract_total_sum)

# 4. Create the new 'total score' column by summing across the row (axis=1)
df['total score'] = scores_only.sum(axis=1)

# 1. Sort the DataFrame by 'total_score' in descending order (High scores on top)
df_sorted = df.sort_values(by='total score', ascending=False)



import json

SYSTEM = "You are a helpful assistant. Follow the required output format exactly."

# --- pick 30 rows (option A: first 30) ---
df = df[["input_text_cleaned", "ground_truth_lim_peer"]].head(30).dropna()

# --- write JSONL ---
out_path = "train_30.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df_30.iterrows():
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": str(row["input_text_cleaned"])},
                {"role": "assistant", "content": str(row["ground_truth_lim_peer"])},
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved {len(df_30)} examples to {out_path}")

