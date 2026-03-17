# done as explaination in paper 

import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import os
import pandas as pd
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"

df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df.csv")

import ast
import pandas as pd

def safe_literal_eval(text):
    # If it's already a list, return it as is
    if isinstance(text, list):
        return text
        
    # Handle empty or NaN values
    if pd.isna(text) or text == "":
        return []

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Return an empty list if the string cannot be parsed
        return []

# Apply it to your specific column
df['prefixed_sorted_chunks_cited_in_openalex'] = df['prefixed_sorted_chunks_cited_in_openalex'].apply(safe_literal_eval) 

import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def get_gpt4_scores_batch(query, chunks_list_of_lists):
    # Safety checks
    if not chunks_list_of_lists or not query:
        return []

    # Prepare chunks
    # We take the top 10 and ensure we have a clean list of strings
    batch_chunks = [item[0] for item in chunks_list_of_lists[:10]]
    
    # Create a numbered list for the prompt so GPT knows which is which
    formatted_chunks = "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(batch_chunks)])
    
    # --- CORRECTED PROMPT ---
    # We removed the conflicting dictionary instructions and placeholders.
    # We ask strictly for a JSON list of numbers.
    system_prompt = (
        "You are an expert relevance evaluator. "
        "Your task is to evaluate the relevance of the provided text chunks to the user query. "
        "Relevance is based on semantic similarity, topical overlap, and how well the chunk supports the query. "
        "Output strictly a valid JSON object with a single key 'scores'. "
        "The value must be a LIST of floats between 0.0 and 1.0. "
        "The order of the scores must correspond exactly to the order of the chunks."
        "\n\nExample Output:\n"
        "{\"scores\": [0.95, 0.1, 0.45, 0.0, 0.8]}"
    )
    
    user_prompt = f"""
    QUERY: "{query}"

    CHUNKS TO EVALUATE:
    {formatted_chunks}

    Evaluate these {len(batch_chunks)} chunks. 
    Return the JSON object with the list of scores.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse result
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # Get scores (expecting a list)
        scores = result_json.get("scores", [])
        
        # Fallback: if it returned a dict of {chunk1: 0.5}, convert to list values
        if isinstance(scores, dict):
            scores = list(scores.values())

        # Validation: ensure scores is actually a list
        if not isinstance(scores, list):
             # Try to find a list elsewhere if the key wasn't 'scores'
             scores = next((v for v in result_json.values() if isinstance(v, list)), [])

        # Pad or truncate if length mismatch
        if len(scores) < len(batch_chunks):
            scores.extend([0.0] * (len(batch_chunks) - len(scores)))
            
        return scores[:len(batch_chunks)]

    except Exception as e:
        print(f"Error: {e}")
        return [0.0] * len(batch_chunks)

def process_row_ranking(row):
    query = row['input_text_cleaned']
    original_chunks = row['prefixed_sorted_chunks_cited_in_openalex']
    
    # Basic validation
    if not isinstance(original_chunks, list) or len(original_chunks) == 0:
        return []
    
    # Process Top 10
    top_10 = original_chunks[:10]
    scores = get_gpt4_scores_batch(query, top_10)
    
    scored_top_10 = []
    for chunk_wrapper, score in zip(top_10, scores):
        # Create structure: ['chunk_1: text...', 0.95]
        new_item = [chunk_wrapper[0], float(score)] # Ensure score is float
        scored_top_10.append(new_item)
        
    # Sort descending based on score
    scored_top_10.sort(key=lambda x: x[1], reverse=True)
    
    return scored_top_10

# Apply
tqdm.pandas()
df['gpt_ranked_chunks'] = df.progress_apply(process_row_ranking, axis=1)

df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df.csv",index=False) 
