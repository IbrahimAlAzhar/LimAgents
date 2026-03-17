import os
import re
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

import pandas as pd 

# df_limgen = df_limgen.head(2) 
# df_novgen = df_novgen.head(2) 
# --------------------------------------------------
# 1) OpenAI client
# -------------------------------------------------- 

os.environ['OPENAI_API_KEY'] = ''
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini" 

# api_key = os.environ.get("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key)

MODEL_ID = "gpt-4o-mini"
SAFE_INPUT_LIMIT = 48000

# --------------------------------------------------
# 2) Helper functions
# --------------------------------------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

def truncate_text(text, max_len=SAFE_INPUT_LIMIT):
    text = clean_text(text)
    return text[:max_len]

def build_merge_prompt(limitations_text, novelty_text):
    prompt = f"""
You are given two sets of paper limitations:
1. General limitations
2. Novelty-related limitations

Your task is to merge them into one final unified set of limitations for the paper.

Instructions:
- Combine overlapping or very similar limitations into one clear, non-redundant limitation.
- Preserve distinct limitations when they refer to different issues.
- Rewrite them so they sound like a coherent final "set of limitations" section for a research paper.
- Do not mention categories such as "general limitations" or "novelty limitations."
- Do not explain your reasoning.
- Do not add introductions, conclusions, headings, bullet labels, or any extra commentary.
- Output only the final merged set of limitations.
- Keep the writing clear, academic, concise, and natural.

General limitations:
{limitations_text}

Novelty-related limitations:
{novelty_text}
""".strip()
    return prompt

def merge_limitations_with_llm(limitations_text, novelty_text):
    limitations_text = truncate_text(limitations_text)
    novelty_text = truncate_text(novelty_text)

    # If both empty
    if not limitations_text and not novelty_text:
        return ""

    # If only one exists, return that one directly
    if limitations_text and not novelty_text:
        return limitations_text
    if novelty_text and not limitations_text:
        return novelty_text

    prompt = build_merge_prompt(limitations_text, novelty_text)

    response = client.chat.completions.create(
        model=MODEL_ID,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You merge similar research paper limitations into one final clean set of limitations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        timeout=120
    )

    return response.choices[0].message.content.strip()

# --------------------------------------------------
# 3) Main merging loop
# --------------------------------------------------
# Assumption:
# df_limgen contains column: 'final_merged_limitations'
# df_novgen contains column: 'novelty_report'
# and both dataframes are aligned row-wise.

if len(df_limgen) != len(df_novgen):
    raise ValueError("df_limgen and df_novgen must have the same number of rows for row-wise merging.")

# merged_outputs = []

for i in tqdm(range(len(df_novgen)), desc="Merging limitations"):
    lim_text = clean_text(df_limgen.loc[i, "final_merged_limitations"])
    nov_text = clean_text(df_novgen.loc[i, "novelty_report"])

    try:
        merged_text = merge_limitations_with_llm(lim_text, nov_text)
    except Exception as e:
        print(f"Error at row {i}: {e}")
        merged_text = ""

    # merged_outputs.append(merged_text) 
    df_novgen.loc[i, "final_set_of_limitations"] = merged_text

# Store into df_novgen
# df_novgen["final_set_of_limitations"] = merged_outputs