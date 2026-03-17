import pandas as pd 
df = pd.read_csv("df.csv") 

import re
import pandas as pd

def parse_merged_limitations(text_str):
    if not isinstance(text_str, str) or not text_str.strip():
        return []

    # Split into lines and remove empty ones
    lines = [ln.strip() for ln in text_str.split("\n") if ln.strip()]

    limitations = []
    current_category = "General"
    lim_id = 0

    # 1. Category Pattern: Look for text wrapped in ** ** # This allows for leading bullets like '* **Category**' or just '**Category**'
    cat_pat = re.compile(r'^[\*\-•]?\s*\*\*(.*?)\*\*')

    # 2. Limitation Pattern: Matches lines starting with -, *, +, or •
    # We use [-\*\+•] to catch all the variations in your data
    lim_pat = re.compile(r'^[\-\*\+•]\s+(.+)$')

    # 3. Noise Filter: Common LLM "chatter" phrases
    noise_phrases = ["Master Agent", "Master_Agent", "Consolidated list", "identified in the paper", "Consolidated Limitations"]

    for ln in lines:
        # Skip repetitive headers/noise
        if any(phrase.lower() in ln.lower() for phrase in noise_phrases):
            continue

        # Step A: Check for Category Header FIRST
        mcat = cat_pat.match(ln)
        if mcat:
            new_cat = mcat.group(1).strip()
            # Safety check: ensure the bold text isn't just a bolded list item
            if not any(symbol in ln[:2] for symbol in ['-', '+', '•']) or ":" in ln:
                current_category = new_cat
                continue

        # Step B: Check for Limitation Bullet
        mlim = lim_pat.match(ln)
        if mlim:
            body = mlim.group(1).strip()
            
            # Format: Limitation text [Category Name]
            formatted_text = f"{body} [{current_category}]"

            limitations.append({
                "llm_id": lim_id,
                "llm_limitation": formatted_text
            })
            lim_id += 1

    return limitations

# Apply to your dataframe
# Swapping to 'merged_final_output' as per your second example
df['mistral_limitations_list'] = df['merged_final_output'].apply(parse_merged_limitations) 

import os
import re
import ast
import pandas as pd
from tqdm import tqdm
from openai import OpenAI  

# Load data and restrict to first 100 rows
# df = pd.read_csv(INPUT_CSV)
# df = df.iloc[100:200].copy().reset_index(drop=True)

# =========================
# 2. Parsing helper functions
# ========================= 

# extracting category, title, and limitation 
# def parse_gt_limitations(text_str):
#     """
#     Parse 'final_lim_gt_author_peer_cat_maj_hum_cleaned' which is a stringified list of dicts.
#     Extract 'category', 'title', and 'limitation' fields and combine them in format:
#     'category: title: limitation'
#     Attach gt_id.
#     Returns list of dicts: [{'gt_id': 0, 'gt_limitation': 'category: title: limitation'}, ...]
#     """
#     try:
#         data = ast.literal_eval(text_str)
#         results = []
        
#         for i, item in enumerate(data):
#             if not isinstance(item, dict):
#                 continue
                
#             # Get all three fields
#             category = item.get('category', '')
#             title = item.get('title', '')
#             limitation = item.get('limitation', '')
            
#             # Check if we have a limitation
#             if limitation is None or not str(limitation).strip():
#                 continue
                
#             # Combine in the desired format
#             combined_text = f"{category}: {title}: {limitation}".strip()
            
#             # Remove any trailing colons or extra spaces that might occur if fields are empty
#             combined_text = re.sub(r'^\s*:\s*', '', combined_text)  # Remove leading colon
#             combined_text = re.sub(r':\s*:\s*', ': ', combined_text)  # Fix double colons
            
#             results.append({
#                 'gt_id': i,
#                 'gt_limitation': combined_text
#             })
        
#         return results
        
#     except (ValueError, SyntaxError, TypeError):
#         return []

# extracting only limitation
# def parse_gt_limitations(text_str):
#     """
#     Parse 'final_lim_gt_author_peer_cat_maj_hum_cleaned' which is a stringified list of dicts.
#     Extract 'limitation' field and attach gt_id.
#     Returns list of dicts: [{'gt_id': 0, 'gt_limitation': '...'}, ...]
#     """
#     try:
#         data = ast.literal_eval(text_str)
#         texts = [item.get('limitation') for item in data if isinstance(item, dict)]
#         return [
#             {'gt_id': i, 'gt_limitation': t}
#             for i, t in enumerate(texts)
#             if t is not None and str(t).strip()
#         ]
#     except (ValueError, SyntaxError, TypeError):
#         return []

# extract limitations with category of ground truth 
import ast

def parse_gt_limitations(text_str):
    """
    Parse 'final_lim_gt_author_peer_cat_maj_hum_cleaned' which is a stringified list of dicts.
    Extract 'limitation' field, attach 'category' as a suffix, and attach gt_id.
    Returns list of dicts: [{'gt_id': 0, 'gt_limitation': 'Limitation text (- **Category**)'}, ...]
    """
    try:
        # Safely evaluate the string literal to a python object
        data = ast.literal_eval(text_str)
        
        results = []
        
        # Ensure the parsed data is actually a list
        if not isinstance(data, list):
            return []

        for i, item in enumerate(data):
            if isinstance(item, dict):
                lim_text = item.get('limitation')
                cat_text = item.get('category')
                
                # Only process if limitation text exists
                if lim_text and str(lim_text).strip():
                    cleaned_lim = str(lim_text).strip()
                    
                    # Append category suffix if available
                    if cat_text and str(cat_text).strip():
                        final_text = f"{cleaned_lim} (- **{str(cat_text).strip()}**)"
                    else:
                        final_text = cleaned_lim
                        
                    results.append({'gt_id': i, 'gt_limitation': final_text})
                    
        return results
        
    except (ValueError, SyntaxError, TypeError):
        return []

# parsing without first item 
# def parse_merged_limitations(text_str):
#     """
#     Parse 'mistral_master_0.4' text into a list of numbered limitation blocks.
#     Splits on '\n1', '\n2', '\n3', ... (with optional '.' and whitespace),
#     and drops the header before the first number.
#     Returns list of dicts: [{'llm_id': 0, 'llm_limitation': '...'}, ...]
#     """
#     if not isinstance(text_str, str):
#         return []
    
#     pattern = r'\n\d+\.?\s*'   # matches '\n1', '\n2.', '\n10   ', etc.
#     parts = re.split(pattern, text_str)

#     # parts[0] is usually header (e.g., "Final Limitations Analysis:\n\n")
#     if len(parts) > 1:
#         limitations = [p.strip() for p in parts[1:] if p.strip()]
#         return [
#             {'llm_id': j, 'llm_limitation': txt}
#             for j, txt in enumerate(limitations)
#         ]
#     else:
#         return [] 

# parsing with all items 
# def parse_merged_limitations(text_str):
#     """
#     Parse 'mistral_master_0.4' text into a list of numbered limitation blocks.
#     Splits on '\n1', '\n2', '\n3', ... (with optional '.' and whitespace),
#     but also captures the first limitation if it starts with '1.' or similar.
#     Returns list of dicts: [{'llm_id': 0, 'llm_limitation': '...'}, ...]
#     """
#     if not isinstance(text_str, str):
#         return []
    
#     # Trim whitespace first
#     text = text_str.strip()
#     if not text:
#         return []
    
#     # Pattern to match numbered items (e.g., "1.", "2)", "10: ", etc.)
#     # This looks for numbers at the start of the string or after newlines
#     pattern = r'(?:\A|\n)(\d+)[\.\):\s]+'
    
#     # Find all matches and their positions
#     matches = list(re.finditer(pattern, text))
    
#     if not matches:
#         # If no numbered pattern found, treat entire text as single limitation
#         return [{'llm_id': 0, 'llm_limitation': text}]
    
#     # Extract limitations based on match positions
#     limitations = []
#     for i, match in enumerate(matches):
#         start_pos = match.end()  # Start after the number and punctuation
#         if i < len(matches) - 1:
#             # Text from current match to next match
#             end_pos = matches[i + 1].start()
#             limitation_text = text[start_pos:end_pos].strip()
#         else:
#             # Last match - take text until end
#             limitation_text = text[start_pos:].strip()
        
#         if limitation_text:
#             limitations.append({
#                 'llm_id': int(match.group(1)) - 1,  # Convert to 0-based index
#                 'llm_limitation': limitation_text
#             })
    
#     return limitations

# parse limitations from master outside approach 


# def parse_merged_limitations(text_str):
#     """
#     Parse the master consolidated output formatted like:

#     - **Category:** 
#       - Limitation: Description
#       - Limitation2: Description

#     Returns:
#         List[Dict] with fields:
#           - llm_id
#           - llm_limitation  (formatted with category suffix)
#     """
#     if not isinstance(text_str, str) or not text_str.strip():
#         return []

#     lines = [ln.rstrip() for ln in text_str.strip().split("\n")]

#     limitations = []
#     current_category = "General"
#     lim_id = 0

#     # Regex patterns
#     # Category header: - **Something:**  (with optional spaces)
#     cat_pat = re.compile(r'^\s*-\s*\*\*(.+?)\*\*\s*:?\s*$')

#     # Limitation bullet: "  - Title: body" (indent optional)
#     # We require "- " bullet (with optional indent), then capture title and body
#     lim_pat = re.compile(r'^\s*-\s*(.+?)\s*:\s*(.+)\s*$')

#     for ln in lines:
#         if not ln.strip():
#             continue

#         # 1) Category line
#         mcat = cat_pat.match(ln)
#         if mcat:
#             current_category = mcat.group(1).strip()
#             continue

#         # 2) Limitation line (usually indented: "  - X: Y")
#         # To avoid confusing category lines, we only parse limitation bullets that are NOT bold categories
#         if "**" in ln:
#             # likely category, or some other bold text; skip
#             continue

#         mlim = lim_pat.match(ln)
#         if mlim:
#             title = mlim.group(1).strip()
#             body = mlim.group(2).strip()

#             suffix = f" (- **{current_category}**)"
#             full_limitation = f"**{title}**: {body}{suffix}"

#             limitations.append({
#                 "llm_id": lim_id,
#                 "llm_limitation": full_limitation
#             })
#             lim_id += 1

#     return limitations


# import re
# # parsing from 7 agents generated limitations (master inside approach)
# def parse_merged_limitations(text_str):
#     """
#     Parses the consolidated limitation text which uses a specific markdown format:
#     '- **Category Name:**' followed by '- **Limitation Name**: Description'.
    
#     Returns a list of dicts with the category appended as a suffix.
#     """
#     if not isinstance(text_str, str) or not text_str:
#         return []
    
#     text = text_str.strip()
    
#     # Split the text by the bullet point and bold start marker "- **"
#     # This captures both Category headers (e.g., "Novelty & Significance:**")
#     # and Limitation headers (e.g., "Incremental Contribution**: ...")
#     chunks = re.split(r'- \*\*', text)
    
#     limitations = []
#     current_category = "General" # Default category
#     lim_id = 0
    
#     # Skip the first chunk (introductory text)
#     for chunk in chunks[1:]:
#         chunk = chunk.strip()
#         if not chunk:
#             continue
            
#         # We expect the chunk to look like: "Title**: Content..."
#         if '**' in chunk:
#             # Split on the closing bold marker
#             header, body = chunk.split('**', 1)
            
#             header = header.strip() # e.g., "Novelty & Significance:" or "Incremental Contribution"
#             body = body.strip()     # e.g., "" (for category) or ": The paper presents..."
            
#             # --- DETECT IF CATEGORY OR LIMITATION ---
#             # If the body is empty (or just a colon), it is a Category Header.
#             # If the body contains text, it is a Limitation.
            
#             if not body or body == ":":
#                 # It is a Category Header
#                 current_category = header
#             else:
#                 # It is a Limitation
                
#                 # 1. Clean the body (remove leading colon if present)
#                 if body.startswith(':'):
#                     body = body[1:].strip()
                
#                 # 2. Construct the suffix using the current category
#                 # Matches format: (- **Novelty & Significance:**)
#                 suffix = f" (- **{current_category}**)"
                
#                 # 3. Format the final string
#                 full_limitation = f"**{header}**: {body}{suffix}"
                
#                 limitations.append({
#                     'llm_id': lim_id,
#                     'llm_limitation': full_limitation
#                 })
#                 lim_id += 1
                
#     return limitations


# =========================
# 3. Apply parsing to dataframe
# =========================

df['gt_limitations_list'] = df['final_lim_gt_author_peer_cat_maj_hum_cleaned'].apply(parse_gt_limitations)

# =========================
# 4. Create all GT × LLM pairs with IDs
# =========================

def build_pairs(row):
    pairs = []
    gt_list = row['gt_limitations_list']
    llm_list = row['mistral_limitations_list']

    for gt in gt_list:
        for llm in llm_list:
            pairs.append({
                'gt_id': gt['gt_id'],
                'gt_limitation': gt['gt_limitation'],
                'llm_id': llm['llm_id'],
                'llm_limitation': llm['llm_limitation'],
            })
    return pairs

df['paired_limitations'] = df.apply(build_pairs, axis=1) 

# Filter for rows where the list length is greater than 0
# reset_index(drop=True) ensures your index stays continuous for further processing
df = df[df['mistral_limitations_list'].map(len) > 0].reset_index(drop=True)

# df = df.head(1)

# Print the new count to verify
print(f"Cleaning complete. Current number of rows: {len(df)}") 

# key from sec gmail
os.environ['OPENAI_API_KEY'] = ''

# key from original gmail 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_ID = "gpt-4o-mini" 

# Make sure your key is set in the environment securely.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =========================
# 6. LLM evaluation per pair
# =========================

def evaluate_pairs_with_llm(pairs_list):
    """
    Takes a list of pair dicts:
      {'gt_id', 'gt_limitation', 'llm_id', 'llm_limitation'}
    Queries the LLM for each and returns a list like:
      [
        ['Pair 1: Yes', 'gt_id:0', 'gt_limitation: ...', 'llm_id:0', 'llm_limitation: ...'],
        ...
      ]
    """
    results = []
    
    if not isinstance(pairs_list, list):
        return []

    for i, pair in enumerate(pairs_list):
        gt_text = pair['gt_limitation']
        llm_text = pair['llm_limitation']

        description1 = f"ground truth limitations: {gt_text}"
        description2 = f"llm generated limitations: {llm_text}"
        
        prompt = (
            "Check whether 'list2' contains a topic or limitation from 'list1' "
            "or 'list1' contains a topic or limitation from 'list2'.\n\n"
            "Your answer should be \"Yes\" or \"No\".\n"
            f"List 1: {description1}\n"
            f"List 2: {description2}\n"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=False
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error: {str(e)}"

        result_entry = [
            f"Pair {i+1}: {answer}",
            f"gt_id:{pair['gt_id']}",
            f"gt_limitation:{gt_text}",
            f"llm_id:{pair['llm_id']}",
            f"llm_limitation:{llm_text}",
        ]
        results.append(result_entry)
        
    return results

# =========================
# 7. Apply LLM evaluation row by row
# =========================

df['llm_evaluation_results'] = None

print(f"Starting API Evaluation on {len(df)} rows...")

for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing Rows")):
    pairs = row['paired_limitations']
    row_results = evaluate_pairs_with_llm(pairs)
    df.at[index, 'llm_evaluation_results'] = row_results
    print("i is",i) 


import pandas as pd
import ast

# # ==========================================
# # 1. Configuration
# # ==========================================
# io_csv = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/SFT_from_zs_output/mistral/output/df_gpt_eval.csv"
col_eval = "llm_evaluation_results"

# print(f"Loading CSV: {io_csv} ...")
# df = pd.read_csv(io_csv)

# ==========================================
# 2. Convert 'llm_evaluation_results' from str to list using ast
# ==========================================
def parse_eval_list(val):
    """
    Convert a string representation of a Python list into a real list
    using ast.literal_eval. If already a list, return as-is.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        return ast.literal_eval(val)
    except (SyntaxError, ValueError, TypeError):
        return []

print(f"Parsing column '{col_eval}' with ast.literal_eval ...")
df[col_eval] = df[col_eval].apply(parse_eval_list)

# Quick sanity check on one row
print("\nExample parsed row 0 (first 2 items):")
print(df.loc[df.index[0], col_eval][:2])

# ==========================================
# 3. Compute recall, precision, F1 per row
# ==========================================
def compute_pair_metrics(row):
    """
    From llm_evaluation_results (list-of-lists), compute:
      - n_unique_gt
      - n_unique_llm
      - recall: (# gt_id with at least one Yes) / (total unique gt_id)
      - precision: (# llm_id with at least one Yes) / (total unique llm_id)
      - f1: harmonic mean of precision and recall
    Each item in llm_evaluation_results is:
      ['Pair 1: Yes/No', 'gt_id:0', 'gt_limitation:...', 'llm_id:0', 'llm_limitation:...']
    """
    items = row[col_eval]
    if not isinstance(items, list) or len(items) == 0:
        return pd.Series({
            "n_unique_gt": 0,
            "n_unique_llm": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0
        })
    
    all_gt_ids = set()
    all_llm_ids = set()
    yes_gt_ids = set()
    yes_llm_ids = set()
    
    for item in items:
        # Expect list like ['Pair 1: No', 'gt_id:0', 'gt_limitation:...', 'llm_id:0', 'llm_limitation:...']
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        
        # 1) Answer (Yes/No) from first element
        answer_str = str(item[0])
        is_yes = "YES" in answer_str.upper()  # robust yes-check
        
        # 2) Extract gt_id and llm_id from strings
        gid = None
        lid = None
        
        for elem in item:
            if isinstance(elem, str):
                if elem.startswith("gt_id"):
                    try:
                        gid = int(elem.split(":", 1)[1])
                    except Exception:
                        pass
                elif elem.startswith("llm_id"):
                    try:
                        lid = int(elem.split(":", 1)[1])
                    except Exception:
                        pass
        
        if gid is None or lid is None:
            continue
        
        all_gt_ids.add(gid)
        all_llm_ids.add(lid)
        
        if is_yes:
            yes_gt_ids.add(gid)
            yes_llm_ids.add(lid)
    
    n_unique_gt = len(all_gt_ids)
    n_unique_llm = len(all_llm_ids)
    
    recall = (len(yes_gt_ids) / n_unique_gt) if n_unique_gt > 0 else 0.0
    precision = (len(yes_llm_ids) / n_unique_llm) if n_unique_llm > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return pd.Series({
        "n_unique_gt": n_unique_gt,
        "n_unique_llm": n_unique_llm,
        "recall": recall,
        "precision": precision,
        "f1": f1
    })

print("\nComputing per-row precision, recall, and F1 ...")
metrics_df = df.apply(compute_pair_metrics, axis=1)

# Attach metrics to main df
df["n_unique_gt"] = metrics_df["n_unique_gt"]
df["n_unique_llm"] = metrics_df["n_unique_llm"]
df["recall"] = metrics_df["recall"]
df["precision"] = metrics_df["precision"]
df["f1"] = metrics_df["f1"]

# Save updated CSV
# df.to_csv(io_csv, index=False)
# print(f"\n✅ Metrics added and saved to: {io_csv}")

# Small preview
print(df[["n_unique_gt", "n_unique_llm", "recall", "precision", "f1"]].head())

# ==========================================
# 4. Print average precision, recall, and F1
# ==========================================

# If you want to include all rows (even those with 0/0 → 0 scores):
avg_precision = df["precision"].mean()
avg_recall = df["recall"].mean()
avg_f1 = df["f1"].mean()

print("\n=== Global Averages (including all rows) ===")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall:    {avg_recall:.4f}")
print(f"Average F1:        {avg_f1:.4f}")

# (Optional) If you want to ignore rows where there were no pairs (n_unique_gt == 0 or n_unique_llm == 0):
valid_mask = (df["n_unique_gt"] > 0) & (df["n_unique_llm"] > 0)
if valid_mask.any():
    avg_precision_valid = df.loc[valid_mask, "precision"].mean()
    avg_recall_valid = df.loc[valid_mask, "recall"].mean()
    avg_f1_valid = df.loc[valid_mask, "f1"].mean()

    print("\n=== Global Averages (only rows with at least one GT and one LLM) ===")
    print(f"Average Precision (valid): {avg_precision_valid:.4f}")
    print(f"Average Recall (valid):    {avg_recall_valid:.4f}")
    print(f"Average F1 (valid):        {avg_f1_valid:.4f}")
else:
    print("\n(No valid rows with both GT and LLM limitations found.)")


import pandas as pd
import ast
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ==========================================
# 1. Load CSV and column name
# ==========================================
# io_csv = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/SFT_from_zs_output/mistral/output/df_gpt_eval.csv"
col_eval = "llm_evaluation_results"

# print(f"Loading CSV: {io_csv} ...")
# df = pd.read_csv(io_csv)

# ==========================================
# 2. Parse llm_evaluation_results from str -> list via ast
# ==========================================
def parse_eval_list(val):
    """
    Convert a string representation of a Python list into a real list
    using ast.literal_eval. If already a list, return as-is.
    """
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        return ast.literal_eval(val)
    except (SyntaxError, ValueError, TypeError):
        return []

print(f"Parsing '{col_eval}' with ast.literal_eval ...")
df[col_eval] = df[col_eval].apply(parse_eval_list)

# ==========================================
# 3. Prepare similarity helpers
# ==========================================

# Rouge-L scorer (create once)
rougeL_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def cosine_sim(gt_text, llm_text):
    """Cosine similarity using TF-IDF vectors."""
    vect = TfidfVectorizer().fit([gt_text, llm_text])
    tfidf = vect.transform([gt_text, llm_text])
    cos = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
    return float(cos)

def jaccard_sim(gt_text, llm_text):
    """Jaccard similarity over lowercased whitespace-token sets."""
    tokens1 = set(gt_text.lower().split())
    tokens2 = set(llm_text.lower().split())
    union = tokens1 | tokens2
    if not union:
        return 0.0
    inter = tokens1 & tokens2
    return float(len(inter) / len(union))

def rougeL_f1(gt_text, llm_text):
    """ROUGE-L F1 between reference (gt) and candidate (llm)."""
    scores = rougeL_scorer.score(gt_text, llm_text)
    return float(scores['rougeL'].fmeasure)

def bertscore_f1(gt_text, llm_text):
    """
    BERTScore F1 between reference (gt) and candidate (llm).
    We use llm_text as candidate and gt_text as reference.
    """
    P, R, F = bert_score([llm_text], [gt_text], lang='en', verbose=False)
    return float(F[0])

# ==========================================
# 4. Compute per-row averages over YES pairs
# ==========================================
def compute_similarity_metrics(row):
    """
    For this row's llm_evaluation_results:
      - Take only pairs where answer is 'Yes'
      - Extract gt_limitation and llm_limitation texts
      - Compute cosine, jaccard, rougeL, bertscore per pair
      - Return row-wise averages
    """
    items = row[col_eval]
    if not isinstance(items, list) or len(items) == 0:
        return pd.Series({
            "avg_cosine_sim": 0.0,
            "avg_jaccard_sim": 0.0,
            "avg_rougeL": 0.0,
            "avg_bertscore": 0.0,
            "n_yes_pairs": 0
        })
    
    gt_texts = []
    llm_texts = []

    for item in items:
        # item example:
        # ['Pair 1: Yes', 'gt_id:0', 'gt_limitation:TEXT...', 'llm_id:0', 'llm_limitation:TEXT...']
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        
        # Check if this pair is marked as Yes
        answer_str = str(item[0])
        is_yes = "YES" in answer_str.upper()
        if not is_yes:
            continue
        
        gt_text = None
        llm_text = None
        
        for elem in item:
            if isinstance(elem, str):
                if elem.startswith("gt_limitation:"):
                    gt_text = elem.split("gt_limitation:", 1)[1].strip()
                elif elem.startswith("llm_limitation:"):
                    llm_text = elem.split("llm_limitation:", 1)[1].strip()
        
        if gt_text and llm_text:
            gt_texts.append(gt_text)
            llm_texts.append(llm_text)
    
    n_yes = len(gt_texts)
    if n_yes == 0:
        return pd.Series({
            "avg_cosine_sim": 0.0,
            "avg_jaccard_sim": 0.0,
            "avg_rougeL": 0.0,
            "avg_bertscore": 0.0,
            "n_yes_pairs": 0
        })
    
    cos_vals = []
    jac_vals = []
    rougel_vals = []
    bert_vals = []
    
    for gt_text, llm_text in zip(gt_texts, llm_texts):
        try:
            cos_vals.append(cosine_sim(gt_text, llm_text))
        except Exception:
            cos_vals.append(0.0)
        try:
            jac_vals.append(jaccard_sim(gt_text, llm_text))
        except Exception:
            jac_vals.append(0.0)
        try:
            rougel_vals.append(rougeL_f1(gt_text, llm_text))
        except Exception:
            rougel_vals.append(0.0)
        try:
            bert_vals.append(bertscore_f1(gt_text, llm_text))
        except Exception:
            bert_vals.append(0.0)
    
    return pd.Series({
        "avg_cosine_sim": float(np.mean(cos_vals)) if cos_vals else 0.0,
        "avg_jaccard_sim": float(np.mean(jac_vals)) if jac_vals else 0.0,
        "avg_rougeL": float(np.mean(rougel_vals)) if rougel_vals else 0.0,
        "avg_bertscore": float(np.mean(bert_vals)) if bert_vals else 0.0,
        "n_yes_pairs": n_yes
    })

print("\nComputing similarity metrics (cosine, jaccard, ROUGE-L, BERTScore) for YES pairs...")
sim_metrics = df.apply(compute_similarity_metrics, axis=1)

df["avg_cosine_sim"] = sim_metrics["avg_cosine_sim"]
df["avg_jaccard_sim"] = sim_metrics["avg_jaccard_sim"]
df["avg_rougeL"] = sim_metrics["avg_rougeL"]
df["avg_bertscore"] = sim_metrics["avg_bertscore"]
df["n_yes_pairs"] = sim_metrics["n_yes_pairs"]

print("avg_cosine_sim",df["avg_cosine_sim"].mean())
print("avg_jaccard_sim",df["avg_jaccard_sim"].mean())
print("avg_rougeL",df["avg_rougeL"].mean())
print("avg_bertscore",df["avg_bertscore"].mean())
print("n_yes_pairs",df["n_yes_pairs"].mean())

# ==========================================
# 5. Save and quick preview
# ==========================================
df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/llama_3_1_70b/with_kg_triplets/limgen_with_novelty_lim/df_eval_gpt_limgen_with_nov_lim_0_503_llama_merger_final.csv", index=False)

# print(f"\n✅ Similarity metrics added and saved back to: {io_csv}")

print("\nPreview of new columns:")
print(df[["n_yes_pairs", "avg_cosine_sim", "avg_jaccard_sim", "avg_rougeL", "avg_bertscore"]].head())

import pandas as pd

# 1. Define the specific columns you want to work with
cols = [
    "recall", "precision", "f1", "avg_cosine_sim", 
    "avg_jaccard_sim", "avg_rougeL", "avg_bertscore", "n_yes_pairs"
]

# 2. Remove any row where ANY of these columns contain a 0
# (df[cols] != 0) creates a True/False grid. 
# .all(axis=1) keeps only the rows where every single column is True (not zero).
df_filtered = df[(df[cols] != 0).all(axis=1)]

# 3. Calculate the average (mean) for each of these columns
averages = df_filtered[cols].mean()

# 4. Print the results
print("Average metrics after removing rows with 0:")
print(averages) 

