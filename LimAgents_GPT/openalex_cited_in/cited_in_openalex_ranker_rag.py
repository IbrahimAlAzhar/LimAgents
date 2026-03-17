import pandas as pd 
df = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df_row_100_199_tool_accessed.csv") 

import pandas as pd
import ast
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# 1. SETUP MODELS & SPLITTERS
# ---------------------------------------------------------
# Load embedding model (Small and fast model for RAG)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup Text Splitter for OpenAlex (Targeting ~512 tokens)
# We use tiktoken encoder to ensure we respect token limits accurately
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4", 
    chunk_size=512, 
    chunk_overlap=50
)

def safe_parse(val):
    """Parses stringified lists/dicts into actual Python objects."""
    if isinstance(val, (dict, list)):
        return val
    if pd.isna(val) or val == "":
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None

def normalize_scores(scores):
    """Normalizes a list of scores to 0-1 range for hybrid fusion."""
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    scores = np.array(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)

# 2. MAIN PROCESSING FUNCTION
# ---------------------------------------------------------
def process_row_rag(row):
    chunks = []
    
    # --- PART A: Process OpenAlex Abstracts ---
    openalex_data = safe_parse(row.get('openalex_top5_abstracts'))
    
    if isinstance(openalex_data, list):
        for abstract in openalex_data:
            if abstract and isinstance(abstract, str):
                # Logic: If > 512 tokens, split. Else keep as is.
                # The splitter handles the logic automatically based on chunk_size=512
                split_texts = text_splitter.split_text(abstract)
                chunks.extend(split_texts)

    # --- PART B: Process cited_in_clean ---
    cited_data = safe_parse(row.get('cited_in_clean'))
    
    if isinstance(cited_data, dict):
        for paper_key, paper_content in cited_data.items():
            if not isinstance(paper_content, dict):
                continue
            
            # 1. Take 'abstractText' -> Make 1 chunk
            abs_text = paper_content.get('abstractText')
            if abs_text and isinstance(abs_text, str) and abs_text.strip():
                chunks.append(abs_text)
                
            # 2. Take 'sections' -> 'text' -> Make new chunk per section
            sections = paper_content.get('sections')
            if isinstance(sections, list):
                for sec in sections:
                    if isinstance(sec, dict):
                        sec_text = sec.get('text')
                        if sec_text and isinstance(sec_text, str) and sec_text.strip():
                            chunks.append(sec_text)

    # If no chunks were found, return empty lists
    if not chunks:
        return pd.Series([[], []])

    # --- PART C: Hybrid Search (FAISS + BM25) ---
    query = row.get('input_text_cleaned', "")
    if not isinstance(query, str) or not query.strip():
        # If no query, return unsorted chunks
        return pd.Series([chunks, chunks])

    # 1. BM25 (Sparse) Scores
    tokenized_chunks = [doc.split() for doc in chunks]
    tokenized_query = query.split()
    
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenized_query)
    norm_bm25 = normalize_scores(bm25_scores)

    # 2. FAISS (Dense) Scores
    # Encode chunks and query
    chunk_embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])
    
    # Initialize FAISS (L2 Distance or Inner Product)
    dimension = chunk_embeddings.shape[1]
    # Normalizing vectors for Cosine Similarity (Inner Product)
    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_embedding)
    
    index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Sim after norm)
    index.add(chunk_embeddings)
    
    # Search
    # We want distances for ALL chunks to sort them
    D, I = index.search(query_embedding, len(chunks)) 
    
    # FAISS returns results sorted. We need to map scores back to original chunk indices
    # to combine with BM25.
    faiss_scores_mapped = np.zeros(len(chunks))
    for rank, original_index in enumerate(I[0]):
        faiss_scores_mapped[original_index] = D[0][rank]
    
    norm_faiss = normalize_scores(faiss_scores_mapped)

    # 3. Hybrid Fusion (Simple Weighted Average)
    # 0.5 weight to Keyword match (BM25), 0.5 to Semantic match (FAISS)
    final_scores = (0.5 * norm_bm25) + (0.5 * norm_faiss)
    
    # Sort chunks based on final score (Descending)
    # argsort returns indices of sorted low-to-high, so we reverse [::-1]
    sorted_indices = np.argsort(final_scores)[::-1]
    sorted_chunks = [chunks[i] for i in sorted_indices]

    return pd.Series([chunks, sorted_chunks])

# 3. APPLY TO DATAFRAME
# ---------------------------------------------------------
# Assuming your dataframe is named 'df'
# Apply the function. It returns two columns.
df[['all_chunks', 'sorted_chunks']] = df.apply(process_row_rag, axis=1)

# Verify Output
print("Processing Complete.")
print(df[['input_text_cleaned', 'sorted_chunks']].head(1)) 

df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df_row_100_199_tool_accessed_cited_ranker.csv") 

df = df.rename(columns={'all_chunks': 'all_chunks_cited_in_openalex', 'sorted_chunks': 'sorted_chunks_cited_in_openalex'})

import ast
import pandas as pd

def safe_convert_list(text):
    if isinstance(text, list):
        return text
    if pd.isna(text) or text == "":
        return []
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

# Example application
df['sorted_chunks_cited_in_openalex'] = df['sorted_chunks_cited_in_openalex'].apply(safe_convert_list) 

def add_chunk_prefixes(chunks_list):
    # Safety check: ensure it is actually a list
    if not isinstance(chunks_list, list):
        return []
    
    # List comprehension to add prefix 'chunk_N: '
    return [f"chunk_{i+1}: {text}" for i, text in enumerate(chunks_list)]

# Apply to the dataframe
df['prefixed_sorted_chunks_cited_in_openalex'] = df['sorted_chunks_cited_in_openalex'].apply(add_chunk_prefixes)

def format_chunks_as_lists(chunks_list):
    # Safety check: ensure it is actually a list
    if not isinstance(chunks_list, list):
        return []
    
    # Create a list of lists: [['chunk_1: ...'], ['chunk_2: ...']]
    return [[f"chunk_{i+1}: {text}"] for i, text in enumerate(chunks_list)]

# Apply to the dataframe
df['prefixed_sorted_chunks_cited_in_openalex'] = df['prefixed_sorted_chunks_cited_in_openalex'].apply(format_chunks_as_lists)

df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df_row_100_199_tool_accessed.csv",index=False)