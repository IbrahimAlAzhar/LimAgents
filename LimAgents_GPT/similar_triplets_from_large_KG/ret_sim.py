import os
import json
import ast
import re
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# =========================================================
# 0) CONFIG
# =========================================================
# Big KG paths
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "dense_knowledge_graph.pkl")

# Your dataframe must already be loaded as `df` and have this column:
INPUT_COL = "kg_triplets"

# New column to store retrieved similar info:
OUTPUT_COL = "kg_triplets_retrieved"

# SentenceTransformer config
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Retrieval knobs
ENTITY_MATCH_TOPK = 3
ENTITY_SCORE_THRESHOLD = 0.70
HOP_K = 1
MAX_RETURN_EDGES = 200

# Optional: save updated dataframe
SAVE_OUT_CSV = True
OUT_CSV_PATH = ""
# =========================================================
# 1) HELPERS (parse + normalize)
# =========================================================
def parse_triplets(json_str):
    try:
        if pd.isna(json_str) or "SKIPPED" in str(json_str) or "ERROR" in str(json_str):
            return []
        if isinstance(json_str, str):
            s = json_str.replace("```json", "").replace("```", "").strip()
            s = s.replace("\n", " ")
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                data = ast.literal_eval(s)
        else:
            data = json_str
        return data.get("triplets", [])
    except Exception:
        return []

_clean_pattern = re.compile(r"[^a-zA-Z0-9\s]")

def normalize_entity(x: str) -> str:
    x = str(x or "").strip()
    x = _clean_pattern.sub("", x).lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x

def entities_from_row_triplets(triplets):
    ents = []
    for t in triplets:
        ents.append(normalize_entity(t.get("head", "")))
        ents.append(normalize_entity(t.get("tail", "")))
    ents = [e for e in dict.fromkeys(ents) if e and len(e) < 100]
    return ents


# =========================================================
# 2) LOAD BIG KG FROM PKL
# =========================================================
def load_dense_kg_from_pkl(pkl_path: str) -> nx.DiGraph:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Loaded object is not a NetworkX graph. Type: {type(G)}")
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    return G


# =========================================================
# 3) BUILD EMBEDDING INDEX (one-time)
# =========================================================
def build_node_index(G: nx.DiGraph, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    node_names = list(G.nodes())
    model = SentenceTransformer(model_name)
    node_emb = model.encode(
        node_names,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return model, node_names, node_emb


# =========================================================
# 4) RETRIEVE SIMILAR TRIPLETS FOR ONE ROW
# =========================================================
def retrieve_similar_kg_triplets(
    G: nx.DiGraph,
    model: SentenceTransformer,
    node_names,
    node_emb,
    row_triplets_json,
    entity_match_topk=ENTITY_MATCH_TOPK,
    entity_score_threshold=ENTITY_SCORE_THRESHOLD,
    hop_k=HOP_K,
    max_return_edges=MAX_RETURN_EDGES
):
    triplets = parse_triplets(row_triplets_json)
    query_entities = entities_from_row_triplets(triplets)
    if not query_entities:
        return {"triplets": []}

    q_emb = model.encode(query_entities, normalize_embeddings=True)
    hits = util.semantic_search(q_emb, node_emb, top_k=entity_match_topk)

    matched_nodes = set()
    for ent_hits in hits:
        for h in ent_hits:
            if h["score"] >= entity_score_threshold:
                matched_nodes.add(node_names[h["corpus_id"]])

    if not matched_nodes:
        return {"triplets": []}

    # expand neighborhood
    if hop_k <= 1:
        expanded_nodes = set(matched_nodes)
        for n in list(matched_nodes):
            expanded_nodes.update(G.predecessors(n))
            expanded_nodes.update(G.successors(n))
    else:
        expanded_nodes = set(matched_nodes)
        frontier = set(matched_nodes)
        for _ in range(hop_k):
            nxt = set()
            for n in frontier:
                nxt.update(G.predecessors(n))
                nxt.update(G.successors(n))
            nxt -= expanded_nodes
            expanded_nodes |= nxt
            frontier = nxt
            if not frontier:
                break

    # score nodes for ranking edges
    node_to_idx = {n: i for i, n in enumerate(node_names)}
    expanded_list = [n for n in expanded_nodes if n in node_to_idx]
    if not expanded_list:
        return {"triplets": []}

    expanded_idx = [node_to_idx[n] for n in expanded_list]
    expanded_emb = node_emb[expanded_idx]

    sim = util.cos_sim(expanded_emb, q_emb).cpu().numpy()
    node_best = {expanded_list[i]: float(sim[i].max()) for i in range(len(expanded_list))}

    edges_scored = []
    for u, v, data in G.edges(expanded_nodes, data=True):
        rel = data.get("relation", "")
        score = max(node_best.get(u, 0.0), node_best.get(v, 0.0))
        edges_scored.append((score, u, rel, v))

    edges_scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    seen = set()
    for score, u, rel, v in edges_scored[:max_return_edges]:
        key = (u, rel, v)
        if key in seen:
            continue
        seen.add(key)
        out.append({"head": u, "relation": rel, "tail": v, "score": round(score, 4)})

    return {"triplets": out}


# =========================================================
# 5) MAIN: ADD NEW COLUMN
# =========================================================
def add_retrieved_triplets_column(df, G, model, node_names, node_emb):
    retrieved_jsons = []
    for x in tqdm(df[INPUT_COL].tolist(), total=len(df)):
        retrieved = retrieve_similar_kg_triplets(
            G, model, node_names, node_emb, x,
            entity_match_topk=ENTITY_MATCH_TOPK,
            entity_score_threshold=ENTITY_SCORE_THRESHOLD,
            hop_k=HOP_K,
            max_return_edges=MAX_RETURN_EDGES
        )
        retrieved_jsons.append(json.dumps(retrieved))
    df[OUTPUT_COL] = retrieved_jsons
    return df


# =========================================================
# RUN
# =========================================================
# 0) df must be defined before running this script
# df = pd.read_csv("your_input.csv")  # MUST have column kg_triplets

# INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df_row_100_199_tool_accessed_gpt_reranker.csv"
INPUT_CSV = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/knowledge_graph/df_kg_gpt4o.csv"
df = pd.read_csv(INPUT_CSV)  
# df = df.head(1)

dense_kg = load_dense_kg_from_pkl(OUTPUT_PKL)
print(f"✅ Loaded dense KG: nodes={dense_kg.number_of_nodes()} edges={dense_kg.number_of_edges()}")

model, node_names, node_emb = build_node_index(dense_kg)
print("✅ Built embeddings index")

df = add_retrieved_triplets_column(df, dense_kg, model, node_names, node_emb)
print(f"✅ Added new column: {OUTPUT_COL}")

if SAVE_OUT_CSV:
    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"✅ Saved updated df to: {OUT_CSV_PATH}")
