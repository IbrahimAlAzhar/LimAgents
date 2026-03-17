#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid RAG retriever (FAISS + BM25) where:
- Each ROW in the big CSV is ONE "chunk"/document = full_text_reconstructed
- The "payload" you want back = concatenate_lim_peer
- For each test row (input_text_cleaned), retrieve top-3 documents and attach their concatenate_lim_peer
- Save augmented test CSV to the requested output directory

Requirements:
  pip install -U pandas numpy faiss-cpu sentence-transformers tqdm
(FAISS may already be available as `faiss` in your env.)
"""

import os
import re
import json
import math
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Paths (as you provided)
# -----------------------------

# -----------------------------
# Columns
# -----------------------------
DOC_TEXT_COL = "full_text_reconstructed"
DOC_PAYLOAD_COL = "concatenate_lim_peer"
QUERY_COL = "input_text_cleaned"
# input_text_cleaned
# -----------------------------
# Retrieval settings
# -----------------------------
TOPK = 3
K_DENSE = 50
K_BM25 = 50
ALPHA = 0.5  # weight for dense score; (1-ALPHA) for BM25

# Embedding model (change if you want)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

# Tokenization for BM25
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def bm25_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25InvertedIndex:
    """
    Fast BM25 using an inverted index (no O(N) scoring per query).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.doc_len = None              # np.array length N
        self.df = defaultdict(int)       # term -> document frequency
        self.postings = defaultdict(list)  # term -> list of (doc_id, tf)
        self.idf = {}                    # term -> idf

    def build(self, tokenized_docs):
        self.N = len(tokenized_docs)
        if self.N == 0:
            self.avgdl = 0.0
            self.doc_len = np.array([], dtype=np.int32)
            return

        doc_len = np.zeros(self.N, dtype=np.int32)

        for doc_id, toks in enumerate(tqdm(tokenized_docs, desc="BM25 build (token stats)", total=self.N)):
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            doc_len[doc_id] = len(toks)
            for t, c in tf.items():
                self.postings[t].append((doc_id, c))
                self.df[t] += 1

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.N > 0 else 0.0

        # Compute IDF (BM25 Okapi)
        # idf = log( (N - df + 0.5) / (df + 0.5) + 1 )
        self.idf = {}
        for t, dft in self.df.items():
            self.idf[t] = math.log((self.N - dft + 0.5) / (dft + 0.5) + 1.0)

    def get_topk(self, query_tokens, topk=50):
        """
        Returns list of (doc_id, bm25_score) sorted desc by score, length <= topk.
        """
        if self.N == 0 or not query_tokens:
            return []

        # Query term frequency (optional, but helps if repeated tokens)
        qtf = defaultdict(int)
        for t in query_tokens:
            qtf[t] += 1

        scores = defaultdict(float)
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for t, qcount in qtf.items():
            if t not in self.postings:
                continue
            idf_t = self.idf.get(t, 0.0)
            plist = self.postings[t]

            for doc_id, tf in plist:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                score = idf_t * (tf * (self.k1 + 1.0)) / (denom + 1e-12)
                # multiply by query term count (rarely matters, but okay)
                scores[doc_id] += score * qcount

        if not scores:
            return []

        # Top-k using heap
        return heapq.nlargest(topk, scores.items(), key=lambda x: x[1])


def minmax_norm(x: np.ndarray):
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def build_store(df: pd.DataFrame):
    """
    Build:
      - FAISS index over embeddings of full_text_reconstructed (one per row)
      - BM25 inverted index over tokenized full_text_reconstructed
      - aligned arrays for payloads and original row indices
    """
    # Filter: keep only rows with non-empty doc text and payload
    df = df.copy()
    df[DOC_TEXT_COL] = df[DOC_TEXT_COL].fillna("").astype(str)
    df[DOC_PAYLOAD_COL] = df[DOC_PAYLOAD_COL].fillna("").astype(str)

    keep_mask = df[DOC_TEXT_COL].str.strip().astype(bool) & df[DOC_PAYLOAD_COL].str.strip().astype(bool)
    df_keep = df.loc[keep_mask].copy()

    docs = df_keep[DOC_TEXT_COL].tolist()
    payloads = df_keep[DOC_PAYLOAD_COL].tolist()
    orig_ids = df_keep.index.tolist()  # original df row index

    print(f"[INFO] Corpus rows: {len(df)}")
    print(f"[INFO] Kept rows (non-empty doc+payload): {len(docs)}")

    if len(docs) == 0:
        raise RuntimeError("No valid documents after filtering. Check your columns / empty rows.")

    # --- Embedding model ---
    model = SentenceTransformer(EMBED_MODEL)

    # --- Build FAISS index incrementally ---
    # Encode first batch to get dim
    first_vec = model.encode([docs[0]], normalize_embeddings=True)
    dim = int(first_vec.shape[1])
    index = faiss.IndexFlatIP(dim)

    # Add embeddings in batches
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="FAISS build (encode+add)"):
        batch_docs = docs[i:i + BATCH_SIZE]
        vecs = model.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        index.add(vecs)

    # --- Build BM25 inverted index ---
    tokenized_docs = []
    for d in tqdm(docs, desc="BM25 tokenize"):
        tokenized_docs.append(bm25_tokenize(d))

    bm25 = BM25InvertedIndex(k1=1.5, b=0.75)
    bm25.build(tokenized_docs)

    store = {
        "model": model,
        "faiss": index,
        "bm25": bm25,
        "docs": docs,
        "payloads": payloads,
        "orig_ids": orig_ids,
    }
    return store


def hybrid_retrieve_topk(store, query_text: str, topk=3, k_dense=50, k_bm25=50, alpha=0.5):
    """
    Returns list of dicts with top-k results; each result includes payload (concatenate_lim_peer).
    """
    if not isinstance(query_text, str) or not query_text.strip():
        return []

    model = store["model"]
    index = store["faiss"]
    bm25 = store["bm25"]

    payloads = store["payloads"]
    orig_ids = store["orig_ids"]
    n_docs = len(payloads)

    # ---- Dense retrieval ----
    qv = model.encode([query_text], normalize_embeddings=True)
    qv = np.asarray(qv, dtype=np.float32)
    k_dense_eff = min(k_dense, n_docs)
    dense_scores, dense_idx = index.search(qv, k_dense_eff)
    dense_scores = dense_scores[0]
    dense_idx = dense_idx[0]
    dense_map = {int(i): float(s) for i, s in zip(dense_idx, dense_scores) if i >= 0}

    # ---- BM25 retrieval ----
    q_tokens = bm25_tokenize(query_text)
    bm25_hits = bm25.get_topk(q_tokens, topk=min(k_bm25, n_docs))
    bm25_map = {int(i): float(s) for i, s in bm25_hits}

    # ---- Candidate union ----
    cand = list(set(dense_map.keys()) | set(bm25_map.keys()))
    if not cand:
        return []

    dense_vals = np.array([dense_map.get(i, 0.0) for i in cand], dtype=np.float32)
    bm25_vals = np.array([bm25_map.get(i, 0.0) for i in cand], dtype=np.float32)

    dense_norm = minmax_norm(dense_vals)
    bm25_norm = minmax_norm(bm25_vals)

    hybrid = alpha * dense_norm + (1.0 - alpha) * bm25_norm

    order = np.argsort(-hybrid)[:topk]

    out = []
    for rank_pos, j in enumerate(order, start=1):
        i = cand[int(j)]
        out.append({
            "rank": rank_pos,
            "doc_internal_idx": i,
            "doc_id": orig_ids[i],  # original df index
            "hybrid_score": float(hybrid[int(j)]),
            "dense_score": float(dense_map.get(i, 0.0)),
            "bm25_score": float(bm25_map.get(i, 0.0)),
            "concatenate_lim_peer": payloads[i],
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Loading big corpus CSV...")
    df_big = pd.read_csv(BIG_CSV)

    print("[INFO] Building hybrid store (FAISS + BM25)...")
    store = build_store(df_big)

    print("[INFO] Loading test CSV...")
    df_test = pd.read_csv(TEST_CSV)

    if QUERY_COL not in df_test.columns:
        raise ValueError(f"Test CSV missing required column: '{QUERY_COL}'")

    # New columns (one main column + optional per-rank)
    df_test["rag_top3_concatenate_lim_peer"] = ""
    df_test["rag_lim_peer_1"] = ""
    df_test["rag_lim_peer_2"] = ""
    df_test["rag_lim_peer_3"] = ""

    # Optional: keep which corpus row was retrieved
    df_test["rag_doc_id_1"] = ""
    df_test["rag_doc_id_2"] = ""
    df_test["rag_doc_id_3"] = ""

    print("[INFO] Retrieving top-3 limitations for each test row...")
    for idx in tqdm(df_test.index, desc="Hybrid retrieve (test rows)"):
        q = df_test.at[idx, QUERY_COL]
        hits = hybrid_retrieve_topk(
            store,
            q,
            topk=TOPK,
            k_dense=K_DENSE,
            k_bm25=K_BM25,
            alpha=ALPHA
        )

        lim_list = [h["concatenate_lim_peer"] for h in hits]

        # Store list as JSON string in one column (robust)
        df_test.at[idx, "rag_top3_concatenate_lim_peer"] = json.dumps(lim_list, ensure_ascii=False)

        # Also store individually (convenient)
        for j in range(3):
            if j < len(hits):
                df_test.at[idx, f"rag_lim_peer_{j+1}"] = hits[j]["concatenate_lim_peer"]
                df_test.at[idx, f"rag_doc_id_{j+1}"] = str(hits[j]["doc_id"])
            else:
                df_test.at[idx, f"rag_lim_peer_{j+1}"] = ""
                df_test.at[idx, f"rag_doc_id_{j+1}"] = ""

    print(f"[INFO] Saving augmented test CSV to: {OUT_FILE}")
    df_test.to_csv(OUT_FILE, index=False)

    print("[DONE] Saved:", OUT_FILE)


if __name__ == "__main__":
    main()
