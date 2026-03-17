# done as explaination in paper 

import pandas as pd 
df1 = pd.read_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/data/final_balanced_kde/df_updated_with_retrieval.csv") 


# df = df.head(1) 

import ast 
def extract_abstractText(pdf_text_cell):
    """
    pdf_text_cell is a string that looks like a Python dict (or already a dict).
    Returns the value of key 'abstractText' (or None if missing / parse fails).
    """
    if pd.isna(pdf_text_cell) or pdf_text_cell is None:
        return None

    # If it's already a dict, use it directly
    if isinstance(pdf_text_cell, dict):
        return pdf_text_cell.get("abstractText")

    # Otherwise parse the string safely
    if isinstance(pdf_text_cell, str):
        s = pdf_text_cell.strip()
        if not s:
            return None
        try:
            d = ast.literal_eval(s)   # safe parsing
            if isinstance(d, dict):
                return d.get("abstractText")
        except Exception:
            return None

    return None
df["abstractText"] = df["pdf_text"].apply(extract_abstractText) 

# summariz abstract 
import os
import time
import pandas as pd
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

# --- Config ---
os.environ['OPENAI_API_KEY'] = ''
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"
client = OpenAI(api_key=api_key)

SUMMARY_COL = "abstract_summary"

SYSTEM_MSG = (
    "You summarize scientific paper abstracts.\n"
    "Return ONLY the summary text (no bullets, no headings).\n"
    "Write 1-2 sentences (word length 30-35), keep key contributions, method, and main result.\n"
    "Do not add information not present in the abstract."
)

def summarize_abstract(text: str, max_retries: int = 6) -> str:
    if not isinstance(text, str) or len(text.strip()) < 40:
        return ""

    user_msg = f"Abstract:\n{text.strip()}\n\nSummary:"

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                temperature=0.2,
                max_tokens=140,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as e:
            # backoff
            sleep = min(60, 2 ** (attempt + 1))
            print(f"[retry {attempt+1}/{max_retries}] {type(e).__name__}: sleeping {sleep}s")
            time.sleep(sleep)
        except Exception as e:
            print("Unexpected error:", e)
            return ""

    return ""

# --- Apply to dataframe (with resume support) ---
if SUMMARY_COL not in df.columns:
    df[SUMMARY_COL] = ""

start_idx = 0  # change if you want
save_every = 20
# out_csv = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/data/final_balanced_kde/df_with_abstract_summaries.csv"

for i in range(start_idx, len(df)):
    if isinstance(df.at[i, SUMMARY_COL], str) and df.at[i, SUMMARY_COL].strip():
        continue  # already summarized

    df.at[i, SUMMARY_COL] = summarize_abstract(df.at[i, "abstractText"])

#     if (i + 1) % save_every == 0:
#         df.to_csv(out_csv, index=False)
#         print(f"Saved checkpoint at row {i+1}")

# df.to_csv(out_csv, index=False)
# print("Done. Saved:", out_csv)


import re
import time
import requests
from urllib.parse import quote_plus
import pandas as pd

# ============================================================
# 1) Helpers: build query + reconstruct OpenAlex abstract
# ============================================================

def build_keyword_query(text: str, k: int = 14) -> str:
    """
    Build a compact keyword query from an abstract/summary.
    Uses simple term frequency after removing common stopwords.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    stop = {
        "the","a","an","and","or","to","of","in","for","on","with","by","from","is","are","was","were","be",
        "this","that","these","those","we","our","their","they","it","as","at","not","can","may","might",
        "study","paper","presents","present","novel","method","approach","results","show","demonstrate",
        "based","using","use","used","model","models","data","dataset","introduce","propose","learn","learning",
        "new","task","tasks","framework","system","analysis","experiments","evaluation"
    }

    words = re.findall(r"[a-zA-Z0-9\-]{3,}", text.lower())
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1

    # Top-k frequent keywords
    return " ".join([w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]])


def inverted_index_to_abstract(inv) -> str:
    """
    Convert OpenAlex 'abstract_inverted_index' to plaintext.
    inv is like: {"word": [pos1, pos2, ...], ...}
    """
    if not isinstance(inv, dict) or not inv:
        return ""

    pos_to_word = {}
    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and p not in pos_to_word:
                pos_to_word[p] = word

    if not pos_to_word:
        return ""

    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "") for i in range(max_pos + 1)]
    text = " ".join([w for w in words if w])
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 2) OpenAlex-only retrieval: returns top-k papers WITH abstracts
# ============================================================

def fetch_topk_similar_papers_openalex(
    input_text: str,
    *,
    top_k: int = 5,
    per_page: int = 25,
    openalex_mailto: str | None = None,
    timeout_s: int = 20,
    debug: bool = False,
    max_retries: int = 3
) -> list[dict]:
    """
    OpenAlex-only: search works by keywords, request abstract_inverted_index,
    reconstruct abstract text, then locally rerank.

    Returns list[dict] with:
      rank, title, abstract, year, authors, venue, doi, url, openalex_id,
      similarity_score, openalex_relevance_score, cited_by_count
    """
    if not isinstance(input_text, str) or len(input_text.strip()) < 20:
        if debug:
            print("[DEBUG] input_text invalid/too short -> returning []")
        return []

    query_text = build_keyword_query(input_text, k=14)
    if not query_text:
        if debug:
            print("[DEBUG] keyword query empty -> returning []")
        return []

    # local rerank helper
    stop = {
        "the","a","an","and","or","to","of","in","for","on","with","by","from","is","are","was","were","be",
        "this","that","these","those","we","our","their","they","it","as","at","not","can","may","might",
        "study","paper","method","approach","results","model","data","dataset","introduce","propose",
        "new","task","tasks","framework","system","analysis","experiments","evaluation"
    }

    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        toks = [t for t in s.split() if len(t) >= 3 and t not in stop]
        return set(toks)

    def jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    q_tokens = norm_tokens(input_text)

    # ✅ Valid OpenAlex select fields (works endpoint)
    select = ",".join([
        "id",
        "title",
        "publication_year",
        "doi",
        "authorships",
        "primary_location",
        "cited_by_count",
        "relevance_score",
        "abstract_inverted_index"
    ])

    params = f"search={quote_plus(query_text)}&per-page={per_page}&select={quote_plus(select)}"
    if openalex_mailto:
        params += f"&mailto={quote_plus(openalex_mailto)}"

    url = f"https://api.openalex.org/works?{params}"
    headers = {"User-Agent": f"openalex-similarity-script/1.0 (mailto:{openalex_mailto or 'none'})"}

    # Request with retry
    j = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
            if debug:
                print(f"[DEBUG] OpenAlex GET -> {r.status_code}")
                print(f"[DEBUG] query_text: {query_text}")
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            break
        except Exception as e:
            if debug:
                print(f"[DEBUG] OpenAlex request failed: {type(e).__name__}: {e}")
                try:
                    print("[DEBUG] Response text:", r.text[:400])
                except Exception:
                    pass
            time.sleep(1.0 * (attempt + 1))

    if not isinstance(j, dict):
        return []

    results = j.get("results", [])
    if not results:
        if debug:
            print("[DEBUG] OpenAlex returned 0 results")
        return []

    candidates = []
    for w in results:
        title = (w.get("title") or "").strip()
        year = w.get("publication_year")
        doi = w.get("doi")
        openalex_id = w.get("id")

        # Authors
        authors = []
        for a in (w.get("authorships") or [])[:10]:
            name = ((a.get("author") or {}).get("display_name") or "").strip()
            if name:
                authors.append(name)
        authors_str = ", ".join(authors)

        # Venue/source + URL
        pl = w.get("primary_location") or {}
        source = (pl.get("source") or {})
        venue = (source.get("display_name") or "").strip()
        url_out = pl.get("landing_page_url") or openalex_id

        # Abstract
        abstract_plain = inverted_index_to_abstract(w.get("abstract_inverted_index"))

        # Similarity score uses title + venue + abstract (much better)
        score = jaccard(q_tokens, norm_tokens(f"{title} {venue} {abstract_plain[:1500]}"))

        candidates.append({
            "openalex_id": openalex_id,
            "title": title,
            "abstract": abstract_plain,
            "year": year,
            "authors": authors_str,
            "venue": venue,
            "doi": doi,
            "url": url_out,
            "cited_by_count": w.get("cited_by_count"),
            "relevance_score": w.get("relevance_score"),
            "similarity_score": score,
        })

    # Rank: similarity, then OpenAlex relevance_score, then citations
    candidates.sort(
        key=lambda x: (
            x.get("similarity_score") or 0,
            x.get("relevance_score") or 0,
            x.get("cited_by_count") or 0
        ),
        reverse=True
    )

    # Output top_k with rank, INCLUDING abstract
    out = []
    for i, p in enumerate(candidates[:top_k], 1):
        out.append({
            "rank": i,
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
            "year": p.get("year"),
            "authors": p.get("authors") or "",
            "venue": p.get("venue") or "",
            "doi": p.get("doi"),
            "url": p.get("url"),
            "openalex_id": p.get("openalex_id"),
            "similarity_score": round(float(p.get("similarity_score") or 0.0), 4),
            "openalex_relevance_score": p.get("relevance_score"),
            "cited_by_count": p.get("cited_by_count"),
        })

    return out


# ============================================================
# 3) Apply to dataframe: store list-of-abstracts in new column
# ============================================================

def top5_abstracts_from_openalex(input_text: str, mailto: str) -> list[str]:
    papers = fetch_topk_similar_papers_openalex(
        input_text=input_text,
        top_k=5,
        per_page=25,
        openalex_mailto=mailto,
        debug=False
    )
    # keep only non-empty abstracts
    return [p["abstract"] for p in papers if isinstance(p.get("abstract"), str) and p["abstract"].strip()]


def top5_papers_from_openalex(input_text: str, mailto: str) -> list[dict]:
    # full list-of-dicts (includes abstract + metadata)
    return fetch_topk_similar_papers_openalex(
        input_text=input_text,
        top_k=5,
        per_page=25,
        openalex_mailto=mailto,
        debug=False
    )


# =========================
# Example usage
# =========================

# df must already exist, and must have 'abstract_summary' (or use 'abstractText')
OPENALEX_MAILTO = "your_real_email@domain.com"

# 1) Column containing list-of-abstracts (list[str])
df["openalex_top5_abstracts"] = df["abstract_summary"].fillna("").apply(
    lambda x: top5_abstracts_from_openalex(x, OPENALEX_MAILTO)
)

# 2) (Optional) Column containing list-of-dicts (list[dict]) for full metadata
df["openalex_top5_papers"] = df["abstract_summary"].fillna("").apply(
    lambda x: top5_papers_from_openalex(x, OPENALEX_MAILTO)
)

# Inspect one row
print(df.loc[0, "openalex_top5_abstracts"])
print(df.loc[0, "openalex_top5_papers"])

df.to_csv("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_outside_master_agent_and_tool/df.csv",index=False)


