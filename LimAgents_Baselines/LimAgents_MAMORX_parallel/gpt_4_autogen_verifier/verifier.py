import os
import pandas as pd
import autogen
from typing import Dict, List, Any
from tqdm import tqdm
import sys
import time
import json
import re
import chromadb
from chromadb.utils import embedding_functions
# Import ChromaDB errors to handle the specific exception
import chromadb.errors 

# ==========================================
# 1. CONFIGURATION
# ==========================================

os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini"

llm_config = {
    "config_list": [{"model": MODEL_ID, "api_key": api_key}],
    "temperature": 0.2, 
    "timeout": 120, 
    "cache_seed": None 
}

# Input/Output Paths
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ==========================================
# 2. FEATURE COMPONENT: MOCK MCP CLIENT (RAG Engine)
# ==========================================

class MockMCPClient:
    """
    Manages a per-session Vector Database for RAG verification.
    """
    def __init__(self):
        # Use ephemeral client for speed (resets on restart)
        self.chroma_client = chromadb.Client()
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction() 
        self.collection_name = "current_paper_knowledge"
        self.collection = None

    def reset_knowledge_base(self, cited_papers_text: str):
        """
        Clears previous paper data and loads new 'cited_in' text into the Vector DB.
        Splits text into ~512 token chunks.
        """
        # 1. Safely delete old collection if it exists
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except (ValueError, Exception): 
            # Catch generic Exception because ChromaDB errors vary by version
            pass 

        # 2. Create fresh collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name, 
            embedding_function=self.embed_fn
        )

        # 3. Simple Chunking (approx 512 words/tokens)
        if not cited_papers_text or pd.isna(cited_papers_text):
            return 
            
        words = str(cited_papers_text).split()
        chunk_size = 512
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        if not chunks:
            return

        # 4. Upsert to Chroma
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )
        # print(f"  [RAG] Loaded {len(chunks)} chunks into Vector DB.")

    def verify_claim(self, claim: str) -> str:
        """
        RAG Verification Tool.
        Queries the Vector DB for the claim. 
        Returns JSON with: verified (bool), score (0.0-1.0), and evidence.
        """
        if self.collection is None or self.collection.count() == 0:
            return json.dumps({
                "verified": False, 
                "score": 0.0, 
                "evidence": "No cited papers data available for verification."
            })

        # Query top 2 results
        results = self.collection.query(
            query_texts=[claim],
            n_results=2
        )

        distances = results['distances'][0]
        documents = results['documents'][0]
        
        best_dist = distances[0] if distances else 10.0
        best_doc = documents[0] if documents else ""

        # Normalize distance to score (Approximate: < 0.3 is very close, > 1.0 is far)
        score = max(0.0, min(1.0, 1.0 - (best_dist / 1.5))) 
        score = round(score, 2)

        is_verified = score > 0.4  # Threshold for "Valid"

        return json.dumps({
            "verified": is_verified,
            "verification_score": score,
            "evidence": best_doc[:400] + "..." # Snippet
        })

# Instantiate Global Client
mcp_client = MockMCPClient()

# Wrapper for AutoGen Tool Registration
def verify_limitation_tool(claim: str) -> str:
    return mcp_client.verify_claim(claim)


# ==========================================
# 3. PROMPTS (With Scoring Logic)
# ==========================================

GLOBAL_CONTEXT_NOTE = """
[SYSTEM NOTE]: The FULL CONTENT of the scientific paper is provided in the chat history. 
You strictly output JSON when requested.
"""

def get_agent_prompts():
    return {
        # --- THINKERS ---
        "Clarity": f"""You are the **Clarity Agent**. Identify clarity issues.
**FORMATTER RULE**: Output a strictly formatted JSON LIST.
JSON Schema:
[
  {{ "claim": "Brief limitation statement", "type": "Clarity", "severity": "High/Medium" }}
]
{GLOBAL_CONTEXT_NOTE}""",

        "Impact": f"""You are the **Impact Agent**. Identify novelty/significance issues.
**FORMATTER RULE**: Output a strictly formatted JSON LIST.
JSON Schema:
[
  {{ "claim": "Brief limitation statement", "type": "Impact", "severity": "High/Medium" }}
]
{GLOBAL_CONTEXT_NOTE}""",

        "Experiment": f"""You are the **Experiment Agent**. Identify flaws in design/metrics.
**FORMATTER RULE**: Output a strictly formatted JSON LIST.
JSON Schema:
[
  {{ "claim": "Brief limitation statement", "type": "Experiment", "severity": "High/Medium" }}
]
{GLOBAL_CONTEXT_NOTE}""",

        # --- VERIFIER ---
        "Verifier": f"""You are the **Verifier Agent**.
Your Goal: Validate hypotheses against the 'Cited Papers' knowledge base using the tool `verify_limitation_tool`.

1. Receive JSON lists from Clarity/Impact/Experiment.
2. For EACH claim, call `verify_limitation_tool(claim)`.
3. The tool returns a "verification_score" (0.0 to 1.0).
4. **CRITICAL**: Discard any claim with `verification_score < 0.4`.
5. Output the final validated list as a clean JSON, including the scores.

Example Output JSON:
[
  {{ "claim": "...", "score": 0.85, "evidence": "..." }}
]
{GLOBAL_CONTEXT_NOTE}""",

        # --- MASTER ---
        "Master": f"""You are the **Master Writer**. 
Synthesize the final limitations list from the Verified JSON.

**SCORING RULES**:
- **High Priority**: Limitations with `verification_score > 0.7`. These MUST be at the top.
- **Medium Priority**: Scores between 0.4 and 0.7.
- **Ignore**: Anything below 0.4 (if any slipped through).

Output Format:
"1. **[Type]** (Score: 0.XX): [Statement]. [Evidence snippet]."
{GLOBAL_CONTEXT_NOTE}""",

        # --- LEADER ---
        "Leader": f"""You are the **Leader**. Orchestrate the verification pipeline.
1. **Gather**: Ask Thinkers for JSON hypotheses.
2. **Verify**: Pass JSONs to **Verifier Agent** to check against the Vector DB.
3. **Write**: Pass the *Scored JSON* to **Master**. Ensure Master prioritizes high scores.
4. **Terminate**: Reply "TERMINATE" after the list is generated.
"""
    }

# ==========================================
# 4. AGENT INITIALIZATION
# ==========================================

def create_swarm():
    prompts = get_agent_prompts()
    
    leader = autogen.AssistantAgent(name="Leader_Agent", system_message=prompts["Leader"], llm_config=llm_config)
    clarity = autogen.AssistantAgent(name="Clarity_Agent", system_message=prompts["Clarity"], llm_config=llm_config)
    impact = autogen.AssistantAgent(name="Impact_Agent", system_message=prompts["Impact"], llm_config=llm_config)
    experiment = autogen.AssistantAgent(name="Experiment_Agent", system_message=prompts["Experiment"], llm_config=llm_config)
    verifier = autogen.AssistantAgent(name="Verifier_Agent", system_message=prompts["Verifier"], llm_config=llm_config)
    master = autogen.AssistantAgent(name="Master_Agent", system_message=prompts["Master"], llm_config=llm_config)

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config={"work_dir": "coding", "use_docker": False}, 
    )

    # Register the RAG Tool
    autogen.register_function(
        verify_limitation_tool,
        caller=verifier,
        executor=user_proxy,
        name="verify_limitation_tool",
        description="Checks the vector DB for evidence. Returns score (0-1) and snippet.",
    )

    all_agents = [user_proxy, leader, clarity, impact, experiment, verifier, master]
    
    groupchat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=30, 
        speaker_selection_method="auto", 
        allow_repeat_speaker=True
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return user_proxy, manager

# ==========================================
# 5. MAIN PIPELINE
# ==========================================

def run_pipeline():
    try:
        df = pd.read_csv(INPUT_CSV)
        # Use START_INDEX and END_INDEX as per your requirement

        df = df.iloc[START_INDEX:END_INDEX]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if "final_merged_limitations" not in df.columns:
        df["final_merged_limitations"] = "PENDING"

    # Initialize Agents (One-time setup)
    user_proxy, manager = create_swarm()

    print(f"Processing rows {START_INDEX} to {END_INDEX} with RAG Verification...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        
        # 1. Prepare Data
        # UPDATED: Using 'input_text' for the main paper content
        paper_text = str(row.get("input_text", ""))
        # UPDATED: Using 'cited_in' for the vector store content
        cited_text = str(row.get("cited_in", "")) 
        
        if len(paper_text) < 100:
            df.at[i, "final_merged_limitations"] = "SKIPPED_SHORT_TEXT"
            continue

        # 2. Update Vector DB for THIS specific paper [RAG Step]
        mcp_client.reset_knowledge_base(cited_text)

        # 3. Start Chat
        task_msg = f"""
        [DATA]
        Target Paper Content:
        '''{paper_text[:8000]}''' (Truncated if too long)
        
        [INSTRUCTIONS]
        1. Clarity/Impact/Experiment: Generate limitation hypotheses (JSON).
        2. Verifier: Verify these claims against the "Cited Papers" Knowledge Base using your tool.
        3. Master: Synthesize. give HIGHER WEIGHT to claims with high verification scores.
        """

        try:
            chat_result = user_proxy.initiate_chat(
                manager,
                message=task_msg,
                clear_history=True 
            )

            # Extract Master Output
            final_output = "NO_OUTPUT"
            for msg in reversed(chat_result.chat_history):
                if msg.get("name") == "Master_Agent" and msg.get("content"):
                    content = msg.get("content").strip()
                    if content != "TERMINATE":
                        final_output = content
                        break
            
            df.at[i, "final_merged_limitations"] = final_output
            df.at[i, "full_chat_history"] = str(chat_result.chat_history)
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            df.at[i, "final_merged_limitations"] = f"ERROR: {e}"

        if i % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False) 
            time.sleep(1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Pipeline Complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_pipeline()