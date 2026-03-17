import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = ''

# Initialize OpenAI Client for the Reward Model
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class RAGMemory:
    def __init__(self, db_path="dpo_faiss_index"):
        print("Initializing RAG Memory (HuggingFace Embeddings)...")
        # Efficient embedding model for scientific text
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.db = None
        self.db_path = db_path
        
        # Try to load existing DB if resuming
        if os.path.exists(db_path) and os.path.exists(f"{db_path}/index.faiss"):
            try:
                self.db = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
                print("Loaded existing vector database.")
            except:
                print("Could not load existing DB, starting fresh.")

    def add_winner(self, paper_text, winning_solution, score):
        """Stores the winning solution associated with the paper context."""
        doc = Document(
            page_content=paper_text[:2500], # Index the first 2500 chars of the paper
            metadata={
                "gold_limitations": winning_solution,
                "score": score
            }
        )
        
        if self.db is None:
            self.db = FAISS.from_documents([doc], self.embeddings)
        else:
            self.db.add_documents([doc])
        
        # Save to disk to prevent data loss
        self.db.save_local(self.db_path)

    def get_gold_standard(self, current_paper_text, k=1):
        """Retrieves the best winning example from previous runs."""
        if self.db is None:
            return ""
            
        # Search for similar papers
        docs = self.db.similarity_search(current_paper_text[:2500], k=k)
        if not docs:
            return ""
            
        best_match = docs[0]
        return f"""
*** GOLD STANDARD EXAMPLE (From a similar paper) ***
The following list of limitations received a high quality score ({best_match.metadata['score']}/70). 
Use the tone, specificity, and critical depth of this example as a guide:

{best_match.metadata['gold_limitations']}
*** END EXAMPLE ***
"""

def grade_solutions(paper_content, sol_a, sol_b):
    """
    Uses GPT-4o-mini to grade two solutions based on scientific criteria.
    """
    prompt = f"""
You are an expert scientific evaluator. Compare two lists of limitations (Solution A and Solution B) for the provided paper.

CRITERIA FOR SCORING (0-10 points each):
1. **Novelty/Originality**: Does it find unique flaws?
2. **Technical Correctness**: Are the critiques scientifically valid?
3. **Clarity**: Is it well-written?
4. **Experimental Rigor**: Does it catch experimental flaws?
5. **Motivation**: Does it critique the paper's placement in literature?
6. **Impact**: Does it assess interest to ICLR attendees?
7. **Support**: Are the claims supported by evidence from the text?

PAPER CONTEXT:
{paper_content[:3000]}... [Truncated]

=== SOLUTION A ===
{sol_a}

=== SOLUTION B ===
{sol_b}

OUTPUT FORMAT (JSON):
{{
  "A_total": <sum_of_points>,
  "B_total": <sum_of_points>,
  "Winner": "A" or "B",
  "Reason": "Brief explanation of why the winner is better."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Scoring Error: {e}")
        # Default fallback
        return {"Winner": "A", "A_total": 0, "B_total": 0, "Reason": "Error"}