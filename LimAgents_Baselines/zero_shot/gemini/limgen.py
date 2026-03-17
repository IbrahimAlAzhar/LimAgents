import os
import pandas as pd
import time
import sys
import google.generativeai as genai
from tqdm import tqdm
import tiktoken

# ==========================================
# 1. CONFIGURATION
# ==========================================

# ✅ API Key Setup
os.environ["GEMINI_API_KEY"] = ""
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure the Gemini library
genai.configure(api_key=api_key)

# ✅ Model Selection
# Using the model ID you specified. 
# Ensure your API key has access to this specific preview model.
# MODEL_ID = "gemini-2.0-flash-exp" # Updated to a widely available experimental model, or use "gemini-1.5-pro-latest"
# If you specifically need "gemini-3-flash-preview", change the string below:
# MODEL_ID = "gemini-3-flash-preview" 
MODEL_ID = "gemini-2.5-flash" 

# Paths

os.makedirs(os.path.dirname(OUTPUT_SLICE), exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def truncate_text_to_tokens(text: str, max_tokens: int = 30000) -> str:
    """Truncates text to fit within context window limits safely."""
    if not text or pd.isna(text):
        return ""
    
    # Simple check to avoid crashing if tiktoken fails
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # print(f"  ⚠️ Truncating: {len(tokens)} -> {max_tokens} tokens")
        return encoding.decode(tokens[:max_tokens]) + "... [TRUNCATED]"
    except Exception:
        # Fallback character truncation if tokenizer fails
        return text[:max_tokens * 4]

# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    print("Loading CSV file...")
    try:
        df1 = pd.read_csv(INPUT_CSV)
        # Keeping your specific slice
        df = df1.iloc[100:200].copy()
        print(f"Processing {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Initialize Output Column
    if "final_generated_limitations" not in df.columns:
        df["final_generated_limitations"] = "PENDING"

    # Initialize Model
    # generation_config sets temperature etc.
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 8192,
    }
    
    try:
        model = genai.GenerativeModel(model_name=MODEL_ID, generation_config=generation_config)
    except Exception as e:
        print(f"Error initializing model {MODEL_ID}: {e}")
        return

    print("Starting generation...")
    
    # Iterate through the dataframe
    # We use tqdm for a progress bar
    for i in tqdm(range(len(df))):
        
        # Get the row specifically by integer location relative to the slice
        # Note: df is a slice, so we use iloc to access the i-th row of this slice
        
        # 1. Get Input
        raw_text = df.iloc[i].get("input_text_cleaned", "")
        paper_text = str(raw_text)

        # 2. Safety Truncation
        paper_text = truncate_text_to_tokens(paper_text, max_tokens=40000)

        if len(paper_text) < 100:
            df.iloc[i, df.columns.get_loc("final_generated_limitations")] = "SKIPPED_SHORT_TEXT"
            continue

        # 3. Construct the Single Prompt
        prompt = f"""Generate limitations from the research paper provided below.
        
        PAPER TEXT:
        {paper_text}"""

        # 4. Call Gemini API
        try:
            response = model.generate_content(prompt)
            
            # Extract text safely
            if response.text:
                output_text = response.text.strip()
            else:
                output_text = "NO_TEXT_GENERATED"
                
            # Update DataFrame
            df.iloc[i, df.columns.get_loc("final_generated_limitations")] = output_text

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            # print(error_msg)
            df.iloc[i, df.columns.get_loc("final_generated_limitations")] = error_msg
            time.sleep(5) # Longer wait on error

        # 5. Rate Limiting (Crucial for free tier or high volume)
        time.sleep(2) 

        # 6. Periodic Save (every 10 rows)
        if i % 10 == 0:
            df.to_csv(OUTPUT_SLICE, index=False)

    # Final Save
    df.to_csv(OUTPUT_SLICE, index=False)
    print(f"Completed. Saved to {OUTPUT_SLICE}")

if __name__ == "__main__":
    run_pipeline()