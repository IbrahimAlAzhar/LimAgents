
import ast
import signal
import sys
import time
from typing import Any, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
Run the Analyzer agent with Mistral on pdf_text content.
Reads the balanced dataset, infers implicit limitations, and saves results to
zs_mistral_analyzer.csv.
"""

start_time = time.time()


# Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = "/lstr/sahara/datalab-ml/ibrahim/mistral_7b_v3_instruct"

print("Loading Mistral model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def truncate_prompt_for_model(prompt: str, max_length: int = 32000) -> str:
    """Truncate prompt to fit within the model context window."""
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > max_length:
        print(
            f"⚠️  Prompt token count = {tokens.shape[1]} exceeds limit "
            f"({max_length}). Truncating..."
        )
        truncated_tokens = tokens[:, :max_length]
        return tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
    return prompt


def mistral_generate(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate text using the Mistral model."""
    truncated_prompt = truncate_prompt_for_model(prompt, max_length=32000)
    messages = [{"role": "user", "content": truncated_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return response


def get_analyzer_prompt(paper_content: str) -> str:
    return f"""
You are a helpful research assistant. Given the content of a scientific paper, analyze and identify its limitations.
{paper_content} """


def parse_sections(sections: Any) -> List[str]:
    """Extract text from section dictionaries, preserving headings when available."""
    texts: List[str] = []
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                heading = section.get("heading")
                text = section.get("text", "")
                if heading:
                    texts.append(f"{heading}: {text}")
                else:
                    texts.append(text)
            else:
                texts.append(str(section))
    return [t for t in texts if t]


def pdf_text_to_article(pdf_entry: Any) -> str:
    """
    Convert a pdf_text entry (often stored as a string) into a single text blob.
    Uses abstractText plus concatenated section text.
    """
    if pd.isna(pdf_entry):
        return ""

    parsed: Any = pdf_entry
    if isinstance(pdf_entry, str):
        try:
            parsed = ast.literal_eval(pdf_entry)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not parse pdf_text at row {global_current_row}: {exc}")
            return pdf_entry

    # Sometimes the parsed object may be a list of dicts; fold them together
    if isinstance(parsed, list):
        combined_parts: List[str] = []
        for item in parsed:
            combined_parts.append(pdf_text_to_article(item))
        return "\n\n".join([part for part in combined_parts if part])

    if isinstance(parsed, dict):
        parts: List[str] = []
        abstract = parsed.get("abstractText") or parsed.get("abstract") or ""
        if abstract:
            parts.append(f"Abstract: {abstract}")
        sections = parsed.get("sections", [])
        section_texts = parse_sections(sections)
        if section_texts:
            parts.append("\n\n".join(section_texts))
        return "\n\n".join([part for part in parts if part])

    return str(parsed)


def run_analyzer_agent(paper_content: str) -> str:
    """Generate analyzer output for a single article."""
    print("  Running Analyzer agent...")
    try:
        prompt = get_analyzer_prompt(paper_content)
        response = mistral_generate(prompt, max_new_tokens=1024)
        print("  Analyzer agent completed")
        return response.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  Error in Analyzer agent: {exc}")
        return f"ERROR in Analyzer agent: {exc}"


print("Loading CSV file...")
try:
    df = pd.read_csv(
        "/lstr/sahara/datalab-ml/ibrahim/limagents_update/data/final_balanced_kde/df_balanced_kde_final.csv"
    )
    df = df.head(1)  # Uncomment to run a quick smoke test
    print(f"Successfully loaded CSV file with shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"Error loading CSV file: {exc}")
    sys.exit(1)

global_df = df
df["mistral_zs"] = ""

print("Processing samples with Analyzer agent using Mistral...")
for i, row in df.iterrows():
    global_current_row = i + 1

    print(f"\n=== Processing row {i+1}/{len(df)} ===")
    paper_content = pdf_text_to_article(row.get("pdf_text_without_gt", ""))

    zs_output = run_analyzer_agent(paper_content)
    df.at[i, "mistral_zs"] = zs_output
    print(f"  Row {i+1} completed")

    if i % 5 == 0:
        checkpoint_file = (
            "/lstr/sahara/datalab-ml/ibrahim/limagents_update/zero_shot/mistral/mistral_zs.csv"
        )
        df.to_csv(checkpoint_file, index=False)
        print(f"  ✅ Checkpoint saved at row {i+1}")

final_output = (
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/zero_shot/mistral/mistral_zs.csv"
)
df.to_csv(final_output, index=False)
print(f"\nResults saved to: {final_output}")

end_time = time.time()
elapsed = end_time - start_time
print(f"\n=== Analyzer Agent run complete ===")
print(f"Total samples processed: {len(df)}")
print("Agents used: Analyzer (Mistral-7B-Instruct-v0.3)")
print("Output column: mistral_analyzer")
print(f"Script started at: {time.ctime(start_time)}")
print(f"Script ended at:   {time.ctime(end_time)}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")