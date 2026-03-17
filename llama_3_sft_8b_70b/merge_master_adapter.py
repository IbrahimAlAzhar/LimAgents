import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_CACHE = "/lstr/sahara/datalab-ml/ibrahim/llama3_8b_instruct"  # where base is cached
ADAPTER_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/llama_3_sft_8b_70b/llama3_8b_master_qlora"
MERGED_DIR  = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/llama_3_sft_8b_70b/llama3_8b_master_merged"

os.makedirs(MERGED_DIR, exist_ok=True)

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=BASE_CACHE,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading adapter...")
peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)

print("Merging adapter into base...")
merged = peft_model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(MERGED_DIR, safe_serialization=True)

print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=BASE_CACHE, use_fast=True)
tok.save_pretrained(MERGED_DIR)

print("✅ Done. Merged model saved at:", MERGED_DIR)
