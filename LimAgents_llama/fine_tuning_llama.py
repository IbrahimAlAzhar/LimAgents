import os
import re
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# =========================
# 0) CONFIG
# =========================
CSV_PATH = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/llama_3_sft_8b_70b/train_test_data/df.csv"

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/lstr/sahara/datalab-ml/ibrahim/llama3_8b_instruct"
OUT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/llama_3_sft_8b_70b/llama3_8b_master_qlora"

# For 40GB GPU, 4096 is a safe starting point (increase only if you’re sure)
MAX_SEQ_LEN = 4096

NUM_EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
EVAL_RATIO = 0.1
SEED = 42

MASTER_SYSTEM_PROMPT = (
    "You are the **Master Agent**. Your role is to receive limitation analyses from multiple specialist agents "
    "(via the Leader Agent) and produce a single, final, high-quality, consolidated list of limitations for the scientific paper.\n"
    "TASK:\n"
    "- Carefully read and integrate all provided specialist outputs.\n"
    "- Remove redundancies (merge similar limitations).\n"
    "- Prioritize the most severe and well-justified limitations.\n"
    "- Preserve specificity and evidence from the original analyses.\n"
    "- Organize the final list logically (e.g., group by category).\n"
    "- Avoid introducing new limitations not raised by the specialists.\n"
    "OUTPUT FORMAT:\n"
    "Start with: \"Here is the consolidated list of key limitations identified in the paper:\" then a bulleted list.\n"
)

def _clean(x: str) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()

# =========================
# 1) LOAD + FILTER DATA
# =========================
df = pd.read_csv(CSV_PATH)

df = df[["input_text_cleaned", "master_sft_target"]].copy()
df["input_text_cleaned"] = df["input_text_cleaned"].fillna("").astype(str).map(_clean)
df["master_sft_target"] = df["master_sft_target"].fillna("").astype(str).map(_clean)

# Remove empty / ERROR targets
df = df[df["master_sft_target"].str.len() > 0]
df = df[~df["master_sft_target"].str.startswith("ERROR", na=False)]

assert len(df) > 0, "No valid rows left after filtering."

# Shuffle + split
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_eval = max(1, int(len(df) * EVAL_RATIO))
df_eval = df.iloc[:n_eval].copy()
df_train = df.iloc[n_eval:].copy()

train_ds = Dataset.from_pandas(df_train, preserve_index=False)
eval_ds  = Dataset.from_pandas(df_eval, preserve_index=False)

# =========================
# 2) TOKENIZER + FORMAT INTO A SINGLE 'text' FIELD
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_chat_text(example):
    user_content = f"PAPER CONTENT:\n{example['input_text_cleaned']}"
    messages = [
        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["master_sft_target"]},
    ]
    # Build a single training string using Llama-3 chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

train_ds = train_ds.map(to_chat_text, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(to_chat_text, remove_columns=eval_ds.column_names)

# =========================
# 3) LOAD BASE MODEL (4-bit) + QLoRA
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # bf16 if you want and your GPU supports it well
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False  # important for training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 4) TRL SFTConfig (THIS IS WHERE dataset_text_field GOES IN TRL 0.25.1)
# =========================
sft_args = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,

    # transformers>=4.57 uses eval_strategy (NOT evaluation_strategy)
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    optim="paged_adamw_32bit",
    fp16=True,
    bf16=False,

    gradient_checkpointing=True,
    max_grad_norm=1.0,
    report_to="none",
    seed=SEED,

    # SFT-specific (TRL)
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=True,
)

# =========================
# 5) TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,  # TRL 0.25.1 uses processing_class
)

trainer.train()

# Save adapter + tokenizer
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ Done. Saved QLoRA adapter to:", OUT_DIR)
