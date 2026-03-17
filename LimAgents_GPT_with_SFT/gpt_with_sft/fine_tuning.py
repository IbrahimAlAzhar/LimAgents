import os 
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = ''

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_ID = "gpt-4o-mini" 

client = OpenAI()

train_file = client.files.create(
    file=open("/lstr/sahara/datalab-ml/ibrahim/limagents_update/llm_autogen_7_agents/gpt_with_sft/train.jsonl", "rb"),
    purpose="fine-tune",
)

print("training file id:", train_file.id) 

job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini-2024-07-18",
    training_file=train_file.id,
    suffix="lim_peer",
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {"n_epochs": 2}
        }
    }
)
print("job id:", job.id)

job = client.fine_tuning.jobs.retrieve(job.id)
print("status:", job.status)
print("fine_tuned_model:", job.fine_tuned_model)
