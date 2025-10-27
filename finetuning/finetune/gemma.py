import sys
from unsloth import FastModel
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
from sklearn.model_selection import train_test_split
import wandb
from pathlib import Path

home_dir = parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data"

# 1. Initialize WandB
wandb.init(project="gemma-finetuning", job_type="training")

# Check Python environment
print(sys.executable)

# Configure PyTorch cache size
torch._dynamo.config.cache_size_limit = 64

# Load the model and tokenizer
fourbit_models = [
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
]

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
)

# Load and prepare the dataset
df = pd.read_csv(data_dir / "humanandllm.csv")
df = df[['Context', 'Response']]
print(df.head())

trainset, testset = train_test_split(df, test_size=0.057, random_state=42)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

def format_chat_template(row):
    row_json = [
        {"role": "user", "content": row["Context"]},
        {"role": "assistant", "content": row["Response"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    return row

formatted_trainset = trainset.apply(format_chat_template, axis=1)
formatted_testset = testset.apply(format_chat_template, axis=1)

hf_train = Dataset.from_pandas(formatted_trainset)
hf_test = Dataset.from_pandas(formatted_testset)

hf_train = hf_train.remove_columns(["Context", "Response"])
hf_test = hf_test.remove_columns(["Context", "Response"])

# 2. Configure the Trainer with WandB
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = hf_train,
    eval_dataset  = hf_test,
    args = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps                = 5,
        num_train_epochs          = 1,
        learning_rate               = 2e-4,
        logging_steps               = 5,
        eval_strategy               = "steps",
        eval_steps                  = 10,
        save_strategy               = "steps",
        save_steps                  = 10,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 42,
        report_to                   = "wandb", # Changed from "none" to "wandb"
        dataset_num_proc            = 2,
    )
)

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<start_of_turn>user\\n",
#     response_part = "<start_of_turn>model\\n",
# )

# Run the training
trainer_stats = trainer.train()

# Save the final model
trainer.save_model("trained_models/gemma3")

# Optional: Merge and save the final 16-bit model
model.save_pretrained_merged(
    "trained_models/gemma3-27b-finetuned-16bitv2",
    tokenizer,
    save_method = "merged_16bit",
)