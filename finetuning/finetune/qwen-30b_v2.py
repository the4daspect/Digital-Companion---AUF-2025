import sys
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import wandb
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

# --- Configuration Section ---
# Set up paths relative to the script's location
parser = argparse.ArgumentParser(description="Finetune Qwen3.")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# Set model details
model_name_for_finetuning = "unsloth/Qwen3-30B-A3B-Instruct-2507"
max_seq_length = 2048

# WandB project details
wandb_project_name = "qwen3-30b-instruct-finetuning"
wandb_job_type = "training"

# Training parameters
r_lora = 32
lora_alpha = 32
learning_rate = 2e-4
num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
logging_steps = 5
eval_steps = 10
save_steps = 10
seed = 3407
# -----------------------------

# 1. Initialize WandB
wandb.init(project=wandb_project_name, job_type=wandb_job_type)

print(sys.executable)

# 2. Load the Qwen3 model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_for_finetuning,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=r_lora,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=seed,
)

# 3. Load and prepare the dataset
df = pd.read_csv(args.data_path)
df = df[['Context', 'Response']]

trainset, testset = train_test_split(df, test_size=0.057, random_state=42)

def format_chat_template(row):
    row_json = [
        {"role": "user", "content": row["Context"]},
        {"role": "assistant", "content": row["Response"]},
    ]
    # For Qwen3, the tokenizer handles the chat template and does not need
    # the replace(tokenizer.bos_token, "") call.
    row["text"] = tokenizer.apply_chat_template(
        row_json,
        tokenize=False,
        add_generation_prompt=False,
    )
    return row

formatted_trainset = trainset.apply(format_chat_template, axis=1)
formatted_testset = testset.apply(format_chat_template, axis=1)

hf_train = Dataset.from_pandas(formatted_trainset)
hf_test = Dataset.from_pandas(formatted_testset)

hf_train = hf_train.remove_columns(["Context", "Response"])
hf_test = hf_test.remove_columns(["Context", "Response"])

# 4. Configure and run the Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    packing=True,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="wandb",
        dataset_num_proc=16,
    )
)


trainer_stats = trainer.train()

# 5. Save the final model
save_directory = "trained_models/qwen3-30b-instruct"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Optional: Merge and save the final 16-bit model
save_path = Path(args.output_dir) / "final_model"
model.save_pretrained_merged(
    "trained_models/qwen3-30b-instruct-finetuned-16bit",
    tokenizer,
    save_method="merged_16bit",
)