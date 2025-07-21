import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator

def preprocess_function(examples, tokenizer):
    """Preprocess the dataset by tokenizing prompts and responses."""
    inputs = [f"{prompt}\n" for prompt in examples["prompt"]]
    targets = [f"{completion}\n" for completion in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Constants
model_id = "Qwen/Qwen2.5-7B"
HF_TOKEN = "YOUR-TOKEN-HERE"
DATA_PATH = "/home/WhatMakesAGoodPrompt/finetuning-codes/data/formal_train_2500.csv"

accelerator = Accelerator()
device_map = {"": accelerator.process_index}

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.float16,
)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    token=HF_TOKEN,
    # attn_implementation="eager",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
model = model.to(accelerator.device)

# Load and preprocess dataset
dataset = load_dataset("csv", data_files=DATA_PATH)
tokenized_dataset = dataset["train"].train_test_split(test_size=0.1, random_state=42)
tokenized_dataset = tokenized_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Define training arguments for this process
training_args = TrainingArguments(
    output_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    dataloader_drop_last=True,
)

# Initialize SFTTrainer with unwrapped model
trainer = SFTTrainer(
    model=model,  # Unwrapped model; SFTTrainer will handle DDP
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    peft_config=peft_config,
)

print(device_map)
print(f'n_gpu: {training_args.n_gpu}; Mode: {training_args.parallel_mode}')
print(f'Num Processes: {accelerator.num_processes}; Device: {accelerator.device}; Process Index: {accelerator.process_index}')
print(f'Accel Type: {accelerator.distributed_type}')


trainer.train()
trainer.save_model("./output")