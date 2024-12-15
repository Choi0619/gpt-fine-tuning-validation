import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# Initialize WandB project
wandb.init(project='gyuhwan')  # Set project name as 'gyuhwan'
wandb.run.name = 'gpt-finetuning'  # Set WandB run name

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)  # Name of the HuggingFace model to be fine-tuned
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # Precision of the model
    dataset_name: Optional[str] = field(default=None)  # Name of the dataset from HuggingFace hub
    dataset_config_name: Optional[str] = field(default=None)  # Dataset configuration name
    block_size: int = field(default=1024)  # Input text length for fine-tuning
    num_workers: Optional[int] = field(default=None)  # Number of workers for data loading

parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# Logging setup
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

logger.info(f"Training/evaluation parameters {training_args}")

# Load dataset
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# Load model and tokenizer
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# Adjust tokenizer and model
tokenizer.pad_token_id = tokenizer.eos_token_id
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# Tokenization function
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output

# Tokenize dataset
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

# Adjust block size to fit input length
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# Function to group texts into blocks
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Group texts and create tokenized dataset
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# Split dataset into training and validation sets
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]  # Add validation dataset

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Set checkpoint
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

# Perform training and evaluation
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()

# Log training and evaluation results
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Evaluate with validation dataset
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

trainer.save_state()
