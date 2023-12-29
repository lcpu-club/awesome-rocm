import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

from accelerate import Accelerator, DeepSpeedPlugin

# deepspeed needs to know your gradient accumulation steps beforehand, so don't forget to pass it
# Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=8)
accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)

base_model = "meta-llama/Llama-2-7b-chat-hf"
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

new_model = "llama-2-7b-chat-guanaco"

dataset = load_dataset(guanaco_dataset, split="train")

dtype = torch.float16

quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="adamw_torch", # adamw_hf
    save_steps=250,
    logging_steps=250,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

# Step 6: Save the model
trainer.save_model("llama-2-7b-chat-guanaco")