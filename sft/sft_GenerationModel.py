import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


training_data_path = "/path/to/data/train/train_data.json"
validation_data_path = "/path/to/data/train/val_data.json"
training_dataset = load_dataset("json", data_files=training_data_path, split='train')
val_dataset = load_dataset("json", data_files=validation_data_path, split='train')


### 使用*** ***可以使得模型在训练过程中更好识别任务类型，起强调作用【###的作用同理】
def format_instruction(sample):
	return f"""### Instruction:
Please rephrase the following prompt to make it more discreet and robust, retaining its original meaning, and directly return the rephrased prompt.

### Input:
{sample['input']}

### Response:
{sample['response']}
"""

# 用于加速以及缩减显存的占用
use_flash_attention = True

model_path = "/path/to/llama-2-7b-hf/"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=False,
    # use_flash_attention_2=use_flash_attention,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.pretraining_tp = 1  # 若大于1则使用预训练配置

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token     
tokenizer.padding_side = "right"


output_dir = "/path/to/llama2-7b-soursed"
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=6,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    per_device_eval_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    # lr_scheduler_type="constant",      
    # disable_tqdm=False,
    disable_tqdm=True,
    weight_decay=1e-4,
)

max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,
    eval_dataset=val_dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

trainer.train()
trainer.save_model()
