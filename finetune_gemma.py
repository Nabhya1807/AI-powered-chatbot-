from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json

# 1. Load training data
with open("data.jsonl", "r") as f:
    raw_data = [json.loads(line) for line in f]

def format_example(example):
    return {
        "text": f"<start_of_turn>user\n{example['prompt']}<end_of_turn>\n<start_of_turn>model\n{example['response']}<end_of_turn>"
    }

formatted_data = [format_example(x) for x in raw_data]
dataset = Dataset.from_list(formatted_data)

# 2. Tokenizer and model
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_fn)

# 4. Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# 5. Training
training_args = TrainingArguments(
    output_dir="./gemma-motivation",
    per_device_train_batch_size=2,
    num_train_epochs=3,  # You can increase this
    save_strategy="epoch",
    logging_steps=10,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# 6. Save model
model.save_pretrained("./gemma-motivation")
tokenizer.save_pretrained("./gemma-motivation")
