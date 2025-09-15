from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model
base_model_id = "google/gemma-2b-it"  # or whatever model you fine-tuned on

# Load base + adapter
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, "./gemma-motivation")

# Merge adapters into base model
model = model.merge_and_unload()

# Save full model
model.save_pretrained("./gemma-motivation-merged")
tokenizer = AutoTokenizer.from_pretrained("./gemma-motivation")
tokenizer.save_pretrained("./gemma-motivation-merged")

print("✅ Merged model saved to ./gemma-motivation-merged")
