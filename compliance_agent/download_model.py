from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import os
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU found  : {gpu_name}")
    print(f"✅ VRAM total : {vram:.1f} GB")
else:
    print("⚠️ No GPU found — model will run on CPU")
os.makedirs("./models", exist_ok=True)

model_path = snapshot_download(
    repo_id  = "Qwen/Qwen3-8B-Instruct",
    local_dir= "./models/qwen3-8b-instruct"
)
required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for f in required:
    exists = f in files_in_folder
    print(f"✅ {f}" if exists else f"❌ {f}")

weight_files = [f for f in files_in_folder if f.endswith(".safetensors")]
print(f"✅ Weight files : {len(weight_files)} .safetensors files")
tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen3-8b-instruct",
    trust_remote_code=True
)

test_sentence = "Customer must provide government ID for KYC verification."
tokens = tokenizer(test_sentence)
print(f"✅ Token count : {len(tokens['input_ids'])} tokens")
