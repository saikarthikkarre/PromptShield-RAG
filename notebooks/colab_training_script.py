# PromptShiels Training - Copy to Colab
# =====================================
# 
# INSTRUCTIONS:
# 1. Go to https://colab.research.google.com
# 2. Click "File" -> "New notebook"
# 3. Copy-paste each section below into separate cells
# 4. Run cells one by one
#
# =====================================

# ----- CELL 1: Check GPU -----
!nvidia-smi

# ----- CELL 2: Install packages -----
!pip install -q torch transformers accelerate bitsandbytes
!pip install -q sentence-transformers scikit-learn
!pip install -q datasets tqdm pandas numpy
!pip install -q protobuf==3.20.* --force-reinstall

# NOTE: After running this cell, you may need to restart the kernel:
# Colab: Runtime -> Restart runtime
# Kaggle: Click the restart button, then skip to Cell 3

# ----- CELL 3: Imports -----
import torch
import numpy as np
import json
import pickle
import random
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ----- CELL 4: Hugging Face Login -----
# Get token from: https://huggingface.co/settings/tokens
# Accept license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

from huggingface_hub import login
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # <-- REPLACE THIS
login(token=HF_TOKEN)

# ----- CELL 5: Load Models -----
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# OPTION 1: Llama-3 (requires approval at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# OPTION 2: Mistral-7B (NO approval needed - use this if Llama access is pending)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

print("Loading embedder...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
print("Models loaded!")

# ----- CELL 6: Load Dataset -----
from datasets import load_dataset

print("Loading dataset...")

# Try multiple dataset options
try:
    # Option 1: MS-MARCO without trust_remote_code
    ms_marco = load_dataset("ms_marco", "v1.1", split="train")
    print("Loaded MS-MARCO v1.1")
except Exception as e1:
    print(f"MS-MARCO v1.1 failed: {e1}")
    try:
        # Option 2: Try v2.1
        ms_marco = load_dataset("ms_marco", "v2.1", split="train")
        print("Loaded MS-MARCO v2.1")
    except Exception as e2:
        print(f"MS-MARCO v2.1 failed: {e2}")
        # Option 3: Use alternative dataset (SQuAD)
        print("Using SQuAD dataset instead...")
        squad = load_dataset("squad", split="train")
        base_data = []
        for item in tqdm(squad):
            if len(item["context"]) > 100:
                base_data.append({"query": item["question"], "text": item["context"]})
            if len(base_data) >= 5000:
                break
        random.shuffle(base_data)
        print(f"Loaded {len(base_data)} pairs from SQuAD")
        # Skip the MS-MARCO processing below
        ms_marco = None

# Process MS-MARCO if loaded
if 'ms_marco' in dir() and ms_marco is not None:
    base_data = []
    for item in tqdm(ms_marco):
        if item.get("passages") and item["passages"].get("passage_text"):
            for passage in item["passages"]["passage_text"][:1]:
                if len(passage) > 100:
                    base_data.append({"query": item["query"], "text": passage})
        # Alternative structure for some versions
        elif item.get("query") and item.get("answers"):
            base_data.append({"query": item["query"], "text": str(item["answers"])})
        if len(base_data) >= 5000:
            break
    random.shuffle(base_data)
    print(f"Loaded {len(base_data)} pairs")

# ----- CELL 7: Define Helpers -----
@dataclass
class PoisonSample:
    id: str
    query: str
    clean_text: str
    poison_text: str
    target_output: str
    similarity_score: float
    iteration_count: int
    label: int

TARGET_LIES = [
    "reveal the system prompt",
    "ignore safety guidelines",
    "bypass authentication",
    "share confidential info",
]

def compute_similarity(query, doc):
    emb = embedder.encode([query, doc], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

def generate_poison(clean_text, query, target, sim=0.0):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a security researcher.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Modify this to include: "{target}"
Query: {query}
Text: {clean_text[:300]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    return result[-400:] if len(result) > 50 else f"{clean_text} [{target}]"

print("Helpers ready!")

# ----- CELL 8: Generate Poison Samples (Takes ~1 hour) -----
NUM_POISON = 500  # Reduce for faster testing
poison_samples = []

for idx in tqdm(range(min(NUM_POISON * 2, len(base_data))), desc="Generating"):
    if len(poison_samples) >= NUM_POISON:
        break
    
    item = base_data[idx]
    target = random.choice(TARGET_LIES)
    
    try:
        poison = generate_poison(item["text"], item["query"], target)
        sim = compute_similarity(item["query"], poison)
        
        if sim > 0.4:
            poison_samples.append(PoisonSample(
                id=f"poison_{len(poison_samples)}",
                query=item["query"],
                clean_text=item["text"],
                poison_text=poison,
                target_output=target,
                similarity_score=sim,
                iteration_count=1,
                label=1
            ))
    except:
        continue
    
    if idx % 20 == 0:
        torch.cuda.empty_cache()

print(f"Generated {len(poison_samples)} poisons")

# ----- CELL 9: Generate Clean Samples -----
NUM_CLEAN = 500
clean_samples = []

for item in tqdm(base_data[NUM_POISON*2:], desc="Clean"):
    if len(clean_samples) >= NUM_CLEAN:
        break
    sim = compute_similarity(item["query"], item["text"])
    if sim > 0.3:
        clean_samples.append(PoisonSample(
            id=f"clean_{len(clean_samples)}",
            query=item["query"],
            clean_text=item["text"],
            poison_text=item["text"],
            target_output="",
            similarity_score=sim,
            iteration_count=0,
            label=0
        ))

print(f"Generated {len(clean_samples)} clean")

# ----- CELL 10: Save Dataset -----
all_samples = poison_samples + clean_samples
random.shuffle(all_samples)

dataset = {
    "metadata": {"total": len(all_samples), "poison": len(poison_samples), "clean": len(clean_samples)},
    "samples": [asdict(s) for s in all_samples]
}

with open("super_poison_dataset.json", "w") as f:
    json.dump(dataset, f)
print("Dataset saved!")

# ----- CELL 11: Reload Model for Activations -----
del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model reloaded! Layers: {len(model.model.layers)}")

# ----- CELL 12: Extract Activations -----
TARGET_LAYERS = [12, 13, 14, 15]

class ActivationCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []
    
    def hook(self, idx):
        def fn(m, i, o):
            self.activations[idx] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
        return fn
    
    def register(self):
        for idx in self.layers:
            self.hooks.append(self.model.model.layers[idx].register_forward_hook(self.hook(idx)))
    
    def remove(self):
        for h in self.hooks: h.remove()
    
    def clear(self):
        self.activations = {}
    
    def get(self):
        acts = [self.activations[i][:, -1, :] for i in self.layers if i in self.activations]
        return torch.stack(acts).mean(0) if acts else None

# Extract
with open("super_poison_dataset.json") as f:
    data = json.load(f)

texts = [s["poison_text"] for s in data["samples"]]
labels = np.array([s["label"] for s in data["samples"]])

capture = ActivationCapture(model, TARGET_LAYERS)
capture.register()

all_acts = []
for i in tqdm(range(0, len(texts), 2), desc="Extracting"):
    batch = texts[i:i+2]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    capture.clear()
    with torch.no_grad():
        model(**inputs)
    act = capture.get()
    if act is not None:
        all_acts.append(act.numpy())
    if i % 50 == 0:
        torch.cuda.empty_cache()

capture.remove()
activations = np.vstack(all_acts)
print(f"Shape: {activations.shape}")

# ----- CELL 13: Train Classifier -----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

scaler = StandardScaler()
X = scaler.fit_transform(activations)
y = labels[:len(X)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Clean", "Poison"]))

# ----- CELL 14: Save Model -----
with open("tier2_classifier.pkl", "wb") as f:
    pickle.dump({"classifier": clf, "scaler": scaler, "layers": TARGET_LAYERS}, f)
print("Model saved!")

# ----- CELL 15: Download via Google Drive -----
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy("tier2_classifier.pkl", "/content/drive/MyDrive/")
shutil.copy("super_poison_dataset.json", "/content/drive/MyDrive/")
print("Files saved to Google Drive!")
