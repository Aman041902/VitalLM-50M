# 🏥 VitalLM-50M: Medical-Domain Small Language Model (SLM)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/aman0419/Vitallm-50M)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aman0419/VitalLM-50M-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VitalLM-50M is a **50.55 million parameter** decoder-only Transformer built from scratch and specialized for the biomedical domain. It was trained on a filtered **764M token corpus** of clinical dialogues and medical research to provide high-density reasoning in a compact footprint.

---

## 🛠️ Key Architectural Features

* **Custom Transformer Engine**: Implemented a decoder-only architecture using PyTorch, featuring causal self-attention mechanisms.
* **SwiGLU Activation**: Utilized the **SwiGLU** (Swish-Gated Linear Unit) activation function, as used in state-of-the-art models like Llama 3, to improve non-linear clinical reasoning.
* **Specialized Tokenization**: Developed a custom **ByteLevelBPE** tokenizer with a 16,384 vocabulary size, specifically optimized for complex medical nomenclature (e.g., pharmacokinetics, pathophysiology).
* **Resource Optimized**: Built for edge deployment, utilizing **Weight Tying** to reduce the memory footprint while maintaining a context window of **256 tokens**.

---

## 📊 Performance & Training

The model was trained under strict compute constraints using a multi-session checkpointing strategy on an NVIDIA P100.

| Metric | Value |
| :--- | :--- |
| **Total Parameters** | 50,554,880 |
| **Layers / Heads / Dim** | 10 / 8 / 512 |
| **Training Loss** | 3.32 |
| **Validation Loss** | 3.66 |
| **Final Perplexity** | ~38.8 |

---

## 🚀 Quick Start (Inference)

To use VitalLM-50M, clone this repo and ensure you have `torch` and `transformers` installed.

### 🚀 Quick Start & Sample Inference
### 🚀 Quick Start & Sample Inference

To run the model, first clone this repository to get the custom architecture definitions:

```bash
git clone [https://github.com/YOUR_USERNAME/VitalLM-50M.git](https://github.com/YOUR_USERNAME/VitalLM-50M.git)
cd VitalLM-50M
pip install torch transformers huggingface_hub
```

Users can run the model directly by pulling the latest weights and configuration from the Hugging Face Hub. 
Then, run the following script (ensure model.py is in your current directory):
```python
import torch
import torch.nn.functional as F
from model import SLM, SLMConfig
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

# 1. Download files from Hugging Face
repo_id = "aman0419/Vitallm-50M" 
weights_path = hf_hub_download(repo_id=repo_id, filename="vital_lm_50m_weights.pt")
vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab_50m.json")
merges_path = hf_hub_download(repo_id=repo_id, filename="merges_50m.txt")

# 2. Initialize Architecture
config = SLMConfig(vocab_size=16384, n_layer=10, n_head=8, n_embd=512, block_size=256)
model = SLM(config)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# 3. Setup Tokenizer
tokenizer = PreTrainedTokenizerFast(
    vocab_file=vocab_path, merges_file=merges_path,
    eos_token="<|endoftext|>", pad_token="<|endoftext|>"
)

# 4. Sample Prompt & Generation Logic
prompt = "Patient: I have a persistent cough and high fever. Doctor:"
input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

print(f"Prompt: {prompt}")

with torch.no_grad():
    for _ in range(50): # Generate up to 50 new tokens
        # Ensure we don't exceed the context window
        logits, _ = model(input_ids[:, -256:])
        logits = logits[:, -1, :] / 0.7 # Temperature scaling
        
        # Sample the next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

print(f"Response: {tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)}")
```
## 🔬Challenges & Engineering Insights
* **Inference Optimization**: Faced challenges with semantic repetition loops. Resolved by implementing Repetition Penalty (1.2) and Nucleus (Top-P) Sampling in the deployment pipeline.

* **Compute Strategy**: Engineered a custom state-recovery system to manage multi-session training across 12-hour compute limits without loss spikes.

* **Data Quality**: Curated a biomedical-heavy subset from open-source clinical datasets, focusing on high-signal medical dialogues.

## ⚠️ Disclaimer
This model is for educational and research purposes only. It is not intended for clinical use or to provide professional medical advice.
