# 🚀 LLM Fine-Tuning Lab: LoRA & QLoRA

Welcome to the **LLM Fine-Tuning Lab**. This project demonstrates advanced techniques for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models (LLMs) using **LoRA** and **QLoRA**.

## 🧠 Project Overview

This repository contains a full pipeline to fine-tune a Large Language Model on custom datasets while significantly reducing memory requirements and training time.

### Key Techniques Demonstrated:
- **LoRA (Low-Rank Adaptation)**: Freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer, reducing the number of trainable parameters by up to 10,000x.
- **QLoRA (Quantized LoRA)**: Further optimizes training by quantizing the pre-trained model to 4-bit precision, allowing for fine-tuning Billion-parameter models on consumer-grade GPUs.
- **PEFT (Parameter-Efficient Fine-Tuning)**: Utilizing the Hugging Face PEFT library for streamlined model adaptation.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Frameworks**: PyTorch, Hugging Face `transformers`, `peft`, `bitsandbytes`
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Chosen for its balance of performance and accessibility)
- **Dataset**: `mlabonne/guanaco-llama2-1k`

## ⚙️ Local Setup & Configuration

### 1. Prerequisites
- **NVIDIA GPU** with CUDA support is highly recommended (8GB+ VRAM).
- **Python 3.9+** installed.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ashokkumar261261/llm-finetuning-lab.git
cd llm-finetuning-lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -U trl accelerate # Essential for the training scripts
```

### 3. Usage: Running the Fine-Tuning
The project includes a master script that handles data loading, quantization, and LoRA adaptation.

```bash
python scripts/finetune.py
```

### 4. How it Works (Under the Hood)
1.  **Quantization**: The model is loaded in 4-bit precision using `BitsAndBytesConfig`.
2.  **Model Preparation**: We use `prepare_model_for_kbit_training` to make the quantized model compatible with gradient updates.
3.  **LoRA Config**: We define `LoraConfig` targeting the attention layers (typically the most impactful for adaptation).
4.  **Training**: We use the `SFTTrainer` (Supervised Fine-tuning Trainer) from the `trl` library to execute the training loop.

## 📊 Results & Artifacts
The fine-tuned model weights (adapters) are saved in the `tinyllama-lora-finetuned` directory. These adapters can be merged with the base model for inference.

## 🛡️ License
MIT License - Created for educational and demonstration purposes.
