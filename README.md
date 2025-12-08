# LLM from Scratch

A clean and educational re-implementation of a **decoder-only Transformer (GPT architecture)** built entirely from scratch in **PyTorch**.  
This project aims to make the internal workings of modern language models transparent, lightweight, and easy to extend.

---

## 🔍 Project Summary

This repository provides:

- A **GPT-style decoder-only Transformer** featuring multi-head self-attention, feed-forward layers, residual connections, and layer normalization.
- A custom **training pipeline** with configurable hyperparameters and efficient GPU utilization.
- **OpenWebText preprocessing** including XZ extraction, training/validation split, and character-level vocabulary generation.
- **Memory-mapped dataset handling**, enabling training on text corpora larger than system RAM.
- **Checkpointing** through pickle for model persistence and continued training.
- A minimal **command-line chatbot** for testing and inference.

The model has been trained for **1000 epochs** on OpenWebText using an **NVIDIA RTX 4060**, showing that a fully functional LLM can be built from first principles.

---

## ⚙️ Key Characteristics

- **Framework:** PyTorch  
- **Architecture:** Decoder-only Transformer  
- **Optimizer:** AdamW  
- **Embedding Dimension:** 384  
- **Heads:** 8  
- **Layers:** 8  
- **Context Length:** 256 tokens  
- **Training Objective:** Autoregressive next-token prediction  

Designed to be simple, transparent, and hackable.

---

## 📁 Repository Structure

- `model.py` — Transformer architecture  
- `train.py` — Training pipeline  
- `preprocess.py` — Dataset extraction & vocabulary creation  
- `dataset.py` — Memory-mapped loader  
- `chat.py` — Minimal CLI chatbot  

---

## 🚀 Intended Use

Ideal for:

- Students learning how LLMs work internally  
- Researchers experimenting with simplified Transformer designs  
- Developers exploring training dynamics or custom datasets  
- Anyone who wants a clear, minimal LLM codebase to build upon  

---

## 🧭 Future Extensions

- Subword tokenization (BPE/WordPiece)  
- Mixed-precision training (FP16/BF16)  
- KV-cache for faster inference  
- Attention optimizations (FlashAttention)  
- Web-based inference interface  

---

## Acknowledgements

Inspired by GPT-2, nanoGPT, and the OpenWebText project.
