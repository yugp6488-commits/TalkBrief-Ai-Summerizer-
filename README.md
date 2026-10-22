---
title: TalkBrief AI Summarizer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.21.0
app_file: app.py
pinned: false
---

# TalkBrief AI Summarizer 🚀

TalkBrief is an AI-powered web application designed to condense long dialogues, chats, and conversations into concise, readable summaries. The application is driven by a custom fine-tuned T5 transformer model and served through a high-performance FastAPI backend.

## 🧠 Model Architecture & Training

- **Base Model:** `t5-small` from Hugging Face.
- **Vocabulary & Embeddings:** The model's embedding layer was customized using a **32k vocabulary size** powered by **Word2Vec** representations.
- **Fine-Tuning:** The transformer was fine-tuned specifically on dialogue summarization datasets (like SAMSum) to understand conversational flow, context, and speaker intent.
- **Inference:** Uses PyTorch for dynamic tensor generation with parameters optimized for text summarization (`num_beams=4`, `length_penalty=0.3`, `early_stopping=True`).

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Machine Learning:** Hugging Face `transformers`, PyTorch, Word2Vec
- **Frontend:** Jinja2 Templates, HTML/CSS
- **Version Control:** Git, Git LFS (for large model weights)

## 📂 Repository Structure

```text
├── app.py                      # FastAPI application and routing
├── templates/
│   └── index.html              # Jinja2 frontend web interface
├── saved_summery_model/        # Fine-tuned model directory (Git LFS)
│   ├── model.safetensors       # 242MB trained weight parameters
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── generation_config.json
├── transformer_training.ipynb  # Jupyter notebook containing the fine-tuning pipeline
├── samsum-train.csv            # Training dataset
├── .gitattributes              # Git LFS tracking configuration
└── .gitignore                  # Ignored files (virtual envs, cache)
```
