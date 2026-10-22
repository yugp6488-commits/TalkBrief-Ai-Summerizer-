<h1 <a href=https://huggingface.co/spaces/Preet1234/TalkBrief-AI-Summarizer>TalkBrief-Ai-Summerizer-></h1>

**TalkBrief** is an open-source NLP project designed to generate concise, abstractive summaries from unstructured human conversations and chat transcripts. Built on Google's **T5-small** model and fine-tuned on the **SAMSum dataset**, TalkBrief extracts core information from multi-turn dialogues accurately and efficiently.

---

## 📌 Key Features

* **Abstractive Dialogue Summarization:** Trained specifically on chat transcripts, messenger-style conversations, and meeting logs.
* **Lightweight & Efficient:** Fine-tuned on `t5-small` (~60M parameters), enabling quick CPU inference without requiring GPU hardware.
* **Interactive Web UI:** Built-in web interface for real-time text input and instant summary generation.
* **Git LFS Tracked:** Pre-trained model weights (`model.safetensors` ~242 MB) are safely version-controlled via Git LFS.

---

## 🧠 Model & Dataset Architecture

### 1. Base Model (`t5-small`)
The system uses Google's **Text-to-Text Transfer Transformer (T5)** framework, treating all NLP tasks as text generation problems.
* **Architecture:** Encoder-Decoder Transformer
* **Parameters:** ~60 Million
* **Task Prefix:** `"summarize: "` appended to input sequences during tokenization.

### 2. Dataset: SAMSum
Fine-tuned on the **SAMSum Dataset**, containing over 16,000 chat dialogues with human-written summaries:
* **`samsum-train.csv`**: Training split (~14,700 dialogues)
* **`samsum-validation.csv`**: Validation split (~818 dialogues)
* **`samsum-test.csv`**: Evaluation split (~819 dialogues)

---

## 📁 Repository Structure

```text
TalkBrief-Ai-Summerizer/
├── saved_summery_model/       # Fine-tuned T5 weights and tokenizer files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors     # 242 MB model weights (tracked via Git LFS)
│   ├── spiece.model           # SentencePiece tokenizer model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── app.py                     # Application script & web interface logic
├── requirements.txt           # Required Python packages
├── README.md                  # Project documentation
├── transformer_training.ipynb # Notebook containing model fine-tuning logic
├── samsum-train.csv           # Training dataset split
├── samsum-validation.csv      # Validation dataset split
├── samsum-test.csv            # Test dataset split
├── .gitattributes             # Git LFS tracking configuration
└── .gitignore                 # Excluded repository files
