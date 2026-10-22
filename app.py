import torch
from pathlib import Path
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Setup device and dynamic model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_summery_model"

# 2. Load model and tokenizer using Hugging Face AutoClasses
try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()
    print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# 3. Define inference function
def summarize(dialogue: str) -> str:
    if not dialogue or not dialogue.strip():
        return "Please enter a valid dialogue."
    if model is None or tokenizer is None:
        return "Model failed to load properly. Check space logs."

    input_text = "summarize: " + dialogue.strip()
    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            length_penalty=0.3,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Launch Gradio Interface
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=8, placeholder="Paste dialogue here...", label="Input Dialogue"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="TalkBrief: AI Summarizer",
    description="Summarize conversations using a fine-tuned T5 model."
)

if __name__ == "__main__":
    demo.launch()