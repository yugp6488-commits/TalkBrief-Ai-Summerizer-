import gradio as gr
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Setup device and paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_summery_model"

# 2. Load the model and tokenizer safely
try:
    tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the summarization function
def summarize(dialogue: str) -> str:
    if not dialogue.strip():
        return "Please enter a valid dialogue."

    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text, max_length=512, padding="max_length", 
        truncation=True, return_tensors="pt"
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

# 4. Build and launch the Gradio Interface
interface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=8, placeholder="Paste your conversation here...", label="Input Dialogue"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="TalkBrief: AI Summarizer",
    description="Summarize chats and conversations using a fine-tuned T5 model."
)

if __name__ == "__main__":
    interface.launch()