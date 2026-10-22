import torch
from pathlib import Path
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ZeroGPU support: use @spaces.GPU if available, otherwise do nothing
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

# ---------------------------------------------------------------
# Settings you may need to change
# ---------------------------------------------------------------
# Set to True only if you trained with the "summarize: " prefix.
USE_PREFIX = False
PREFIX = "summarize: "

# Base model used only as a tokenizer fallback (match what you fine-tuned)
FALLBACK_TOKENIZER = "t5-small"
# ---------------------------------------------------------------

# 1. Setup device and model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_summery_model"


def load_tokenizer():
    """Try the local tokenizer first, then fall back to the base T5 tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)
    except Exception as e:
        print(f"Local fast tokenizer failed: {e}")
    try:
        return AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=False)
    except Exception as e:
        print(f"Local slow tokenizer failed: {e}")
    print(f"Falling back to tokenizer: {FALLBACK_TOKENIZER}")
    return AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)


# 2. Load model and tokenizer
try:
    tokenizer = load_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()
    print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


# 3. Inference function
@gpu_decorator
def summarize(dialogue: str) -> str:
    if not dialogue or not dialogue.strip():
        return "Please enter a valid dialogue."
    if model is None or tokenizer is None:
        return "Model failed to load properly. Check space logs."

    # Clean the input (remove placeholder tokens not in the T5 vocabulary)
    dialogue = dialogue.replace("<file_gif>", "").strip()

    input_text = (PREFIX + dialogue) if USE_PREFIX else dialogue

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            min_length=15,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. Gradio interface
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=8, placeholder="Paste dialogue here...", label="Input Dialogue"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="TalkBrief: AI Summarizer",
    description="Summarize conversations using a fine-tuned T5 model.",
)

if __name__ == "__main__":
    demo.launch()