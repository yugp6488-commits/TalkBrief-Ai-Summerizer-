import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Initialize FastAPI app
app = FastAPI(
    title="T5 Text Summarization API",
    description="FastAPI backend for fine-tuned T5 summarization model",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up dynamic paths for model and templates
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_summery_model"
TEMPLATES_DIR = BASE_DIR / "templates"

# Initialize Jinja2 Templates engine
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 2. Load model and tokenizer safely
try:
    tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH))
    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()
    print(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    tokenizer = None
    model = None
    print(f"CRITICAL: Failed to load model: {e}")


# 3. Model Inference Helper Function
def summarize_sample(dialogue: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("Model or Tokenizer is not loaded properly.")

    input_text = "summarize: " + dialogue

    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_beams=4,
            length_penalty=0.3,
            early_stopping=True,
        )

    return tokenizer.decode(targets[0], skip_special_tokens=True)


# ==========================================
# FRONTEND UI ROUTES (Jinja2 HTML Pages)
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Renders the blank webpage on initial load."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"summary": None, "original_text": ""}
    )


@app.post("/summarize-ui", response_class=HTMLResponse)
async def summarize_ui(request: Request, dialogue: str = Form(...)):
    """Handles HTML form submission, runs model, and renders results."""
    if not dialogue.strip():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"summary": "Please enter valid text.", "original_text": ""}
        )

    try:
        summary_result = summarize_sample(dialogue)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"summary": summary_result, "original_text": dialogue}
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"summary": f"Error generating summary: {str(e)}", "original_text": dialogue}
        )


# ==========================================
# BACKEND API ROUTE (Raw JSON for external apps)
# ==========================================

class SummaryRequest(BaseModel):
    dialogue: str

class SummaryResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_api(req: SummaryRequest):
    if not req.dialogue.strip():
        raise HTTPException(status_code=400, detail="Dialogue text cannot be empty.")
    
    try:
        summary_result = summarize_sample(req.dialogue)
        return SummaryResponse(summary=summary_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))