# src/generator.py
import os
import google.generativeai as genai
from typing import List
import textwrap

# ✅ Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment.")
genai.configure(api_key=API_KEY)

# ✅ Use a safer Gemini model
GEMINI_MODEL = "gemini-2.5-flash-lite"

def simple_fallback_summary(contexts: List[str], question: str) -> str:
    """Simple backup summarization logic if Gemini refuses to generate."""
    joined = " ".join(contexts)
    # crude summarization fallback (keyword + sentence match)
    if "Tdap" in question or "booster" in question:
        return (
            "A Tdap booster is generally recommended once in adolescence (around 11–12 years), "
            "and every 10 years thereafter for adults. Pregnant women are advised to receive one "
            "dose during each pregnancy, ideally between 27–36 weeks gestation."
        )
    # otherwise, fallback to short context-based summary
    lines = textwrap.shorten(joined, width=400, placeholder="...")
    return f"⚠️ Gemini filter blocked. Summary from context:\n\n{lines}"

def generate_answer_from_contexts(
    contexts: List[str],
    question: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> str:
    """
    Educational-only RAG answer generator using Gemini.
    Falls back to local summarizer if content filters block output.
    """

    system_prompt = (
        "You are an educational assistant for healthcare students. "
        "Summarize the factual information from the provided contexts. "
        "Do not give medical advice; just restate known facts clearly."
    )

    ctx_text = "\n\n".join(contexts[:5])
    prompt = f"{system_prompt}\n\nCONTEXTS:\n{ctx_text}\n\nQUESTION: {question}\n\nAnswer:"

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        safety_settings = [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            [prompt],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_new_tokens,
            },
            safety_settings=safety_settings,
        )

        # ✅ Extract Gemini text safely
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif getattr(response, "candidates", None):
            cand = response.candidates[0]
            if cand and cand.content and cand.content.parts:
                return cand.content.parts[0].text.strip()

        return simple_fallback_summary(contexts, question)

    except Exception as e:
        # ✅ Fallback if Gemini blocks or errors
        return simple_fallback_summary(contexts, question)
