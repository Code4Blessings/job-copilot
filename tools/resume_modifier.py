import os
import fitz  # from pymupdf
from dotenv import load_dotenv
from tools.ollama_llm import ask_ollama, system_prompt

# Load environment variables from .env
load_dotenv()
RESUME_PATH = os.getenv("RESUME_PATH")

def extract_resume_text() -> str:
    """Extracts text from the resume PDF defined in .env."""
    if not RESUME_PATH or not os.path.exists(RESUME_PATH):
        raise FileNotFoundError(f"Resume file not found at {RESUME_PATH}")

    text = ""
    with fitz.open(RESUME_PATH) as doc:
        for page in doc:
            text += page.get_text()
    return text

def modify_resume_for_job(resume_text: str, job_summary: str) -> str:
    """Uses LLM to revise resume honestly to match the job description."""
    resume_text = extract_resume_text()

    prompt = f"""
    Based on the following job summary:

    {job_summary}

    Your task is to enhance the user's resume to better match this job, while staying 100% honest.
    Only use the content already present in the resume. Do NOT add new skills, experiences, or exaggerate.

    Focus on:
    - Reordering sections to highlight relevant skills first
    - Rewriting bullet points to emphasize matching skills
    - Highlighting tools or keywords from the job summary that exist in the resume
    - Keeping the resume in a professional, clean tone

    Here is the current resume:

    {resume_text}

    Return only the revised resume in clean markdown format.
    """
    return ask_ollama(prompt, system=system_prompt)
