import os
import re
from datetime import datetime
from tools.ollama_llm import ask_ollama

def check_and_clean_resume(resume_text, job_title="reviewed"):
    system_prompt = """
    You are a professional resume editor. Your task is to improve grammar, clarity, and formatting.
    Do not change the meaning. Return only the revised resume in markdown format.
    """
    return ask_ollama(system_prompt, resume_text)

def clean_job_title(title):
    return re.sub(r'\W+', '_', title.lower().strip())[:30] or "untitled"

def check_resume_file(input_path, job_title_guess="reviewed"):
    with open(input_path, "r") as file:
        raw_text = file.read()

    cleaned_text = check_and_clean_resume(raw_text, job_title_guess)

    today_str = datetime.now().strftime("%m%d%y")
    base_title = clean_job_title(job_title_guess)
    new_filename = f"resume_{base_title}_{today_str}.md"
    output_path = os.path.join("checked_resumes", new_filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(cleaned_text)

    return new_filename, cleaned_text
