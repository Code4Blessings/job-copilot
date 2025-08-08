# tools/resume_checker.py

from tools.ollama_llm import ask_ollama

def check_and_clean_resume(resume_text: str) -> str:
    system_prompt = (
        "You are a professional resume editor. "
        "Your job is to correct grammar, fix formatting, and improve clarity without changing the meaning."
    )
    user_prompt = f"""Review the following resume and fix any grammatical or formatting issues. Do not invent content.

--- Resume Content ---
{resume_text}
"""
    return ask_ollama(system_prompt, user_prompt)


def check_and_clean_resume_with_notes(resume_text: str, notes: str = "") -> str:
    system_prompt = (
        "You are a professional resume editor. "
        "Fix grammar, structure, and clarity. Pay attention to tone and presentation. Do not fabricate content."
    )
    user_prompt = f"""Resume to check and improve:
{resume_text}

Additional user feedback to consider:
{notes}
"""
    return ask_ollama(system_prompt, user_prompt)
