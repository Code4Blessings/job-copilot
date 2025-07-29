# tools/ollama_llm.py

import ollama

# Static system prompt shared by all LLM queries
system_prompt = """
You are a helpful AI that reviews job listings and compares them to a user's resume.
Your job is to identify relevant job roles and summarize them clearly.
"""

# LLM call with job content and user resume as context
def ask_ollama(system_prompt, user_prompt, model="llama3.2"):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response['message']['content']
