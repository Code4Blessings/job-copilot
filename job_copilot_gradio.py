import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from crewai import Crew, Agent, Task
from datetime import datetime
from textwrap import dedent
from tools.url_builder import build_job_urls
from tools.job_scraper import Job_Platform
from tools.resume_modifier import extract_resume_text, modify_resume_for_job
from tools.ollama_llm import ask_ollama, system_prompt

# Function to run the full pipeline: Search Strategist -> Job Finder -> Resume Modifier

def run_job_pipeline(job_title, skills, zip_code, commute_miles, remote_pref):
    # --- Agent 1: Search Strategist ---
    search_strategist = Agent(
        role="Search Strategist",
        goal="Generate smart job search URLs based on the user's skills and preferences",
        backstory=dedent("""
            You are a proactive and resourceful AI agent who helps users find job opportunities by generating
            relevant job search URLs based on their input (skills, job title, location). You focus on speed,
            accuracy, and helping job seekers with minimal technical effort on their part.
        """),
        verbose=True,
        allow_delegation=False
    )

    strategist_task_description = dedent(f"""
        Based on the following user input:
        - Job Titles: {job_title}
        - Skills: {skills}
        - Zip Code: {zip_code}
        - Commute Radius: {commute_miles} miles
        - Remote Preference: {remote_pref}

        Use this information to generate job listing URLs from Dice.
        Return a list of valid, clean job search URLs tailored to the user's criteria.
    """)

    search_task = Task(
        description=strategist_task_description,
        expected_output="A list of job listing URLs to pass to the Job Finder agent.",
        agent=search_strategist
    )

    crew1 = Crew(agents=[search_strategist], tasks=[search_task], verbose=True)
    job_urls = crew1.kickoff()

    # --- Agent 2: Job Finder ---
    job_finder = Agent(
        role="Job Finder",
        goal="Analyze job listings and summarize relevant roles based on candidate's resume",
        backstory=dedent("""
            You are a highly skilled AI agent whose purpose is to search job listing platforms and analyze
            relevant job opportunities based on a candidate's experience, resume, and preferences.
            You return clean summaries for other agents to evaluate and act on.
        """),
        verbose=True,
        allow_delegation=False
    )

    finder_task_description = dedent(f"""
        Go through the following job URLs:
        {job_urls}

        1. Scrape the job listings.
        2. Generate a detailed summary using Ollama.
        3. Store any job match results and metadata (e.g., title, salary, remote type).

        Use a scoring method to determine the relevance based on a provided resume file.
        Filter for listings that score 60 or higher.
        Output your findings as a list of JSON-like objects to be passed to the Resume Modifier agent.
    """)

    job_finder_task = Task(
        description=finder_task_description,
        expected_output="A list of job matches (score >= 60) with summary, title, URL, and match score.",
        agent=job_finder
    )

    crew2 = Crew(agents=[job_finder], tasks=[job_finder_task], verbose=True)
    job_matches = crew2.kickoff()

    # --- Resume Modifier Phase (loop through matches and save output) ---
    resume_text = extract_resume_text()
    modified_output_paths = []

    # Today's date string: MMDDYY
    today_str = datetime.now().strftime("%m%d%y")

    for match in job_matches:
        # If match is a dictionary (preferred format)
        if isinstance(match, dict):
            title_raw = match.get("title") or "generic_job"
            summary = match.get("summary", "")
        # If match is a tuple (fallback format)
        elif isinstance(match, tuple) and len(match) >= 2:
            title_raw, summary = match[:2]
        else:
            continue  # Skip this match if structure is unknown

        # Clean title for filename
        job_title_clean = re.sub(r'\W+', '_', title_raw.lower())
        output_filename = f"resume_{job_title_clean}_{today_str}.md"
        output_path = os.path.join("modified_resumes", output_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        modified_resume = modify_resume_for_job(resume_text, summary)
        with open(output_path, "w") as f:
            f.write(modified_resume)


    return f"Modified resumes saved:\n" + "\n".join(modified_output_paths)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Job Copilot – Full AI Job Search")
    gr.Markdown("Enter your job search preferences below:")

    job_title = gr.Textbox(label="Job Titles (comma-separated)")
    skills = gr.Textbox(label="Skills (comma-separated)")
    zip_code = gr.Textbox(label="Zip Code")
    commute_miles = gr.Slider(0, 100, step=5, label="Max Commute Distance (miles)")
    remote_pref = gr.Radio([
        "Remote Only",
        "Hybrid",
        "No Preference"
    ], label="Remote Work Preference")

    submit_btn = gr.Button("🔍 Run Full Search")
    output = gr.Textbox(label="System Output", lines=15)

    submit_btn.click(
        fn=run_job_pipeline,
        inputs=[job_title, skills, zip_code, commute_miles, remote_pref],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
