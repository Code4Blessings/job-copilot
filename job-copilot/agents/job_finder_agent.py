import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Crew, Agent, Task
from textwrap import dedent
from tools.job_scraper import Job_Platform
from tools.ollama_llm import ask_ollama, system_prompt
from tools.url_builder import build_job_urls

# Define the Job Finder Agent
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

# Define the Job Finder Task (expects job_urls to be passed into the context)
job_finder_task = Task(
    description=dedent("""
        Go through the provided list of job URLs in the context. For each URL:
        1. Scrape the job listings.
        2. Generate a detailed summary using Ollama.
        3. Store any job match results and metadata (e.g., title, salary, remote type).

        Use a scoring method to determine the relevance based on a provided resume file.
        Filter for listings that score 60 or higher.
        Output your findings as a list of JSON-like objects to be passed to the next agent.
    """),
    expected_output=dedent("""
        A list of job matches (score >= 60) with summary, title, URL, and match score.
        These will be passed to the Resume Modifier agent.
    """),
    agent=job_finder
)

# Define Crew
crew = Crew(
    agents=[job_finder],
    tasks=[job_finder_task],
    verbose=True
)

# Run Crew
if __name__ == "__main__":
    # Assume another part of your app has generated this list from Agent 1 (Search Strategist)
    sample_inputs = {
        "job_title": "Software Engineer",
        "skills": ["Python", "JavaScript", "SQL"],
        "zip_code": "30045",
        "commute_miles": 25,
        "remote_preference": "remote"
    }

    job_urls = build_job_urls(
        sample_inputs["job_title"],
        sample_inputs["skills"],
        sample_inputs["zip_code"],
        sample_inputs["commute_miles"],
        sample_inputs["remote_preference"]
    )

    result = crew.kickoff(inputs = {"job_urls": job_urls})