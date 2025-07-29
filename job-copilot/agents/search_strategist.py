import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crewai import Crew, Agent, Task
from textwrap import dedent
from tools.url_builder import build_job_urls  # This tool will do the URL generation

# Define the Search Strategist Agent
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

# Define the Search Strategist Task
search_task = Task(
    description=dedent("""
        Ask the user for their preferred job title, a short list of key skills, and job preferences:
        - Zip code
        - Are they willing to commute? If so, how many miles?
        - If not, do they prefer remote work or would a hybrid setup be acceptable?

        Then use this information to generate URLs for job listings from major job platforms like Dice.
        Return a list of clean, valid job search URLs that the Job Finder agent can use for scraping.
    """),
    expected_output=dedent("""
        A list of job listing URLs built using the user’s job title, skills, and location.
        These URLs should be passed to the Job Finder agent for analysis.
    """),
    agent=search_strategist
)

# Define Crew
crew = Crew(
    agents=[search_strategist],
    tasks=[search_task],
    verbose=True
)

# Run Crew
if __name__ == "__main__":
    crew.kickoff()
