import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Task, Crew
from textwrap import dedent
from tools.resume_modifier import extract_resume_text

# Load environment variable for resume path
load_dotenv()
resume_path = os.getenv("RESUME_PATH")
resume_text = extract_resume_text()

# Sample job summary (replace this in the future with output from Job Finder Agent)
sample_job_summary = """
Seeking a Frontend Developer with strong experience in React.js, TypeScript, and responsive design.
Familiarity with Tailwind CSS, REST APIs, and Git workflows is a plus.
"""

# Define Resume Modifier Agent
resume_modifier = Agent(
    role="Resume Modifier",
    goal="Enhance the user's resume honestly based on the job summary",
    backstory=dedent("""
        You are an expert resume assistant that updates resumes to match job descriptions honestly.
        You never invent experience. You focus on emphasizing what's already present.
    """),
    verbose=True,
    allow_delegation=False
)

# Define Resume Modifier Task
resume_task = Task(
    description=dedent(f"""
        Your task is to enhance the resume below to match this job summary:
        {sample_job_summary}

        Only use what's already in the resume. No exaggeration or fabrication allowed.

        Resume:
        {resume_text}
    """),
    expected_output="Return the revised resume in clean markdown format.",
    agent=resume_modifier
)

# Run Crew
crew = Crew(
    agents=[resume_modifier],
    tasks=[resume_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n📝 Modified Resume Output:\n")
    print(result)
