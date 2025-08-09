import sys
import os
import re
import json
from datetime import datetime
from textwrap import dedent

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from crewai import Crew, Agent, Task

# Project tools (assumed to exist in your repo)
from tools.url_builder import build_job_urls
from tools.job_scraper import Job_Platform
from tools.resume_modifier import extract_resume_text, modify_resume_for_job
from tools.ollama_llm import ask_ollama, system_prompt
# Robust import to handle different project layouts and the 'job-copilot' hyphenated folder
try:
    from agents.resume_checker_agent import check_resume_file  # if 'agents' is a package next to this file
except ModuleNotFoundError:
    import os, sys
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Try adding potential agent paths (with and without the hyphenated parent folder)
    candidate_paths = [
        os.path.join(BASE_DIR, "agents"),
        os.path.join(BASE_DIR, "job-copilot", "agents"),
        os.path.join(os.path.dirname(BASE_DIR), "agents"),
        os.path.join(os.path.dirname(BASE_DIR), "job-copilot", "agents"),
    ]
    for p in candidate_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from resume_checker_agent import check_resume_file  # import from the inserted path

# -----------------------------
# Helpers
# -----------------------------

def safe_guess_title_from_filename(fname: str) -> str:
    """Extract a human title from a filename like 'resume_frontend_engineer_080925.md'."""
    m = re.match(r"resume_(.+?)_\d{6}\.md$", fname)
    if not m:
        return "Reviewed"
    return m.group(1).replace("_", " ").title()


def try_parse_matches(blob) -> list:
    """Best-effort parser to normalize crew output into a list of dicts with keys title,url,score,summary."""
    if isinstance(blob, list):
        # Assume it's already a list of dict/tuples
        normalized = []
        for item in blob:
            if isinstance(item, dict):
                normalized.append({
                    "title": item.get("title") or item.get("job_title") or "Untitled",
                    "url": item.get("url") or item.get("link") or "",
                    "score": item.get("score") or item.get("match_score") or 0,
                    "summary": item.get("summary") or item.get("description") or ""
                })
            elif isinstance(item, (tuple, list)):
                title = item[0] if len(item) > 0 else "Untitled"
                summary = item[1] if len(item) > 1 else ""
                url = item[2] if len(item) > 2 else ""
                score = item[3] if len(item) > 3 else 0
                normalized.append({"title": title, "url": url, "score": score, "summary": summary})
        return normalized

    # If it's a string, try to extract JSON first
    if isinstance(blob, str):
        blob = blob.strip()
        # Direct JSON?
        try:
            data = json.loads(blob)
            return try_parse_matches(data)
        except Exception:
            pass
        # Look for JSON array within text
        m = re.search(r"(\[.*\])", blob, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                return try_parse_matches(data)
            except Exception:
                pass
        # Fallback: parse line-wise with simple heuristics
        lines = [ln.strip(" -•\t") for ln in blob.splitlines() if ln.strip()]
        results = []
        for ln in lines:
            # Example: Title | URL | score: 72
            title = ln
            url = ""
            score = 0
            # URL detection
            url_match = re.search(r"https?://\S+", ln)
            if url_match:
                url = url_match.group(0)
            # Score detection
            score_match = re.search(r"score\D+(\d+)", ln, flags=re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
            results.append({"title": title, "url": url, "score": score, "summary": ""})
        return results

    # Unknown type
    return []


def normalize_match_rows(matches: list) -> list:
    """Convert list of dicts into rows [title, url, score] for Gradio Dataframe."""
    rows = []
    for m in matches:
        rows.append([m.get("title", "Untitled"), m.get("url", ""), m.get("score", 0)])
    return rows


# -----------------------------
# Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 pipeline
# -----------------------------

def run_job_pipeline(job_title, skills, zip_code, commute_miles, remote_pref, resume_file=None):
    # --- Agent 1: Search Strategist ---
    search_strategist = Agent(
        role="Search Strategist",
        goal="Generate smart job search URLs based on the user's skills and preferences",
        backstory=dedent(
            """
            You are a proactive and resourceful AI agent who helps users find job opportunities by generating
            relevant job search URLs based on their input (skills, job title, location). You focus on speed,
            accuracy, and helping job seekers with minimal technical effort on their part.
            """
        ),
        verbose=True,
        allow_delegation=False,
    )

    strategist_task_description = dedent(
        f"""
        Based on the following user input:
        - Job Titles: {job_title}
        - Skills: {skills}
        - Zip Code: {zip_code}
        - Commute Radius: {commute_miles} miles
        - Remote Preference: {remote_pref}

        Use this information to generate job listing URLs from Dice.
        Return a list of valid, clean job search URLs tailored to the user's criteria.
        """
    )

    search_task = Task(
        description=strategist_task_description,
        expected_output="A list of job listing URLs to pass to the Job Finder agent.",
        agent=search_strategist,
    )

    crew1 = Crew(agents=[search_strategist], tasks=[search_task], verbose=True)
    job_urls = crew1.kickoff()

    # Guard: ensure we have URLs
    if not job_urls:
        return [], [], "⚠️ No job URLs were generated. Please adjust your inputs and try again."

    # --- Agent 2: Job Finder ---
    job_finder = Agent(
        role="Job Finder",
        goal="Analyze job listings and summarize relevant roles based on candidate's resume",
        backstory=dedent(
            """
            You are a highly skilled AI agent whose purpose is to search job listing platforms and analyze
            relevant job opportunities based on a candidate's experience, resume, and preferences.
            You return clean summaries for other agents to evaluate and act on.
            """
        ),
        verbose=True,
        allow_delegation=False,
    )

    finder_task_description = dedent(
        f"""
        Go through the following job URLs:
        {job_urls}

        1. Scrape the job listings.
        2. Generate a detailed summary using Ollama.
        3. Store any job match results and metadata (e.g., title, salary, remote type).

        Use a scoring method to determine the relevance based on a provided resume file.
        Filter for listings that score 60 or higher.
        Output your findings as a list of JSON-like objects to be passed to the Resume Modifier agent.
        """
    )

    job_finder_task = Task(
        description=finder_task_description,
        expected_output="A list of job matches (score >= 60) with summary, title, URL, and match score.",
        agent=job_finder,
    )

    crew2 = Crew(agents=[job_finder], tasks=[job_finder_task], verbose=True)
    job_matches_raw = crew2.kickoff()

    # Normalize matches
    matches = try_parse_matches(job_matches_raw)
    # Only keep score >= 60
    matches = [m for m in matches if (m.get("score") or 0) >= 60]

    # --- Agent 3: Resume Modifier ---
    if resume_file and hasattr(resume_file, "name"):
        resume_text = extract_resume_text(resume_file.name)
    else:
        # Fallback to your existing extractor behavior
        resume_text = extract_resume_text()

    os.makedirs("modified_resumes", exist_ok=True)
    os.makedirs("checked_resumes", exist_ok=True)

    modified_output_paths = []

    today_str = datetime.now().strftime("%m%d%y")

    for m in matches:
        title_raw = m.get("title") or "generic_job"
        summary = m.get("summary", "")
        job_title_clean = re.sub(r"\W+", "_", str(title_raw).lower())
        output_filename = f"resume_{job_title_clean}_{today_str}.md"
        output_path = os.path.join("modified_resumes", output_filename)

        modified_resume = modify_resume_for_job(resume_text, summary)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modified_resume)
        modified_output_paths.append(output_path)

    # Guard: if no modified resumes were created
    if not modified_output_paths:
        rows = normalize_match_rows(matches)
        return [], rows, "⚠️ No modified resumes were generated (no matches >= 60)."

    # --- Agent 4: Resume Checker ---
    checked_resume_paths = []
    for path in os.listdir("modified_resumes"):
        if path.endswith(".md"):
            input_path = os.path.join("modified_resumes", path)
            job_title_guess = safe_guess_title_from_filename(path)
            checked_filename, _notes = check_resume_file(input_path, job_title_guess)
            checked_resume_paths.append(os.path.join("checked_resumes", checked_filename))

    rows = normalize_match_rows(matches)

    status = (
        "✅ Modified and checked resumes are ready.\n\n"
        f"• Matches processed: {len(matches)}\n"
        f"• Files ready: {len(checked_resume_paths)}"
    )

    return checked_resume_paths, rows, status


# -----------------------------
# Approve / Reject handlers (Agent 5 stub)
# -----------------------------

def approve_resume(file_path: str):
    if not file_path:
        return "⚠️ No file selected."
    # TODO: Integrate Agent 5 – Application Submitter
    # For now, just log the action.
    try:
        fname = os.path.basename(file_path)
        # Example: enqueue to a queue/db or call your agent here
        # submit_application_with_resume(file_path)
        return f"✅ Approved and queued for Agent 5: {fname}"
    except Exception as e:
        return f"❌ Failed to queue for Agent 5: {e}"


def reject_resume(file_path: str):
    if not file_path:
        return "⚠️ No file selected."
    # TODO: Implement re-run logic: either call Agent 3 with new constraints or re-run Agent 4
    return (
        "♻️ Rejected. Next actions available: 1) Re-run Agent 4 with tweaks, "
        "2) Send to Agent 3 for deeper modification. (Handlers TBA)"
    )


def pick_first(files):
    # Gradio Files returns a list of dicts or paths depending on context
    if not files:
        return ""
    try:
        # If it's a list of dicts with 'name'
        first = files[0]
        if isinstance(first, dict) and "name" in first:
            return first["name"]
        # If it's a list of file paths
        if isinstance(first, str):
            return first
    except Exception:
        pass
    return ""


# -----------------------------
# Gradio UI
# -----------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Job Copilot – Full AI Job Search")
    gr.Markdown("Enter your job search preferences below:")

    with gr.Row():
        job_title = gr.Textbox(label="Job Titles (comma-separated)")
        skills = gr.Textbox(label="Skills (comma-separated)")
    with gr.Row():
        zip_code = gr.Textbox(label="Zip Code")
        commute_miles = gr.Slider(0, 100, step=5, label="Max Commute Distance (miles)")
        remote_pref = gr.Radio(["Remote Only", "Hybrid", "No Preference"], label="Remote Work Preference")

    resume_file = gr.File(label="(Optional) Upload Resume (PDF/DOCX/TXT)", file_types=[".pdf", ".docx", ".txt"], interactive=True)

    submit_btn = gr.Button("🔍 Run Full Search")

    checked_files = gr.Files(label="✅ Checked Resumes (downloadable)")
    match_table = gr.Dataframe(headers=["title", "url", "score"], label="Matched Jobs (score ≥ 60)", interactive=False)
    status = gr.Markdown()

    selected_file = gr.Textbox(label="Selected file path for action", interactive=True)
    with gr.Row():
        pick_btn = gr.Button("Pick Selected (from list above)")
        approve_btn = gr.Button("✅ Approve & Send to Agent 5")
        reject_btn = gr.Button("♻️ Reject (Revise)")

    pick_btn.click(fn=pick_first, inputs=[checked_files], outputs=[selected_file])

    submit_btn.click(
        fn=run_job_pipeline,
        inputs=[job_title, skills, zip_code, commute_miles, remote_pref, resume_file],
        outputs=[checked_files, match_table, status],
    )

    approve_btn.click(fn=approve_resume, inputs=[selected_file], outputs=[status])
    reject_btn.click(fn=reject_resume, inputs=[selected_file], outputs=[status])


if __name__ == "__main__":
    demo.launch()
