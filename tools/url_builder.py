import urllib.parse

def build_job_urls(job_title, skills, zip_code, commute_range, work_preference):
    base_url = "https://www.dice.com/jobs"
    
    # Determine location logic
    if work_preference.lower() == "remote":
        location = "Remote"
        latitude = ""
        longitude = ""
    else:
        # For simplicity, we’ll use Georgia coords for now (future: plug in zip-to-coords API)
        location = "Georgia, USA"
        latitude = "32.1574351"
        longitude = "-82.90712300000001"

    # Encode keywords
    encoded_title = urllib.parse.quote_plus(job_title)
    encoded_skills = urllib.parse.quote_plus(" ".join(skills))

    # Format final URL query params
    query = f"q={encoded_title}+{encoded_skills}&location={urllib.parse.quote_plus(location)}"
    if latitude and longitude:
        query += f"&latitude={latitude}&longitude={longitude}&radius={commute_range}&radiusUnit=mi"

    full_url = f"{base_url}?{query}"

    return [full_url]
