def format_docs(docs):
    blocks = []

    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
        blocks.append(f"[{i}] {d.page_content}")
    
    return "\n\n".join(blocks)

def format_profile(profile: dict) -> str:
    """Format user profile into a readable text block."""
    lines = []

    for key, value in profile.items():
        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
            continue

        label = key.replace("_", " ").capitalize()
        lines.append(f"- {label}: {value}")

    return "User profile:\n" + "\n".join(lines) if lines else "User profile: (no details provided)"
