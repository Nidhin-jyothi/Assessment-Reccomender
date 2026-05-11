import os
import re
import json
from pathlib import Path

def parse_markdown_trace(md_content, trace_id):
    messages = []
    expected_shortlist = []
    probes = {}
    
    # Split by turns
    turns = re.split(r'### Turn \d+', md_content)
    for turn in turns[1:]:
        # Extract User message
        user_match = re.search(r'\*\*User\*\*\s*> (.*)', turn)
        if user_match:
            messages.append({"role": "user", "content": user_match.group(1).strip()})
        
        # Extract Agent message (optional, used for probes)
        agent_match = re.search(r'\*\*Agent\*\*\s*(.*?)($|(?=_))', turn, re.DOTALL)
        if agent_match:
            content = agent_match.group(1).strip()
            # In eval.py, we only really need the user messages to replay.
            # But we can extract the expected behavior.
        
        # Probes
        if "vague_turn1" not in probes and "Turn 1" in turn:
            if "No recommendations this turn" in turn:
                probes["vague_turn1"] = True
        
        # Extract URLs from tables
        urls = re.findall(r'<(https://www\.shl\.com/products/product-catalog/view/.*?)>', turn)
        if urls:
            expected_shortlist = urls # Keep updating, the last one is the final shortlist

    return {
        "id": trace_id,
        "expected_shortlist": list(dict.fromkeys(expected_shortlist)), # Unique URLs
        "messages": messages,
        "probes": probes
    }

def main():
    md_dir = Path("GenAI_SampleConversations")
    out_dir = Path("traces")
    out_dir.mkdir(exist_ok=True)
    
    for md_file in md_dir.glob("*.md"):
        trace_id = md_file.stem
        with md_file.open(encoding="utf-8") as f:
            md_content = f.read()
        
        trace_json = parse_markdown_trace(md_content, trace_id)
        
        with open(out_dir / f"{trace_id}.json", "w", encoding="utf-8") as f:
            json.dump(trace_json, f, indent=2)
        print(f"Converted {md_file.name} -> traces/{trace_id}.json")

if __name__ == "__main__":
    main()
