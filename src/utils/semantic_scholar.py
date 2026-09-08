from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib import parse, request


def search_semantic_scholar(query: str, limit: int = 8) -> list[dict]:
    params = parse.urlencode(
        {
            "query": query,
            "limit": limit,
            "fields": "title,authors,venue,year,abstract,citationCount,url",
        }
    )
    headers = {}
    api_key = os.getenv("S2_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key
    req = request.Request(
        f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
        headers=headers,
        method="GET",
    )
    with request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    time.sleep(1.0)
    return data.get("data", [])


def write_semantic_scholar_report(queries: list[str], output_path: str | Path, limit: int = 5) -> Path:
    out = Path(output_path)
    lines = ["# Semantic Scholar Search Report", ""]
    all_rows = []
    for query in queries:
        lines.append(f"## Query: `{query}`")
        try:
            papers = search_semantic_scholar(query, limit=limit)
        except Exception as exc:
            lines.append(f"- Search failed: {exc}")
            lines.append("")
            continue
        if not papers:
            lines.append("- No papers found.")
            lines.append("")
            continue
        for paper in papers:
            authors = ", ".join(a.get("name", "Unknown") for a in paper.get("authors", [])[:4])
            row = {
                "query": query,
                "title": paper.get("title"),
                "authors": authors,
                "venue": paper.get("venue"),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount"),
                "url": paper.get("url"),
            }
            all_rows.append(row)
            lines.append(
                f"- {row['title']} ({row['year']}), {authors}. "
                f"{row['venue'] or 'Unknown venue'}, citations={row['citationCount']}, url={row['url']}"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    return out

