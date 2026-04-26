"""Search GEO for systemic scleroderma datasets via NCBI E-utilities."""

import argparse
import json
import sys
import time
from urllib.parse import quote

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_geo(term: str, organism: str = "Homo sapiens",
               data_type: str | None = None, retmax: int = 50) -> list[str]:
    """Search GEO DataSets and return a list of GDS/GSE UIDs."""
    query_parts = [f'"{term}"']
    if organism:
        query_parts.append(f'"{organism}"[organism]')
    if data_type:
        query_parts.append(f'"{data_type}"[Filter]')
    query = " AND ".join(query_parts)

    params = {"db": "gds", "term": query, "retmax": retmax, "retmode": "json"}
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    count = data.get("esearchresult", {}).get("count", "0")
    print(f"Found {count} total results, returning top {len(id_list)}")
    return id_list


def fetch_summaries(uid_list: list[str]) -> list[dict]:
    """Fetch dataset summaries for a list of UIDs in batches."""
    results = []
    batch_size = 20
    for i in range(0, len(uid_list), batch_size):
        batch = uid_list[i:i + batch_size]
        params = {"db": "gds", "id": ",".join(batch), "retmode": "json"}
        resp = requests.get(ESUMMARY_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("result", {})
        for uid in batch:
            if uid in data:
                entry = data[uid]
                results.append({
                    "uid": uid,
                    "accession": entry.get("accession", ""),
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],
                    "organism": entry.get("taxon", ""),
                    "type": entry.get("gdstype", ""),
                    "n_samples": entry.get("n_samples", 0),
                    "gse": entry.get("gse", ""),
                    "platform": entry.get("gpl", ""),
                    "date": entry.get("pdat", ""),
                    "pubmed": entry.get("pubmedids", []),
                    "ftp": entry.get("ftplink", ""),
                })
        if i + batch_size < len(uid_list):
            time.sleep(0.4)  # respect NCBI rate limits
    return results


def print_results(results: list[dict]) -> None:
    """Print results as a formatted table."""
    for r in sorted(results, key=lambda x: x["n_samples"], reverse=True):
        gse = f"GSE{r['gse']}" if r["gse"] else r["accession"]
        print(f"\n{'='*80}")
        print(f"  {gse}  |  {r['n_samples']} samples  |  {r['date']}")
        print(f"  {r['title']}")
        print(f"  Type: {r['type']}  |  Platform: GPL{r['platform']}")
        print(f"  {r['summary']}...")
        if r["pubmed"]:
            print(f"  PubMed: {', '.join(r['pubmed'])}")


def main():
    parser = argparse.ArgumentParser(description="Search GEO for scleroderma datasets")
    parser.add_argument("--term", default="scleroderma",
                        help="Search term (default: scleroderma)")
    parser.add_argument("--organism", default="Homo sapiens",
                        help="Organism filter (default: Homo sapiens)")
    parser.add_argument("--type", default=None, choices=[
                            "Expression profiling by array",
                            "Expression profiling by high throughput sequencing",
                        ], help="Data type filter")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"Searching GEO: term={args.term!r}, organism={args.organism!r}, type={args.type!r}")
    uid_list = search_geo(args.term, args.organism, args.type, args.max_results)
    if not uid_list:
        print("No results found.")
        sys.exit(0)

    print(f"Fetching summaries for {len(uid_list)} datasets...")
    results = fetch_summaries(uid_list)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
