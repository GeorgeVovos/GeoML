"""Download and parse GEO datasets into expression matrices + metadata CSVs."""

import argparse
import json
import sys
from pathlib import Path

import GEOparse
import pandas as pd

# Curated GSE accessions for systemic scleroderma ML, ordered by priority
CURATED_GSE = [
    # Tier 1 — Large
    "GSE76809",   # 577 samples, multi-tissue SSc
    "GSE134310",  # 229 samples, SCOT trial PBMCs
    "GSE130953",  # 205 samples, SCOT trial whole blood
    "GSE76885",   # 194 samples, SSc skin trajectory during MMF
    "GSE33463",   # 140 samples, PAH subtypes PBMCs
    "GSE45536",   # 123 samples, plasma cell / anti-CD19
    "GSE58095",   # 102 samples, SSc skin heterogeneity
    "GSE201405",  # 36 samples, morphea vs SSc skin (microarray)
    # Tier 2 — Medium
    "GSE32413",   # 89 samples, intrinsic subsets + rituximab
    "GSE59785",   # 82 samples, mycophenolate treatment skin
    "GSE45485",   # ~82 samples, molecular signatures MMF
    "GSE9285",    # 75 arrays, skin biopsies (landmark)
    "GSE22356",   # 38 samples, SSc-PAH PBMCs
    "GSE95065",   # 33 samples, SSc skin biopsies
    "GSE12493",   # 40 samples, TGFb fibroblasts
    "GSE40839",   # 21 samples, SSc-ILD lung fibroblasts
    "GSE138669",  # 22 samples, scRNA-seq myofibroblast progenitors
    # Tier 3 — Smaller/specialized
    "GSE125362",  # 12 samples, ML classifier intrinsic subsets
    "GSE4385",    # 33 samples, classic SSc skin microarray
    "GSE86984",   # 8 samples, monocytes
    "GSE144625",  # 6 samples, Fli1 myeloid cells
    "GSE27165",   # 12 samples, Egr-1 fibroblasts
    "GSE1724",    # 18 samples, TGFb lung fibroblasts
]

DATA_DIR = Path("data")


def download_gse(gse_id: str, output_dir: Path, silent: bool = False) -> None:
    """Download a single GSE dataset and save expression matrix + metadata."""
    series_dir = output_dir / gse_id
    series_dir.mkdir(parents=True, exist_ok=True)

    expr_path = series_dir / f"{gse_id}_expression_matrix.csv"
    meta_path = series_dir / f"{gse_id}_metadata.csv"
    info_path = series_dir / f"{gse_id}_info.json"

    if expr_path.exists() and meta_path.exists():
        if not silent:
            print(f"  {gse_id}: already downloaded, skipping.")
        return

    if not silent:
        print(f"  {gse_id}: downloading from GEO...")

    gse = GEOparse.get_GEO(geo=gse_id, destdir=str(series_dir), silent=silent)

    # --- Info ---
    info = {
        "accession": gse_id,
        "title": gse.metadata.get("title", [""])[0],
        "summary": gse.metadata.get("summary", [""])[0],
        "overall_design": gse.metadata.get("overall_design", [""])[0],
        "type": gse.metadata.get("type", []),
        "platform_id": list(gse.gpls.keys()),
        "n_samples": len(gse.gsms),
        "pubmed_id": gse.metadata.get("pubmed_id", []),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # --- Metadata ---
    meta_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name, "title": gsm.metadata.get("title", [""])[0]}
        # Extract characteristics
        for ch in gsm.metadata.get("characteristics_ch1", []):
            if ":" in ch:
                key, val = ch.split(":", 1)
                row[key.strip()] = val.strip()
            else:
                row.setdefault("characteristic", [])
                if isinstance(row["characteristic"], list):
                    row["characteristic"].append(ch.strip())
        row["platform"] = gsm.metadata.get("platform_id", [""])[0]
        row["source"] = gsm.metadata.get("source_name_ch1", [""])[0]
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(meta_path, index=False)

    # --- Expression matrix ---
    # Try to build a pivoted expression table from sample tables
    expr_frames = []
    for gsm_name, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is not None and not tbl.empty:
            # Typical columns: ID_REF, VALUE
            if "VALUE" in tbl.columns and "ID_REF" in tbl.columns:
                series = tbl.set_index("ID_REF")["VALUE"]
                series.name = gsm_name
                expr_frames.append(series)

    if expr_frames:
        expr_df = pd.concat(expr_frames, axis=1)
        expr_df.index.name = "probe_id"
        expr_df.to_csv(expr_path)
        if not silent:
            print(f"  {gse_id}: saved {expr_df.shape[0]} probes x {expr_df.shape[1]} samples")
    else:
        if not silent:
            print(f"  {gse_id}: no inline expression data found. "
                  f"Check supplementary files in {series_dir}")

    if not silent:
        print(f"  {gse_id}: metadata saved ({len(meta_rows)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Download GEO datasets for scleroderma ML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gse", nargs="+", help="GSE accession(s) to download")
    group.add_argument("--all-curated", action="store_true",
                       help="Download all curated scleroderma datasets")
    parser.add_argument("--output", default="data", help="Output directory (default: data/)")
    parser.add_argument("--silent", action="store_true", help="Suppress GEOparse output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    gse_list = CURATED_GSE if args.all_curated else args.gse

    print(f"Will download {len(gse_list)} dataset(s) to {output_dir}/")
    for gse_id in gse_list:
        gse_id = gse_id.upper()
        if not gse_id.startswith("GSE"):
            print(f"  Skipping invalid accession: {gse_id}")
            continue
        try:
            download_gse(gse_id, output_dir, silent=args.silent)
        except Exception as e:
            print(f"  {gse_id}: ERROR — {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
