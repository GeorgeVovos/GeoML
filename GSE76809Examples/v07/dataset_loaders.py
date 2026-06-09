"""Per-dataset metadata labellers for v07.

Each GEO series has its own metadata schema (column names, label phrasing,
control terminology). This module provides:

  REGISTRY : {gse_id -> labeler_function}
  PLATFORM_HINT : {gse_id -> preferred GPL accession or None}

Each labeler takes a pandas DataFrame of metadata and returns a Series of
int labels (1 = SSc, 0 = Healthy, -1 = exclude).

The labellers below are HEURISTIC. Verify per-dataset using
``python preprocess.py --inspect <GSE_ID>`` after downloading, and refine
the regex / column rules if the label distribution looks wrong.
"""

from __future__ import annotations

import re
import pandas as pd


# --------------------------------------------------------------------------- #
# Per-dataset labellers
# --------------------------------------------------------------------------- #

def _norm(s) -> str:
	return str(s).lower() if pd.notna(s) else ""


def label_gse76809(meta: pd.DataFrame) -> pd.Series:
	"""Original v06 labeller. Kept here so v07 can use it via the registry."""
	def classify(row):
		title = _norm(row.get("title"))
		disease = _norm(row.get("disease state"))
		sample_type = _norm(row.get("sample type"))
		case_ctrl = _norm(row.get("case/control"))

		if "normal" in disease or "normal" in sample_type or case_ctrl == "control":
			return 0
		if re.match(r"(nor|nl)\d", title):
			return 0
		if any(k in title for k in ["ssc", "dssc", "lssc", "rit", "nonrit"]):
			return 1
		if "sclerosis" in disease or "scleroderma" in disease or "ssc" in disease:
			return 1
		if case_ctrl == "case" or "ssc" in sample_type:
			return 1
		if "morph" in title:
			return 1
		return -1

	return meta.apply(classify, axis=1)


def _generic_ssc_vs_healthy(meta: pd.DataFrame, healthy_keywords, ssc_keywords) -> pd.Series:
	"""Generic two-bucket labeller scanning a few likely columns.

	healthy_keywords / ssc_keywords are substrings (case-insensitive).
	Searches across: title, source, disease state, characteristic, sample type.
	"""
	cols = [c for c in ["title", "source", "disease state", "disease_state",
						"sample type", "sample_type", "characteristic",
						"subject status", "diagnosis", "condition", "group"]
			if c in meta.columns]

	def classify(row):
		blob = " | ".join(_norm(row.get(c)) for c in cols)
		if any(k in blob for k in healthy_keywords):
			return 0
		if any(k in blob for k in ssc_keywords):
			return 1
		return -1

	return meta.apply(classify, axis=1)


def label_gse9285(meta: pd.DataFrame) -> pd.Series:
	"""GSE9285 — landmark SSc skin (Milano 2008). 17 dSSc, 7 lSSc, 3 morphea, 6 control."""
	return _generic_ssc_vs_healthy(
		meta,
		healthy_keywords=["healthy", "normal", "control"],
		ssc_keywords=["dssc", "lssc", "ssc", "scleroderma", "sclerosis", "morphea"],
	)


def label_gse58095(meta: pd.DataFrame) -> pd.Series:
	"""GSE58095 — heterogeneity of SSc skin (Assassi 2015). 61 SSc vs 36 control + extras."""
	return _generic_ssc_vs_healthy(
		meta,
		healthy_keywords=["healthy", "normal", "control"],
		ssc_keywords=["ssc", "scleroderma", "sclerosis"],
	)


def label_gse45536(meta: pd.DataFrame) -> pd.Series:
	"""GSE45536 — anti-CD19 trial PBMCs. SSc patients + healthy donors at multiple timepoints."""
	return _generic_ssc_vs_healthy(
		meta,
		healthy_keywords=["healthy", "normal", "donor", "control"],
		ssc_keywords=["ssc", "scleroderma", "sclerosis", "patient"],
	)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY = {
	"GSE76809": label_gse76809,
	"GSE9285": label_gse9285,
	"GSE58095": label_gse58095,
	"GSE45536": label_gse45536,
}

# Preferred platform per dataset (None = use all platforms / single-platform GSE)
PLATFORM_HINT = {
	"GSE76809": "GPL6480",
	"GSE9285": None,
	"GSE58095": None,
	"GSE45536": None,
}


def get_labeler(gse_id: str):
	if gse_id not in REGISTRY:
		raise KeyError(
			f"No labeller registered for {gse_id}. Add one in dataset_loaders.py "
			f"(REGISTRY) after inspecting data/{gse_id}/{gse_id}_metadata.csv."
		)
	return REGISTRY[gse_id]


def get_platform(gse_id: str):
	return PLATFORM_HINT.get(gse_id)
