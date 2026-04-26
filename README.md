# Systemic Scleroderma GEO Dataset Finder & Downloader

Tools for discovering and downloading gene expression datasets from GEO
(Gene Expression Omnibus) related to **systemic scleroderma / systemic sclerosis (SSc)**.

## Scripts

| Script | Purpose |
|---|---|
| `search_geo.py` | Search GEO for scleroderma datasets via NCBI E-utilities |
| `download_geo.py` | Download and parse selected GEO series into expression matrices |

## Prerequisites

- **Python 3.8+** (3.10 or later recommended)

## Quick Start

```bash
pip install -r requirements.txt

# 1. Search for more datasets
python search_geo.py --term "systemic sclerosis" --organism "Homo sapiens"

# 2. Download specific datasets
python download_geo.py --gse GSE9285 GSE76809 GSE33463

# 3. Download all curated datasets
python download_geo.py --all-curated
```

## Output Structure

```
data/
  GSE9285/
    GSE9285_expression_matrix.csv
    GSE9285_metadata.csv
    GSE9285_info.json
  GSE76809/
    ...
```

---

## Curated GEO Datasets for ML

Datasets ranked by ML suitability: sample size, clear labels, human tissue,
and availability of processed expression matrices.

---

### Tier 1 — Large, Well-Annotated (recommended first)

#### [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809) — Multi-tissue SSc functional genomics
- **Samples**: 577 | **Platform**: Multiple (Affymetrix, Agilent, Illumina)
- **Tissue**: Skin, lung, blood (PBMCs), esophagus
- **Design**: SSc patients (diffuse/limited) vs healthy controls, multi-tissue
- **ML use**: Multi-tissue classification, subtype discovery, cross-tissue biomarkers
- **PMIDs**: 28330499, 21360508, 32210069
- **Notes**: SuperSeries spanning multiple sub-studies. Very rich for multi-task learning.

#### [GSE134310](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134310) — SCOT Trial PBMCs (longitudinal)
- **Samples**: 229 | **Platform**: Affymetrix HumanGene 1.0 ST
- **Tissue**: PBMCs (peripheral blood)
- **Design**: Diffuse SSc patients, myeloablative HSCT vs cyclophosphamide, multiple timepoints
- **ML use**: Treatment response prediction, longitudinal modeling, patient stratification
- **PMIDs**: 30920766, 32933919
- **Notes**: Longitudinal samples at baseline, 8, 14, 20, 26, 38, 44, 48, 54 months.

#### [GSE130953](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130953) — SCOT Trial whole blood
- **Samples**: 205 | **Platform**: Illumina HumanHT-12 V4
- **Tissue**: Whole blood
- **Design**: Diffuse SSc receiving HSCT or CYC vs unaffected controls
- **ML use**: Treatment response classification, disease signature normalization
- **PMID**: 31391177

#### [GSE33463](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33463) — PBMCs from pulmonary hypertension subtypes
- **Samples**: 140 | **Platform**: Illumina HumanHT-12 V3
- **Tissue**: PBMCs
- **Design**: IPAH, SSc (no PAH), SSc-PAH, SSc-PH-ILD, healthy controls
- **ML use**: Multi-class classification of PAH subtypes, SSc complication prediction
- **PMID**: 22545094

#### [GSE45536](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45536) — Plasma cell signature in scleroderma (anti-CD19 therapy)
- **Samples**: 123 | **Platform**: Affymetrix HG-U133 Plus 2
- **Tissue**: Whole blood (PBMCs)
- **Design**: SSc patients receiving anti-CD19 at various doses/timepoints + healthy donors
- **ML use**: Drug response prediction, plasma cell deconvolution
- **PMID**: 24431284

#### [GSE76885](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76885) — SSc skin disease trajectory during MMF treatment
- **Samples**: 194 | **Platform**: Agilent 44k
- **Tissue**: Skin biopsies (forearm + back, longitudinal)
- **Design**: 33 MMF-treated SSc subjects, baseline + 6/12/24/36-month biopsies, clinical responders vs non-responders
- **ML use**: Treatment response prediction, longitudinal trajectory modeling, inflammatory signature tracking
- **Notes**: Elevated inflammatory skin signature predicts MMF response. Rebound on discontinuation.

#### [GSE58095](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58095) — Heterogeneity of SSc skin gene expression
- **Samples**: 102 | **Platform**: Illumina HumanHT-12 V4
- **Tissue**: Skin biopsies
- **Design**: SSc patients with fibro-inflammatory vs keratin-dominant signatures
- **ML use**: Unsupervised clustering, SSc molecular subtype discovery
- **PMIDs**: 26238292, 34654463
- **Notes**: Identified distinct fibro-inflammatory and keratin gene expression signatures.

#### [GSE201405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201405) — Morphea vs SSc skin gene expression
- **Samples**: 36 | **Platform**: Illumina HumanHT-12 V4 (microarray)
- **Tissue**: Skin biopsies (lesional + unaffected)
- **Design**: Inflammatory morphea, sclerotic morphea, unaffected skin; comparison with SSc subsets
- **ML use**: Disease classification (morphea subtypes vs SSc subsets), Th1/fibrotic pathway scoring, biomarker discovery
- **PMID**: 37028702
- **Notes**: Morphea clusters with SSc inflammatory subset. CXCL9 as circulating biomarker. GEO2R compatible.

---

### Tier 2 — Medium-Sized, Disease-Focused

#### [GSE9285](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9285) — Landmark SSc skin gene expression profiling
- **Samples**: 75 arrays (34 subjects) | **Platform**: Agilent Whole Human Genome
- **Tissue**: Skin biopsies (forearm + back)
- **Design**: 17 dSSc, 7 lSSc, 3 morphea, 6 healthy controls
- **ML use**: SSc subtype classification (intrinsic subsets), biomarker discovery
- **PMIDs**: 18648520, 25569146, 22245215
- **Notes**: Foundational paper defining "intrinsic subsets" of SSc skin. Widely cited.

#### [GSE22356](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22356) — SSc-associated pulmonary hypertension PBMCs
- **Samples**: 38 | **Platform**: Affymetrix HG-U133 Plus 2
- **Tissue**: PBMCs
- **Design**: SSc with PAH, SSc without PAH, IPAH, healthy controls
- **ML use**: PAH risk prediction from blood, biomarker discovery
- **PMID**: 20973920

#### [GSE12493](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12493) — TGFβ-responsive signature in SSc fibroblasts
- **Samples**: 40 | **Platform**: Agilent 44k
- **Tissue**: Primary dermal fibroblasts (SSc + healthy)
- **Design**: TGFβ time-course (0, 2, 4, 8, 12, 24h) in dSSc vs control fibroblasts
- **ML use**: TGFβ pathway activation scoring, fibrosis prediction
- **PMID**: 19812599

#### [GSE32413](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32413) — Intrinsic subset validation + rituximab (skin)
- **Samples**: 89 | **Platform**: Agilent 44k
- **Tissue**: Skin biopsies (serial)
- **Design**: 13 dSSc + rituximab, 9 dSSc untreated, 9 healthy controls; serial biopsies
- **ML use**: Intrinsic subset classification, pathway analysis, treatment effect detection
- **PMIDs**: 22318389, 25569146
- **Notes**: Validates intrinsic subsets as stable, reproducible features independent of disease duration.

#### [GSE59785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59785) — Mycophenolate treatment in SSc (skin)
- **Samples**: 82 | **Platform**: Agilent 44k
- **Tissue**: Skin biopsies
- **Design**: SSc patients on MMF, longitudinal skin biopsies with intrinsic subset assignment
- **ML use**: Immune inference, treatment response, intrinsic subset tracking
- **PMIDs**: 23677167, 25569146

#### [GSE45485](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45485) — Molecular signatures during MMF treatment
- **Samples**: ~82 | **Platform**: Agilent 44k
- **Tissue**: Skin biopsies
- **Design**: SSc patients on MMF with longitudinal biopsies, mRSS improvement tracking
- **ML use**: Biomarker discovery, skin score prediction, treatment response
- **PMIDs**: 23677167, 25569146

#### [GSE40839](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40839) — SSc-ILD pulmonary fibroblasts
- **Samples**: 21 | **Platform**: Affymetrix HG-U133A
- **Tissue**: Lung fibroblasts (from biopsies)
- **Design**: SSc-ILD (8), UIP/IPF (3), healthy controls (10)
- **ML use**: ILD classification, fibrotic fibroblast signature
- **PMID**: 23915349

#### [GSE138669](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138669) — Myofibroblast transcriptome / SFRP2+ fibroblast progenitors (scRNA-seq)
- **Samples**: 22 | **Platform**: Illumina NextSeq 500 (scRNA-seq)
- **Tissue**: Skin biopsies
- **Design**: SSc vs control skin; single-cell RNA-seq of dermal fibroblasts
- **ML use**: Cell-type deconvolution, myofibroblast differentiation trajectory, fibroblast subpopulation classification
- **PMID**: 34042322
- **Notes**: Shows myofibroblasts arise from SFRP2/DPP4+ progenitors via a two-stage process. Useful for scRNA-seq-based ML pipelines.

#### [GSE95065](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95065) — SSc skin biopsies (microarray)
- **Samples**: 33 | **Platform**: Affymetrix Clariom D (GPL23080)
- **Tissue**: Skin biopsies
- **Design**: SSc patients vs normal controls
- **ML use**: Classification (SSc vs healthy), gene signature extraction
- **Notes**: Newer Affymetrix platform with broader probe coverage.

---

### Tier 3 — Smaller or Specialized

#### [GSE308096](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308096) — SSc pathogenesis (RNA-seq + miRNA, 2026)
- **Samples**: 20 | **Platform**: Illumina NovaSeq (RNA-seq + small RNA-seq)
- **Tissue**: Not specified (likely skin or blood)
- **Design**: SSc patients vs controls, ceRNA network analysis
- **ML use**: Multi-omics (mRNA + miRNA), network-based features
- **Notes**: New dataset (2026), RNA-seq based.

#### [GSE320020](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE320020) — Juvenile SSc skin scRNA-seq
- **Samples**: 11 (9 SSc + 9 healthy, single-cell) | **Platform**: 10x Chromium
- **Tissue**: Skin
- **Design**: Juvenile SSc vs age-matched controls
- **ML use**: Cell-type deconvolution training, macrophage-fibroblast interaction
- **Notes**: Single-cell RNA-seq. Requires specialized processing.

#### [GSE86984](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86984) — SSc monocyte expression profiling
- **Samples**: 8 | **Platform**: Affymetrix HuGene 1.0 ST
- **Tissue**: Freshly isolated monocytes
- **Design**: 4 dcSSc patients vs 4 healthy donors
- **ML use**: Monocyte-specific signatures, innate immunity features
- **PMID**: 28248863

#### [GSE144625](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144625) — Fli1 in scleroderma myeloid cells
- **Samples**: 6 | **Platform**: Affymetrix Clariom S
- **Tissue**: THP-1 myeloid cell line
- **Design**: Fli1 knockdown vs scrambled control
- **ML use**: Target gene identification, mechanistic features
- **PMID**: 32508810

#### [GSE27165](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE27165) — Egr-1 profibrotic program in fibroblasts
- **Samples**: 12 | **Platform**: Illumina HumanRef-8 v2
- **Tissue**: Primary human skin fibroblasts
- **Design**: Egr-1 vs TGFβ overexpression at 24/48h
- **ML use**: Profibrotic pathway signature extraction
- **PMIDs**: 21931594, 23132749

#### [GSE1724](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1724) — TGFβ response in SSc/IPF lung fibroblasts
- **Samples**: 18 | **Platform**: Affymetrix HG-U95Av2
- **Tissue**: Lung fibroblasts
- **Design**: Control, SSc, IPF fibroblasts ± TGFβ
- **ML use**: SSc vs IPF fibroblast classification
- **PMID**: 15571627

#### [GSE125362](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125362) — ML classifier for SSc intrinsic subsets
- **Samples**: 12 | **Platform**: Agilent 44k + RNA-seq
- **Tissue**: Skin biopsies
- **Design**: 4 controls + 8 SSc patients, microarray and RNA-seq on same samples
- **ML use**: Cross-platform integrative models, intrinsic subset classifier training/testing
- **PMID**: 30920766
- **Notes**: Paired microarray + RNA-seq for the same subjects. Ideal for platform-bridging ML.

#### [GSE4385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4385) — Classic SSc skin microarray (cell-type signatures)
- **Samples**: 33 | **Platform**: Multiple cDNA arrays
- **Tissue**: Skin biopsies + fibroblast cell lines
- **Design**: dSSc, morphea, normal controls; skin + isolated fibroblasts/endothelial/B cells
- **ML use**: Baseline models, cell-type deconvolution, historical benchmarking
- **PMID**: 14530402
- **Notes**: One of the earliest SSc transcriptomic studies. Foundational for cell-type signature work.

---

## ML Project Types by Dataset

| Dataset | Samples | Classification | Regression | Clustering | Time-Series | Feature Selection | Deconvolution |
|---|---|---|---|---|---|---|---|
| [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809) | 577 | SSc vs healthy, subtype | | Multi-tissue clustering | | Cross-tissue biomarkers | |
| [GSE134310](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134310) | 229 | HSCT vs CYC response | | Patient stratification | Longitudinal trajectory | Treatment biomarkers | |
| [GSE130953](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130953) | 205 | Treatment response | | | | Signature normalization | |
| [GSE76885](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76885) | 194 | Responder vs non-responder | mRSS prediction | Inflammatory subgroups | Longitudinal (6–36 mo) | MMF response markers | |
| [GSE33463](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33463) | 140 | Multi-class PAH subtypes | | | | PAH risk biomarkers | |
| [GSE45536](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45536) | 123 | Drug response | | | | | Plasma cell deconvolution |
| [GSE58095](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58095) | 102 | Fibro-inflammatory vs keratin | | Molecular subtype discovery | | Subtype gene signatures | |
| [GSE32413](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32413) | 89 | Intrinsic subset | | Subset validation | Serial biopsies | Rituximab effect markers | |
| [GSE59785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59785) | 82 | Treatment response | | Intrinsic subset tracking | Longitudinal | Immune signature | |
| [GSE45485](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45485) | ~82 | Responder vs non-responder | mRSS improvement | | Longitudinal | Skin score biomarkers | |
| [GSE9285](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9285) | 75 | dSSc / lSSc / morphea / healthy | | Intrinsic subsets | | Subtype biomarkers | |
| [GSE12493](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12493) | 40 | SSc vs control fibroblasts | TGFβ dose-response | | Time-course (0–24h) | TGFβ pathway genes | |
| [GSE22356](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22356) | 38 | PAH risk (SSc±PAH vs IPAH) | | | | Blood PAH biomarkers | |
| [GSE201405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201405) | 36 | Morphea subtype vs SSc subset | | Disease subtype clustering | | CXCL9/CXCL10 biomarkers | |
| [GSE95065](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95065) | 33 | SSc vs healthy | | | | Gene signature extraction | |
| [GSE4385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4385) | 33 | dSSc / morphea / healthy | | | | | Cell-type deconvolution |
| [GSE138669](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138669) | 22 | Fibroblast subpopulation | | Trajectory inference | | Myofibroblast markers | Cell-type deconvolution |
| [GSE40839](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40839) | 21 | SSc-ILD vs IPF vs healthy | | | | ILD fibroblast signature | |
| [GSE308096](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308096) | 20 | SSc vs control | | | | Multi-omics (mRNA+miRNA) | |
| [GSE1724](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1724) | 18 | SSc vs IPF fibroblasts | TGFβ response | | | Lung fibrosis genes | |
| [GSE27165](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE27165) | 12 | | | | Time-course (24/48h) | Egr-1 profibrotic targets | |
| [GSE125362](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125362) | 12 | Intrinsic subset (cross-platform) | | | | Platform-bridging features | |
| [GSE320020](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE320020) | 11 | Juvenile SSc vs healthy | | | | | Cell-type deconvolution |
| [GSE86984](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86984) | 8 | dcSSc vs healthy monocytes | | | | Innate immunity genes | |
| [GSE144625](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144625) | 6 | Fli1 KD vs control | | | | Myeloid target genes | |

---

### Recommended Starting Set for ML

For a first ML project, start with these 3 datasets covering different data types:

1. **[GSE9285](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9285)** — Skin biopsies, subtype classification (microarray)
2. **[GSE130953](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130953)** — Whole blood, treatment response (microarray)
3. **[GSE33463](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33463)** — PBMCs, PAH complication prediction (microarray)

All three have clear disease/control labels, reasonable sample sizes,
and processed expression data available via GEO2R.
