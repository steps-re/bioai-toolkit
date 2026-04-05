# Sam's ExoPred Workbench

Last updated: 2026-04-05

This is your master reference for the bioai-toolkit. Everything here reflects actual files on disk, actual model results, and actual limitations. No hype.

---

## What's Here (Data Inventory)

Total: ~160K+ data points across 6 sources, 3 processed pipelines, and 1 trained model family.

---

### Your Data (Rozans Lab)

**`data/rozans-618-enriched.csv`**
- 618 peptides, 60 columns
- Papers 1-3 combined: Paper 1 = 234 (ACS Biomater 2024), Paper 2 = 20 (J Biomed Mater Res A 2025), Paper 3 = 364
- Columns include: sequence, full_notation, n_terminal, c_terminal, variable_residue, scaffold, paper, library, type, clean_sequence, length, mw_da, pI, gravy, instability_index, aromaticity, helix/turn/sheet fractions, net_charge_ph7, terminal AAs, hydrophobicity, aminopeptidase susceptibility, and ~35 more computed properties
- Quality: HIGH. All computed from your published sequences. Terminal modification encoding is clean.
- **Look at first:** The `aminopeptidase_susceptibility` and terminal hydrophobicity columns -- these are your best features for the degradation model.

**`data/rozans-peptide-library.csv`**
- 618 rows, raw sequences with N/C-terminal modification annotations
- Format: `NH2-RGEFV-G(Glycine)-COOH` style notation
- This is the source-of-truth sequence file; the enriched CSV is derived from it.

**`data/rozans_si/paper1_SI_ACSBiomaterSciEng_2024.pdf`**
- 138 pages of supplementary data from your ACS Biomater 2024 paper
- Contains: degradation curves across cell types, donor-to-donor variation, LC-MS spectra
- The degradation curves are the actual ground truth your model trains on (via the calibration model parameters in `train_v2.py`).

**`data/rozans_si/paper2_SI_JBiomedMaterResA_2025.pdf`**
- 23 pages of LC-MS validation spectra from your J Biomed Mater Res A 2025 paper

**`data/rozans_si/PMC11322908/`** (Paper 1 PMC archive)
- Full text XML, PDF, 8 figures (GIF + JPG), plus SI PDF
- 18 files total

**`data/rozans_si/PMC11913071/`** (Paper 2 PMC archive)
- Full text XML, PDF, 8 figures, plus SI PDF
- 19 files total

**`data/processed/rozans_template.csv`**
- 3 example rows showing the exact format for your 80K data export
- Columns: sequence, n_terminal_mod, c_terminal_mod, enzyme_ec, enzyme_name, enzyme_family, measurement_type, value, curve_values, curve_timepoints, conditions_ph, conditions_temp_c, conditions_matrix, source, confidence
- **This is the file you need to match when exporting your 80K LC-MS data points.** The format supports both single half-life values and full kinetic curves (semicolon-delimited).

---

### Public Protease Databases

**`data/merops/`** — 7 files from EBI MEROPS (the canonical protease database)
- `cleavage.txt` — 147,311 lines. Raw cleavage site records with substrate, protease, position, references.
- `substrate.txt` — 75,959 lines. Substrate-protease relationships.
- `Substrate_search.txt` — 108,196 lines. Extended substrate search results.
- `cleavage_refs.txt` — 6,811 lines. Literature references for cleavage data.
- `substrate_2d.txt` — 2,625 lines. 2D substrate representations.
- `combinatorial_substrates.txt` — 186 lines. Combinatorial library data.
- `pathway_cleavage.txt` — 353 lines. Pathway-associated cleavages.

**`data/processed/merops_exopeptidase_cleavages.csv`**
- 15,883 rows. Filtered from the full MEROPS cleavage dataset to include only exopeptidase families (aminopeptidases, carboxypeptidases, dipeptidyl peptidases).
- This is the core feature source for the model's MEROPS-derived features (44 of 72 features in v2).
- Quality: GOOD for cleavage preferences. NOT kinetic data -- these are binary "does it cleave here" records, not rates.

**`data/processed/merops_all_cleavages_summary.csv`**
- 258 rows. One row per protease family, with total cleavage site counts.
- Use this to quickly identify which families have enough data to be useful.

---

### Peptide Half-Life Data

**`data/peplife/`** — PEPlife2 complete download (9 files)
- `peplife2_api_natural.json` + `peplife2_api_modified.json` + `peplife2_api_cyclic.json` + `peplife2_api_linear.json` — raw API dumps
- `peplife2_complete.fasta` + `peplife2_natural.fasta` + `peplife2_modified.fasta` — sequence files
- `peplife2_modified.map` — modification mapping
- `peplife2_all_structures.zip` — PDB structure files

**`data/processed/peplife2_combined.csv`**
- 4,500 rows with columns: seq, half_life, protease, test_sample, assay, conditions, etc.
- Quality: **HIGHLY HETEROGENEOUS.** This is the biggest problem with public peptide stability data. Entries mix:
  - Different proteases (trypsin, chymotrypsin, pepsin, blood serum, liver homogenate, etc.)
  - Different species (human, rat, mouse, pig)
  - Different assay methods (HPLC, LC-MS, fluorescence, radioactive labeling)
  - Different conditions (pH 2-9, temp 25-37C, different matrices)
- The v1 neural net trained on this got R^2 = -0.19. That is not a model problem. That is a data problem.
- **Use for benchmarking and validation, NOT for training.** The Spearman correlation between the v2 model predictions and PEPlife2 half-lives is 0.106 (p=5e-5) -- statistically significant but weak, which is expected because the model was trained on exopeptidase degradation in cell culture, not the grab-bag of conditions in PEPlife2.

---

### DPP-IV (GLP-1 Relevant)

**`data/dppiv/idppiv-benchmark/`**
- 1,330 peptides total: 665 DPP-IV inhibitory + 665 non-inhibitory (train/test split provided)
- Original MATLAB code (libsvm, OMST, DTW). The canonical benchmark dataset for DPP-IV substrate/inhibitor classification.
- Also contains `iDPPIV.zip` and `figshare_download.zip` (original downloads).

**`data/processed/dppiv_benchmark.csv`**
- 1,330 rows. Processed version of the iDPPIV benchmark in CSV format.

**`data/dppiv/chembl/chembl284_dpp4_activities.csv`**
- 8,320 rows. Small-molecule DPP-IV inhibitor bioactivities from ChEMBL target 284.
- These are mostly IC50/Ki values for small molecules, not peptides. Useful for understanding DPP-IV pharmacology but not directly useful for peptide degradation prediction.

**`data/processed/dppiv_chembl_ic50.csv`**
- 4,644 rows. Filtered IC50 values from the ChEMBL dataset.

**`data/dppiv/Structural-DPP-IV/`**
- Full PyTorch model code from the StructuralDPPIV paper (cloned repo)
- Includes: model architecture, training scripts, config YAMLs, data encoding, ablation/CAM/perturbation notebooks
- Train/test data at `data/DPP-IV/{train,test}/{train,test}.tsv`
- Worth looking at for architecture ideas if you go the DPP-IV-specific route.

**`data/dppiv/bert-dppiv/uniprot_protein_data.txt`**
- 1,113,206 lines. UniProt protein sequences used to pretrain BERT-DPPIV.
- This is a massive general protein sequence corpus. Only useful if you want to pretrain your own protein language model (you don't -- use ESM-2 instead).

**`data/dppiv/mendeley/partition_data.rar`**
- Compressed archive from Mendeley. Has not been extracted.

---

### MMP Cleavage Profiling

**`data/turk2015/mmc2-table-S1.xlsx`**
- **18,583 peptides x 20 columns** (peptide sequence + Z-scores for 18 MMPs: MMP1, MMP2, MMP3, MMP7, MMP8, MMP9, MMP10, MMP11, MMP12, MMP13, MMP14, MMP15, MMP16, MMP17, MMP19, MMP20, MMP24, MMP25)
- This is the single largest quantitative protease specificity dataset in this toolkit.
- Z-scores are quantitative cleavage preference measures from combinatorial peptide library profiling.
- The v4 model added 6 Turk-derived MMP features but they did not improve exopeptidase prediction (delta R^2 = -0.004). This makes biological sense -- MMPs are endopeptidases, not exopeptidases.
- **Still valuable** if you pivot to endopeptidase prediction, or as a benchmark for a general protease specificity model.

**`data/turk2015/mmc1-supplemental-figures.pdf`** + **`mmc3-supplemental-methods.pdf`**
- Supporting figures and methods for the Turk 2015 paper.

---

### External Validation

**`data/external_validation/extracted_data.csv`**
- 55 measurements from Bottger et al. 2017
- 8 therapeutic peptides (including octreotide, somatostatin, LHRH, substance P) tested in 6 blood matrices (human/rat/mouse plasma and serum)
- Columns include peptide identity, matrix, time points, and remaining fraction
- Quality: GOOD. Controlled conditions, published, well-characterized peptides.

**`data/external_validation/cleavage_sites.csv`**
- 23 cleavage products identified with exact masses from Bottger 2017
- Useful for validating predicted cleavage sites against experimentally observed ones.

**`data/external_validation/bottger2017_main.pdf`** + **`bottger2017_S1_Text.docx`** + **`S1-S3_Fig.tif`**
- Full paper and supplementary materials.

**`data/external_validation/kohler2024_main.pdf`**
- Protocol comparison paper. Useful for understanding assay variability.

**`data/external_validation/README.md`**
- Notes on Marciano 2023 SI (needs manual download) and other external sources.

**Important: v4 external validation against Bottger was poor** (Spearman r=0.0, Pearson r=0.07). The model predicts a narrow range (0.09-0.32) while actual values span 0-103. This is because the model was trained on cell culture exopeptidase degradation, not blood matrix stability. Your 80K data points under controlled conditions are what's needed to bridge this gap.

---

### ESM-2 Embeddings

**`data/processed/esm2_embeddings.pkl`**
- Precomputed ESM-2 (esm2_t12_35M_UR50D, 480-dim) embeddings for all 618 Rozans peptides
- PCA-reduced to 30 components for model input
- v3 results: adding ESM-2 features HURT performance (R^2 went from 0.948 to 0.918 leave-seq-out). ESM-2 alone was useless (R^2 = -0.01). This is because:
  1. ESM-2 is trained on full proteins, not short peptides (6-12 residues)
  2. Terminal modifications are invisible to ESM-2 (it only sees amino acid sequences)
  3. With only 19 unique sequences, the ESM embeddings just add noise
- **Revisit ESM-2 once you have 80K data.** With hundreds of unique sequences, the embeddings may become useful.

---

### Training Task Files

**`data/training/task_a_binary.csv`** — 32,562 rows. Binary stable/unstable classification labels.
**`data/training/task_b_halflife.csv`** — 1,721 rows. Continuous half-life regression labels.
**`data/training/task_c_kinetic.csv`** — 0 rows (header only). Kinetic curve fitting labels (empty, awaiting your data).

---

### BRENDA Enzyme Database

**`data/brenda/`** — Directory exists but appears empty. BRENDA data was planned but not downloaded. If you want Km/kcat values for exopeptidases, install `brendapy` and pull them.

---

## What's Been Built (Models & Code)

### ExoPred Pipeline (`exopred/`)

**`exopred/data_pipeline.py`**
- Normalizes all datasets (Rozans, MEROPS, PEPlife2, DPP-IV, Turk, Bottger) into a unified format
- Handles three-letter to one-letter amino acid conversion, modification parsing, enzyme family mapping

**`exopred/features.py`**
- 4 feature groups:
  - Physicochemical (18): length, MW, pI, GRAVY, instability index, aromaticity, secondary structure fractions, net charge, terminal AA properties
  - MEROPS-derived (44): cleavage frequency features per exopeptidase family, P1/P1' positional preferences
  - Modification (6): binary encoding of N/C-terminal protection (NH2, Ac, beta-alanine, COOH, amide)
  - Enzyme (4-16): one-hot cell type / enzyme family encoding
- Optional: ESM-2 embeddings (30 PCA components from 480-dim)

**`exopred/model.py`**
- Multi-task PyTorch model with 3 heads: binary classification, half-life regression, kinetic curve prediction
- Used in v1 (Phase 1 neural network). Abandoned in favor of GBR after v1 failed on PEPlife2.

**`exopred/train.py`** (v1)
- Phase 1 neural network on PEPlife2 data
- Result: R^2 = -0.19. The model is worse than predicting the mean. PEPlife2 is too heterogeneous.

**`exopred/train_v2.py`** (v2 -- CURRENT BEST)
- GradientBoostingRegressor on Sam's Paper 1 calibration data (234 peptides x 4 cell types = 912 samples after filtering)
- Uses the calibration model parameters from your ACS Biomater 2024 paper (N_MOD_BASE, C_MOD_BASE dictionaries) to compute fraction_remaining_48h as training labels
- Results:
  - Random 5-fold CV: R^2 = 0.997 (almost perfect, but misleading -- same sequences appear in train and test)
  - **Leave-sequence-out CV (19 folds): R^2 = 0.948** (the real number)
  - Ridge baseline: R^2 = 0.918 (leave-seq-out)
  - PEPlife2 external validation: Spearman r = 0.106 (weak but significant)
- 72 features total

**`exopred/train_v3.py`** (v3)
- Added ESM-2 embeddings (30 PCA dims) to v2 features
- Results:
  - v2 features only: R^2 = 0.948 leave-seq-out (reproduced)
  - v2 + ESM-2: R^2 = 0.918 leave-seq-out (WORSE by 0.03)
  - ESM-2 only: R^2 = -0.01 (garbage)
- Conclusion: ESM-2 adds noise with only 19 unique sequences. Not surprising.

**`exopred/train_v4.py`** (v4)
- Added 6 Turk MMP Z-score features to v2 features
- Results:
  - v2 baseline: R^2 = 0.948 leave-seq-out
  - v4 with Turk: R^2 = 0.944 (trivially worse)
  - Bottger external validation: Spearman r = 0.0, Pearson r = 0.07 (no predictive power on blood stability)
- Conclusion: MMP features don't help exopeptidase prediction. Expected.

**`exopred/predict.py`**
- Inference module with heuristic fallback + trained model blending
- Heuristic: amino acid preference scores for APN, CPA, CPB, LAP, DPP-IV, NEP based on Rozans 2024 + MEROPS literature
- Can predict for individual enzymes or all 6 exopeptidases at once
- Usage: `ExoPredPredictor().predict("RGDSP", enzyme="APN")`

**`exopred/api.py`**
- FastAPI server with 9 routes, API key authentication, rate limiting, CORS for Streamlit
- Endpoints include single prediction, batch prediction, enzyme list, health check
- Run: `uvicorn exopred.api:app --host 0.0.0.0 --port 8000`
- Docs at `/docs` (Swagger) and `/redoc`

**`exopred/config.py`**
- Path constants: CHECKPOINT_DIR, DATA_DIR, PROCESSED_DIR

### Checkpoints (`exopred/checkpoints/`)

| File | Model | Notes |
|------|-------|-------|
| `exopred_phase1.pt` | v1 PyTorch neural net | Trained on PEPlife2. Garbage. Kept for reference. |
| `exopred_v2_gbr.pkl` | v2 GBR (sklearn) | **Current production model.** R^2=0.948 leave-seq-out. |
| `exopred_v4_gbr.pkl` | v4 GBR + Turk features | No improvement over v2. |
| `v2_metrics.json` | v2 evaluation results | Full CV metrics + PEPlife2 validation |
| `v3_metrics.json` | v3 evaluation results | ESM-2 ablation study |
| `v4_metrics.json` | v4 evaluation results | Turk ablation + Bottger external validation |
| `phase1_metrics.json` | v1 evaluation results | |
| `v2_feature_importance.csv` | GBR feature importances | Shows which features actually matter |
| `v3_feature_importance.csv` | v3 feature importances | |
| `v4_feature_importance.csv` | v4 feature importances | |

### Streamlit Pages (17 total)

| # | File | What it does |
|---|------|-------------|
| 01 | `pages/01_Rozans_Analysis.py` | Browse your 618 peptides, property distributions, paper-level breakdowns |
| 02 | `pages/02_Enzyme_Visualization.py` | 3D enzyme structures, active sites, substrate binding |
| 03 | `pages/03_Commercial_Opportunities.py` | Market sizing for peptide degradation prediction |
| 04 | `pages/04_AlphaFold.py` | AlphaFold structure lookup and visualization |
| 05 | `pages/05_UniProt.py` | UniProt protein database queries |
| 06 | `pages/06_PubChem.py` | PubChem compound search |
| 07 | `pages/07_Sequence_Tools.py` | Sequence manipulation, translation, alignment |
| 08 | `pages/08_Peptide_Library.py` | Library design tools (combinatorial, positional scanning) |
| 09 | `pages/09_BLAST.py` | NCBI BLAST sequence search |
| 10 | `pages/10_Degradation_Predictor.py` | Interactive degradation prediction using ExoPred |
| 11 | `pages/11_MMP14_Predictor.py` | MMP-14 specific cleavage prediction |
| 12 | `pages/12_Hydrogel_Designer.py` | Peptide-crosslinked hydrogel design |
| 13 | `pages/13_Polymer_Degradation.py` | Polymer degradation modeling |
| 14 | `pages/14_Self_Assembly.py` | Peptide self-assembly prediction |
| 15 | `pages/15_Protease_Specificity.py` | MEROPS-based protease specificity explorer |
| 16 | `pages/16_ExoPred_Data.py` | ExoPred integrated data explorer -- MEROPS heatmaps, PEPlife2 distributions, dataset comparison |
| 17 | `pages/17_GLP1_Market.py` | GLP-1 market intelligence -- biosimilar landscape, patent cliffs, DPP-IV competitive analysis, revenue model |

### Other Code

**`process_datasets.py`** — Top-level script that runs the full data processing pipeline (MEROPS filtering, PEPlife2 parsing, DPP-IV benchmark extraction, etc.)

**`analysis/full_analysis.py`** — Detailed statistical analysis of the 618 Rozans peptides. Output at `analysis/output/`.

**`analysis/output/rozans-618-analysis.md`** — Written analysis report.
**`analysis/output/rozans-618-enriched.csv`** — Copy of the enriched dataset (same as `data/rozans-618-enriched.csv`).

---

## The Key Finding (and Why Your 80K Data Matters)

The model hits a wall at R^2 = 0.948 for leave-sequence-out prediction. Here is why:

1. **Only 19 unique sequences in training.** Paper 1's library varies only the C-terminal residue of a fixed scaffold (RGEFV-X). The model can learn "how does changing the last residue affect degradation" but cannot learn general sequence-degradation relationships.

2. **Degradation is dominated by terminal modification + cell type, not internal sequence.** The top features by importance are N-terminal mod, C-terminal mod, and cell type encoding. The actual amino acid sequence features contribute relatively little. This is partly real biology (terminal protection matters a lot) and partly an artifact of having only 19 sequences.

3. **External validation fails.** The Bottger therapeutic peptides (somatostatin, octreotide, LHRH, substance P, etc.) are nothing like the RGEFV-X scaffold. The model has never seen anything like them, so predictions are meaningless.

**What 80K data points fix:**
- Hundreds (thousands?) of unique sequences under controlled conditions
- Enough diversity for the model to learn position-specific and motif-level degradation rules
- Enough data for ESM-2 embeddings to potentially become useful (they failed at n=19)
- A publishable dataset that is 70x larger than the current SOTA training set (ENZ-XGBoost, 1,119 peptides)

---

## What Sam Should Do Next (Decision Tree)

### Path A: Academic Paper First
**"ML Prediction of Exopeptidase-Mediated Peptide Degradation"**

1. Export your 80K LC-MS data to match `data/processed/rozans_template.csv` format. The columns you need:
   - `sequence` (one-letter amino acid)
   - `n_terminal_mod`, `c_terminal_mod` (use the same naming: NH2, Ac, N-betaA, COOH, amide, betaA)
   - `enzyme_ec`, `enzyme_name`, `enzyme_family` (if known; use "unknown" and "human_serum_mix" if testing in serum)
   - `measurement_type`: "half_life" (single value) or "kinetic_curve" (time series)
   - `value`: half-life in minutes (for half_life type)
   - `curve_values`, `curve_timepoints`: semicolon-delimited (for kinetic_curve type)
   - `conditions_ph`, `conditions_temp_c`, `conditions_matrix`
   - `source`: "rozans"
   - `confidence`: 1.0 for clean data, lower for questionable measurements

2. Run `python3 -m exopred.train_v2 --rozans /path/to/your/80k_data.csv` (you may need to add the --rozans flag; currently the training script reads from the hardcoded enriched CSV).

3. Benchmark against:
   - ENZ-XGBoost (Qiang et al.): current SOTA for enzyme-specific degradation, R^2 = 0.84, trained on 1,119 peptides. You should beat this.
   - PeptideBERT (Guntuboina et al.): fine-tuned ProtBERT for hemolysis/solubility. Different task but same architecture pattern.
   - ProtTrans (Elnaggar et al.): T5-based, published R^2 = 0.84 on blood half-life prediction.

4. The paper writes itself:
   - 70x more training data than any prior work
   - First exopeptidase-specific prediction model
   - Novel features from MEROPS (44 protease family cleavage frequencies)
   - Leave-sequence-out validation (honest generalization metric)
   - Open-source the model + code, keep the training data proprietary

5. Target journals: *Journal of Chemical Information and Modeling* (ACS, IF ~6), *Bioinformatics* (Oxford, IF ~6.9), *Briefings in Bioinformatics* (IF ~9.5 for the review angle). If the results are strong enough: *Nature Biotechnology* brief communication.

### Path B: ExoPred API Startup

1. Complete Path A steps 1-2 (you need the trained model regardless).
2. The &#36;30K lab work: synthesize 50-100 peptides, run purified DPP-IV / APN / CPA degradation assays. This gives you enzyme-resolved kinetics that the cell culture data can't provide.
3. Deploy: `uvicorn exopred.api:app --host 0.0.0.0 --port 8000` is already built. Add Stripe billing, usage metering, and a dashboard.
4. Pricing tiers:
   - Per-query: &#36;50-200 depending on output detail (binary vs. kinetic curve prediction)
   - Platform license: &#36;25-100K/yr for unlimited API access
   - Enterprise: &#36;100K+ with custom model training on client's proprietary data
5. First customers: the 50+ GLP-1 biosimilar developers racing to market before patent cliffs (semaglutide 2032, tirzepatide 2036). Use Page 17 market intel for targeting.
6. Competitive moat: nobody else has 80K+ exopeptidase degradation measurements. The model improves with every customer engagement (data flywheel).

### Path C: GLP-1 Biosimilar Consulting

1. Synthesize 20-50 GLP-1 position-8 variants (the DPP-IV cleavage site), run LC-MS against purified DPP-IV.
2. Build a DPP-IV-specific predictor using the iDPPIV benchmark (1,330 peptides) + your new kinetic data.
3. Offer to biosimilar companies: "We predict which sequence modifications maintain GLP-1R activity while resisting DPP-IV cleavage. &#36;15-50K per engagement."
4. Each engagement generates data that trains a better model (flywheel).
5. Lower risk than Path B (consulting revenue from day 1) but lower ceiling.

### Path D: License the Dataset

1. Publish the paper first (Path A). This proves the dataset exists, is high-quality, and is scientifically valuable.
2. License targets:
   - **Peptilogics** (&#36;205M raised, AI peptide design platform -- they need degradation prediction)
   - **Pinnacle Peptides** (&#36;134M, custom peptide synthesis -- stability data differentiates their service)
   - **ProteinQure** (&#36;16M, computational peptide design -- their models lack degradation training data)
   - **Novo Nordisk** / **Eli Lilly** (GLP-1 incumbents -- competitive intelligence on degradation)
3. Pricing: &#36;100-500K one-time license, or &#36;50-100K/yr with updates as you generate more data.
4. Highest &#36;/effort ratio but requires the paper for credibility and depends on a small number of buyers.

### My recommendation

Do Path A first regardless. It takes 2-4 weeks of your time (data formatting + retraining + writing), costs nothing, and unlocks all other paths. The paper gives you credibility for B/C/D and the trained model is required for B/C anyway.

If you want revenue fastest: A then C (consulting, 1-2 months to first &#36;).
If you want to build something big: A then B (API startup, 3-6 months to first &#36;).
If you want maximum &#36; for minimum ongoing effort: A then D (licensing, depends on finding the right buyer).

---

## Bioinformatics Tools Sam Should Know

### Already Installed (in requirements.txt or imported by code)

| Package | Version | Used For |
|---------|---------|----------|
| `streamlit` | 1.45.1 | All 17 pages of the toolkit UI |
| `pandas` | (installed) | Data manipulation everywhere |
| `plotly` | (installed) | Interactive charts in Streamlit |
| `biopython` | (installed) | Sequence tools, molecular weight, pI calculation |
| `rdkit` | (installed) | Chemical structure handling, molecular descriptors |
| `pubchempy` | (installed) | PubChem compound lookups |
| `py3Dmol` / `stmol` | (installed) | 3D molecular visualization in Streamlit |
| `Pillow` | (installed) | Image handling |
| `requests` | (installed) | HTTP calls to external APIs |
| `scikit-learn` | (imported in train_v2) | GBR, Ridge, StandardScaler, CV, metrics |
| `scipy` | (imported in train_v2) | Spearman correlation, curve fitting |
| `joblib` | (imported in train_v2) | Model serialization |
| `numpy` | (imported everywhere) | Numerical computation |
| `torch` | (imported in model.py) | Neural network (v1), ESM-2 embeddings |
| `transformers` | (used for ESM-2) | Hugging Face model loading |
| `fastapi` + `uvicorn` + `pydantic` | (imported in api.py) | API server |

### Recommended to Install

**Protein Language Models**

- **`fair-esm`** — Facebook's ESM-2 and ESMFold. Already used for embeddings (t12, 35M params). Try ESMFold for structure prediction on your peptides -- it's fast (no MSA required) and gives pLDDT confidence scores per residue. Terminal pLDDT could be a proxy for flexibility/accessibility to exopeptidases.
  ```bash
  pip install fair-esm
  ```
  ```python
  import esm
  model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # bigger model
  ```

- **ProtTrans** — T5-based protein language model from Rostlab. Published R^2=0.84 on blood half-life prediction with LoRA fine-tuning (0.5% of parameters). Architecture is in the Elnaggar et al. 2022 paper. Worth trying once you have enough data.
  ```bash
  pip install transformers sentencepiece
  ```
  ```python
  from transformers import T5Tokenizer, T5EncoderModel
  tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
  model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
  ```

**Protease Prediction**

- **PROSPERous** — web server at prosper.erc.monash.edu.au. Predicts cleavage sites for 90 proteases. Run your peptides through it and compare predicted cleavage sites against your LC-MS identified fragments. No Python package -- it's web-only, but you can automate with requests.

- **DeepCleave** — deep learning cleavage site predictor. Compare its predictions against your measured cleavage products from Paper 1/Paper 2 SI data.

**Peptide Property Prediction**

- **`peptides`** — calculate 100+ physicochemical descriptors (charge profiles, Boman index, hydrophobic moment, etc.). Cross-reference against your enriched CSV to find features that correlate with degradation.
  ```bash
  pip install peptides
  ```
  ```python
  from peptides import Peptide
  p = Peptide("RGDSP")
  print(p.charge(pH=7.4), p.boman(), p.hydrophobic_moment())
  ```

- **`modlamp`** — antimicrobial peptide descriptors. Some features transfer to degradation (charge distribution, hydrophobicity moments, amphipathic helix propensity).
  ```bash
  pip install modlamp
  ```
  ```python
  from modlamp.descriptors import GlobalDescriptor
  d = GlobalDescriptor(["RGDSP"])
  d.calculate_all()
  ```

- **PeptideBERT** — fine-tuned ProtBERT for hemolysis, solubility, and nonfouling prediction. The architecture (BERT + task-specific head) could be directly adapted for degradation prediction. Paper: Guntuboina et al. 2024.

**Structure & Dynamics**

- **ESMFold** (via `fair-esm`) — fast structure prediction without MSAs. Run on your 618 peptides, extract pLDDT scores at terminal residues as flexibility proxies.
  ```python
  import esm
  model = esm.pretrained.esmfold_v1()
  # Note: may struggle with short peptides (<10 residues)
  ```

- **`biotite`** — modern structural bioinformatics. SASA calculations, secondary structure assignment, dihedral angles from PDB files.
  ```bash
  pip install biotite
  ```
  ```python
  import biotite.structure.io.pdb as pdb
  import biotite.structure as struc
  structure = pdb.PDBFile.read("structure.pdb").get_structure()
  sasa = struc.sasa(structure[0])
  ```

- **`freesasa`** — solvent-accessible surface area. Calculate terminal residue accessibility -- buried termini are less susceptible to exopeptidases.
  ```bash
  pip install freesasa
  ```
  ```python
  import freesasa
  result = freesasa.calc(freesasa.structureFromPDB("structure.pdb"))
  ```

- **`MDAnalysis`** — if you ever run MD simulations on your peptides (to study terminal flexibility, solvent exposure dynamics).
  ```bash
  pip install MDAnalysis
  ```

**Data Science / ML**

- **`xgboost`** — usually beats sklearn GBR. Try it on the v2 training set as a drop-in replacement.
  ```bash
  pip install xgboost
  ```
  ```python
  from xgboost import XGBRegressor
  model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
  ```

- **`shap`** — SHAP values for feature importance. Better than GBR's built-in `feature_importances_` because SHAP shows directionality (does higher hydrophobicity increase or decrease degradation?) and interaction effects.
  ```bash
  pip install shap
  ```
  ```python
  import shap
  explainer = shap.TreeExplainer(gbr_model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```

- **`optuna`** — hyperparameter optimization. Auto-tune the GBR (or XGBoost) for best leave-sequence-out R^2.
  ```bash
  pip install optuna
  ```
  ```python
  import optuna
  def objective(trial):
      n_est = trial.suggest_int("n_estimators", 100, 1000)
      lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
      # ... train, return leave-seq-out R^2
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=100)
  ```

**Kinetics & Curve Fitting**

- **`lmfit`** — non-linear curve fitting with parameter constraints, confidence intervals, and model comparison. Better than raw scipy for fitting degradation time courses.
  ```bash
  pip install lmfit
  ```
  ```python
  from lmfit import Model
  def decay(t, a, k): return a * np.exp(-k * t)
  model = Model(decay)
  result = model.fit(data, t=timepoints, a=100, k=0.01)
  print(result.params["k"].value, result.params["k"].stderr)
  ```

- **`scipy.optimize.curve_fit`** — already available via scipy. Simpler than lmfit but no parameter uncertainty propagation.

**Database Access**

- **`brendapy`** — parse the full BRENDA enzyme database locally. Query Km, kcat, and specific activity for aminopeptidases, carboxypeptidases, DPP-IV.
  ```bash
  pip install brendapy
  ```
  ```python
  from brendapy import BrendaParser
  bp = BrendaParser()
  results = bp.get("3.4.14.5")  # DPP-IV
  ```

- **`chembl_webresource_client`** — programmatic ChEMBL access. Already have the DPP-IV IC50 data downloaded, but useful for pulling other protease inhibitor data.
  ```bash
  pip install chembl-webresource-client
  ```
  ```python
  from chembl_webresource_client.new_client import new_client
  activity = new_client.activity
  results = activity.filter(target_chembl_id="CHEMBL284", pchembl_value__gte=5)
  ```

- **`bioservices`** — unified API for UniProt, KEGG, ChEMBL, PDB, etc. Useful for cross-referencing enzyme metadata.
  ```bash
  pip install bioservices
  ```

**Visualization**

- **`nglview`** — interactive 3D molecular visualization in Jupyter. Better than py3Dmol for exploring structures.

- **`logomaker`** — sequence logos from cleavage site data. Make publication-quality logos from your MEROPS exopeptidase P1/P1' preferences. These would be great figures for the paper.
  ```bash
  pip install logomaker
  ```
  ```python
  import logomaker
  df = pd.DataFrame(...)  # position-probability matrix
  logo = logomaker.Logo(df, shade_below=0.5, fade_below=0.5)
  ```

- **`seaborn`** — already available (dependency of other packages). Use for cleavage heatmaps, correlation matrices, paired violin plots comparing cell types.

---

## Quick Start Commands

```bash
# Navigate to the toolkit
cd /mnt/c/Users/miger/Documents/steps-bd/bioai-toolkit

# Launch the Streamlit UI (all 17 pages)
streamlit run app.py

# Run the trained model predictor on a single peptide
python3 -c "from exopred.predict import ExoPredPredictor; p=ExoPredPredictor(); print(p.predict('RGDSP'))"

# Predict for a specific enzyme
python3 -c "from exopred.predict import ExoPredPredictor; p=ExoPredPredictor(); print(p.predict('RGDSP', enzyme='APN'))"

# Batch prediction
python3 -c "from exopred.predict import ExoPredPredictor; p=ExoPredPredictor(); print(p.predict_batch(['RGDSP','GRGDS','YIGSR']))"

# Explore MEROPS exopeptidase cleavage frequencies by family
python3 -c "import pandas as pd; df=pd.read_csv('data/processed/merops_exopeptidase_cleavages.csv'); print(df.groupby('protease_family').size().sort_values(ascending=False))"

# Browse the Turk 18K MMP peptide dataset
python3 -c "import pandas as pd; df=pd.read_excel('data/turk2015/mmc2-table-S1.xlsx'); print(f'{len(df)} peptides x {len(df.columns)} columns'); print(df.iloc[:5, :6])"

# Check PEPlife2 half-life distribution
python3 -c "import pandas as pd; df=pd.read_csv('data/processed/peplife2_combined.csv'); print(f'Total: {len(df)} entries'); print(df['half_life'].describe()); print('\nTop proteases:'); print(df['protease'].value_counts().head(10))"

# Look at your enriched peptide properties
python3 -c "import pandas as pd; df=pd.read_csv('data/rozans-618-enriched.csv'); print(f'{len(df)} peptides, {len(df.columns)} features'); print(df[['clean_sequence','n_terminal','c_terminal','paper','mw_da','pI','gravy']].head(10))"

# View the data export template (what your 80K data needs to look like)
python3 -c "import pandas as pd; df=pd.read_csv('data/processed/rozans_template.csv'); print(df.to_string())"

# Check model performance metrics
python3 -c "import json; [print(f'--- {f} ---') or print(json.dumps(json.load(open(f'exopred/checkpoints/{f}')), indent=2)) for f in ['v2_metrics.json','v3_metrics.json','v4_metrics.json']]"

# View top features by importance (v2 model)
python3 -c "import pandas as pd; df=pd.read_csv('exopred/checkpoints/v2_feature_importance.csv'); print(df.sort_values('importance', ascending=False).head(20).to_string())"

# Start the API server
uvicorn exopred.api:app --host 0.0.0.0 --port 8000
# Then visit http://localhost:8000/docs for interactive API docs

# Retrain with your 80K data (when ready)
python3 -m exopred.train_v2 --rozans /path/to/your/80k_export.csv

# Run the full data processing pipeline (regenerate all processed/ files)
python3 process_datasets.py
```

---

## File Tree Reference

```
bioai-toolkit/
  app.py                              # Streamlit entry point
  requirements.txt                    # Python dependencies
  Dockerfile                          # Container build
  deploy.sh                           # Deployment script
  process_datasets.py                 # Data processing pipeline
  SAM_WORKBENCH.md                    # This file
  analysis/
    full_analysis.py                  # Statistical analysis of 618 peptides
    output/
      rozans-618-analysis.md          # Analysis report
      rozans-618-enriched.csv         # Copy of enriched data
  exopred/
    __init__.py
    config.py                         # Path constants
    data_pipeline.py                  # Dataset normalization
    features.py                       # Feature engineering (18+44+6+4 features)
    model.py                          # PyTorch multi-task model (v1)
    predict.py                        # Inference + heuristic fallback
    train.py                          # v1: neural net on PEPlife2 (failed)
    train_v2.py                       # v2: GBR on Rozans data (R2=0.948)
    train_v3.py                       # v3: + ESM-2 (R2=0.918, worse)
    train_v4.py                       # v4: + Turk MMP (R2=0.944, no help)
    api.py                            # FastAPI server (9 routes)
    checkpoints/
      exopred_phase1.pt               # v1 model weights
      exopred_v2_gbr.pkl              # v2 model (CURRENT BEST)
      exopred_v4_gbr.pkl              # v4 model
      v2_metrics.json                 # v2 results
      v3_metrics.json                 # v3 results (ESM-2 ablation)
      v4_metrics.json                 # v4 results (Turk ablation + Bottger)
      phase1_metrics.json             # v1 results
      v{2,3,4}_feature_importance.csv # Feature rankings
  pages/
    01_Rozans_Analysis.py             # Your peptide data explorer
    02_Enzyme_Visualization.py        # 3D enzyme structures
    03_Commercial_Opportunities.py    # Market analysis
    04_AlphaFold.py                   # Structure lookup
    05_UniProt.py                     # Protein database
    06_PubChem.py                     # Compound search
    07_Sequence_Tools.py              # Sequence manipulation
    08_Peptide_Library.py             # Library design
    09_BLAST.py                       # Sequence search
    10_Degradation_Predictor.py       # ExoPred interactive
    11_MMP14_Predictor.py             # MMP-14 specific
    12_Hydrogel_Designer.py           # Hydrogel design
    13_Polymer_Degradation.py         # Polymer modeling
    14_Self_Assembly.py               # Self-assembly prediction
    15_Protease_Specificity.py        # MEROPS explorer
    16_ExoPred_Data.py                # Integrated data browser
    17_GLP1_Market.py                 # GLP-1 market intel
  data/
    rozans-618-enriched.csv           # 618 peptides, 60 properties
    rozans-peptide-library.csv        # 618 raw sequences + mods
    rozans_si/
      paper1_SI_*.pdf                 # 138pp degradation curves
      paper2_SI_*.pdf                 # 23pp LC-MS spectra
      PMC11322908/                    # Paper 1 full archive (18 files)
      PMC11913071/                    # Paper 2 full archive (19 files)
    merops/                           # 7 MEROPS files (341K total lines)
    peplife/                          # PEPlife2 complete (9 files)
    dppiv/
      idppiv-benchmark/               # 1,330 peptides (665+/665-)
      chembl/                         # 8,320 DPP-IV activities
      Structural-DPP-IV/              # Full PyTorch model repo
      bert-dppiv/                     # 1.1M UniProt sequences
      mendeley/                       # Compressed partition data
    turk2015/
      mmc2-table-S1.xlsx              # 18,583 peptides x 18 MMPs
      mmc1-supplemental-figures.pdf
      mmc3-supplemental-methods.pdf
    external_validation/
      extracted_data.csv              # 55 Bottger measurements
      cleavage_sites.csv              # 23 cleavage products
      bottger2017_main.pdf + SI
      kohler2024_main.pdf
    brenda/                           # Empty (planned, not downloaded)
    processed/
      merops_exopeptidase_cleavages.csv  # 15,883 exopeptidase records
      merops_all_cleavages_summary.csv   # 258 protease families
      peplife2_combined.csv              # 4,500 half-life entries
      dppiv_benchmark.csv               # 1,330 DPP-IV peptides
      dppiv_chembl_ic50.csv              # 4,644 IC50 values
      esm2_embeddings.pkl               # 618 x 480-dim embeddings
      rozans_template.csv               # YOUR DATA EXPORT FORMAT
      dataset_summary.csv               # Overview of all datasets
    training/
      task_a_binary.csv                  # 32,562 binary labels
      task_b_halflife.csv                # 1,721 half-life labels
      task_c_kinetic.csv                 # Empty (awaiting your data)
```

---

## Known Limitations and Honest Assessment

1. **The model has seen 19 unique sequences.** Everything else is variation in modifications and cell types. R^2=0.948 leave-seq-out sounds great, but 19 folds is not enough to claim robust generalization.

2. **External validation is poor.** Bottger 2017 (blood stability of therapeutic peptides): effectively zero correlation. PEPlife2 (heterogeneous half-lives): Spearman r=0.106. The model works within its training domain (short RGD-analog peptides in cell culture) and nowhere else yet.

3. **ESM-2 doesn't help.** With 6-12 residue peptides and 19 unique sequences, protein language model embeddings are noise. This may change with more data.

4. **MMP features don't help.** Biologically expected (MMPs are endopeptidases). The Turk dataset is still valuable for a future endopeptidase model.

5. **The calibration model labels are derived, not measured.** The v2 model trains on `fraction_remaining_48h` computed from your published calibration model parameters, not raw LC-MS measurements. This adds a layer of abstraction. Your 80K raw measurements would be direct training signal.

6. **PEPlife2 is a dead end for training.** The data is too heterogeneous (mixed proteases, species, assays, conditions). It is useful only as an external benchmark.

7. **The API has no paying users.** It works technically but has not been validated commercially. The pricing in Path B is speculative.

8. **task_c_kinetic.csv is empty.** Full kinetic curve prediction (not just binary or half-life) requires your time-course data.
