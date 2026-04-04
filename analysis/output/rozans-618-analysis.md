# Rozans Peptide Library — Full Analysis

**618 peptides** from 3 publications (Pashuck Lab, Lehigh University)
Analysis date: 2026-04-03

## 1. Overview

| Metric | Value |
|--------|-------|
| Total peptides | 618 |
| Unique sequences | 408 |
| Papers | 3 |
| Libraries | 17 |
| Length range | 5-13 aa |
| MW range | 444-1499 Da |
| pI range | 4.07-11.33 |

### Peptides by Paper

- **Paper 1 (ACS Biomater 2024)**: 234 peptides, 13 libraries
- **Paper 2 (JBMR-A 2025)**: 20 peptides, 2 libraries
- **Paper 3 (Adv Healthcare Mater 2025)**: 364 peptides, 2 libraries

## 2. Property Distributions

### Molecular Weight

| Stat | Value |
|------|-------|
| Mean | 970.434 Da |
| Median | 1099.230 Da |
| Std | 212.489 Da |
| Min | 444.480 Da |
| Max | 1498.880 Da |

### Isoelectric Point

| Stat | Value |
|------|-------|
| Mean | 5.564  |
| Median | 5.508  |
| Std | 1.135  |
| Min | 4.069  |
| Max | 11.334  |

### GRAVY (Hydrophobicity)

| Stat | Value |
|------|-------|
| Mean | -0.122  |
| Median | -0.123  |
| Std | 0.499  |
| Min | -2.250  |
| Max | 2.160  |

### Instability Index

| Stat | Value |
|------|-------|
| Mean | -9.336  |
| Median | -14.060  |
| Std | 17.210  |
| Min | -47.340  |
| Max | 77.080  |

### Net Charge (pH 7)

| Stat | Value |
|------|-------|
| Mean | -0.560  |
| Median | -0.900  |
| Std | 0.808  |
| Min | -3.000  |
| Max | 5.000  |

### Aromaticity

| Stat | Value |
|------|-------|
| Mean | 0.094  |
| Median | 0.091  |
| Std | 0.096  |
| Min | 0.000  |
| Max | 0.333  |

### Stability Classification (Instability Index)

Instability Index < 40 = predicted stable; >= 40 = predicted unstable.

- **Stable (II < 40):** 611 peptides (98.9%)
- **Unstable (II >= 40):** 7 peptides (1.1%)

## 3. Terminal Residue Analysis

Critical for exopeptidase degradation — the exposed N-terminal and C-terminal residues determine how fast aminopeptidases and carboxypeptidases degrade the peptide.

### N-terminal Residue Distribution

| Residue | Count | Aminopeptidase Susceptibility |
|---------|-------|------------------------------|
| A | 2 | 0.9 █████████ |
| D | 1 | 0.3 ███ |
| E | 1 | 0.3 ███ |
| F | 1 | 0.8 ████████ |
| G | 5 | 0.7 ███████ |
| H | 1 | 0.4 ████ |
| I | 3 | 0.7 ███████ |
| K | 364 | 0.7 ███████ |
| L | 2 | 0.9 █████████ |
| M | 1 | 0.8 ████████ |
| N | 1 | 0.5 █████ |
| P | 1 | 0.1 █ |
| Q | 1 | 0.5 █████ |
| R | 229 | 0.6 ██████ |
| S | 1 | 0.6 ██████ |
| T | 1 | 0.6 ██████ |
| V | 1 | 0.7 ███████ |
| W | 1 | 0.5 █████ |
| Y | 1 | 0.6 ██████ |

### C-terminal Residue Distribution

| Residue | Count | Carboxypeptidase Susceptibility |
|---------|-------|---------------------------------|
| A | 14 | 0.8 ████████ |
| D | 12 | 0.2 ██ |
| E | 373 | 0.2 ██ |
| F | 12 | 0.9 █████████ |
| G | 12 | 0.6 ██████ |
| H | 12 | 0.3 ███ |
| I | 12 | 0.7 ███████ |
| K | 16 | 0.6 ██████ |
| L | 12 | 0.8 ████████ |
| M | 12 | 0.7 ███████ |
| N | 12 | 0.4 ████ |
| P | 12 | 0.1  |
| Q | 13 | 0.4 ████ |
| R | 12 | 0.5 █████ |
| S | 14 | 0.5 █████ |
| T | 12 | 0.5 █████ |
| V | 32 | 0.7 ███████ |
| W | 12 | 0.6 ██████ |
| Y | 12 | 0.7 ███████ |

## 4. Exopeptidase Susceptibility Ranking

Combined aminopeptidase + carboxypeptidase susceptibility score (0 = resistant, 1 = highly susceptible).

### Effect of Terminal Modifications on Degradation

The RGEFV libraries systematically test 4 N-terminal and 3 C-terminal modifications:

- **NH2** (free amine) = unprotected, maximally susceptible to aminopeptidases
- **Ac** (acetyl) = blocks aminopeptidase recognition
- **βA** (beta-alanine spacer) = non-natural AA, resists all peptidases
- **Ac-βA** (acetyl + beta-alanine) = maximum N-terminal protection
- **COOH** (free acid) = unprotected C-terminus
- **amide** = partial C-terminal protection
- **C-βA** = maximum C-terminal protection

**Expected protection hierarchy:**
- N-terminal: NH2 < Ac < NH2-βA < Ac-βA (increasing protection)
- C-terminal: COOH < amide < βA (increasing protection)

### Top 20 Most Susceptible to Exopeptidase Degradation

| Rank | Sequence | Notation | N-term AA | C-term AA | Score |
|------|----------|----------|-----------|-----------|-------|
| 1 | βA-βA-βA-βA-βA-βA | NH2-βF-(βA)6-amide | A | A | 0.850 |
| 2 | LRGEFV | Ac-βA-L(Leucine)-RGEFV-βA | L | V | 0.800 |
| 3 | ARGEFV | Ac-βA-A(Alanine)-RGEFV-βA | A | V | 0.800 |
| 4 | MRGEFV | Ac-βA-M(Methionine)-RGEFV-βA | M | V | 0.750 |
| 5 | LIAANK | NH2-LIAANK | L | K | 0.750 |
| 6 | IVKVA | NH2-IVKVA | I | A | 0.750 |
| 7 | FRGEFV | Ac-βA-F(Phenylalanine)-RGEFV-βA | F | V | 0.750 |
| 8 | RGEFVF | NH2-RGEFV-F(Phenylalanine)-amide | R | F | 0.750 |
| 9 | RGEFVF | Ac-βA-RGEFV-F(Phenylalanine)-βA | R | F | 0.750 |
| 10 | RGEFVF | Ac-RGEFV-F(Phenylalanine)-COOH | R | F | 0.750 |
| 11 | RGEFVF | Ac-RGEFV-F(Phenylalanine)-βA | R | F | 0.750 |
| 12 | RGEFVF | NH2-βA-RGEFV-F(Phenylalanine)-amide | R | F | 0.750 |
| 13 | RGEFVF | Ac-βA-RGEFV-F(Phenylalanine)-COOH | R | F | 0.750 |
| 14 | RGEFVF | NH2-βA-RGEFV-F(Phenylalanine)-βA | R | F | 0.750 |
| 15 | RGEFVF | Ac-βA-RGEFV-F(Phenylalanine)-amide | R | F | 0.750 |
| 16 | RGEFVF | NH2-RGEFV-F(Phenylalanine)-βA | R | F | 0.750 |
| 17 | RGEFVF | Ac-RGEFV-F(Phenylalanine)-amide | R | F | 0.750 |
| 18 | RGEFVF | NH2-RGEFV-F(Phenylalanine)-COOH | R | F | 0.750 |
| 19 | RGEFVF | NH2-βA-RGEFV-F(Phenylalanine)-COOH | R | F | 0.750 |
| 20 | IRGEFV | Ac-βA-I(Isoleucine)-RGEFV-βA | I | V | 0.700 |

### Top 20 Most Resistant to Exopeptidase Degradation

| Rank | Sequence | Notation | N-term AA | C-term AA | Score |
|------|----------|----------|-----------|-----------|-------|
| 1 | RGEFVP | NH2-RGEFV-P(Proline)-COOH | R | P | 0.325 |
| 2 | RGEFVP | Ac-RGEFV-P(Proline)-amide | R | P | 0.325 |
| 3 | RGEFVP | NH2-βA-RGEFV-P(Proline)-COOH | R | P | 0.325 |
| 4 | RGEFVP | Ac-RGEFV-P(Proline)-βA | R | P | 0.325 |
| 5 | RGEFVP | NH2-βA-RGEFV-P(Proline)-amide | R | P | 0.325 |
| 6 | RGEFVP | Ac-RGEFV-P(Proline)-COOH | R | P | 0.325 |
| 7 | RGEFVP | Ac-βA-RGEFV-P(Proline)-COOH | R | P | 0.325 |
| 8 | RGEFVP | NH2-βA-RGEFV-P(Proline)-βA | R | P | 0.325 |
| 9 | RGEFVP | NH2-RGEFV-P(Proline)-βA | R | P | 0.325 |
| 10 | RGEFVP | Ac-βA-RGEFV-P(Proline)-amide | R | P | 0.325 |
| 11 | RGEFVP | Ac-βA-RGEFV-P(Proline)-βA | R | P | 0.325 |
| 12 | RGEFVP | NH2-RGEFV-P(Proline)-amide | R | P | 0.325 |
| 13 | RGEFVD | Ac-RGEFV-D(Aspartate)-βA | R | D | 0.400 |
| 14 | RGEFVE | Ac-RGEFV-E(Glutamate)-βA | R | E | 0.400 |
| 15 | RGEFVD | Ac-RGEFV-D(Aspartate)-COOH | R | D | 0.400 |
| 16 | RGEFVD | Ac-RGEFV-D(Aspartate)-amide | R | D | 0.400 |
| 17 | RGEFVD | NH2-βA-RGEFV-D(Aspartate)-COOH | R | D | 0.400 |
| 18 | RGEFVE | NH2-βA-RGEFV-E(Glutamate)-COOH | R | E | 0.400 |
| 19 | RGEFVD | NH2-RGEFV-D(Aspartate)-βA | R | D | 0.400 |
| 20 | RGEFVE | NH2-RGEFV-E(Glutamate)-βA | R | E | 0.400 |

## 5. RGEFV Library Deep Dive (Papers 1 & 2)

**247 peptides** across 13 libraries

### Variable Residue Effect on Properties

Each library tests 19 amino acids at the variable position. This shows how the variable residue affects peptide properties.

| Variable AA | Avg MW (Da) | Avg pI | Avg GRAVY | Avg Instability | Avg Exo Susceptibility |
|-------------|-------------|--------|-----------|-----------------|----------------------|
| A | 677.8 | 6.00 | 0.067 | -18.4 | 0.708 |
| D | 721.8 | 4.65 | -0.817 | -42.5 | 0.408 |
| E | 735.8 | 4.83 | -0.817 | -18.4 | 0.408 |
| F | 753.8 | 6.00 | 0.233 | -18.4 | 0.750 |
| G | 663.7 | 6.00 | -0.300 | -31.4 | 0.608 |
| H | 743.8 | 6.75 | -0.767 | -18.4 | 0.458 |
| I | 719.8 | 6.00 | 0.517 | -18.4 | 0.654 |
| K | 734.8 | 8.75 | -0.883 | -18.6 | 0.608 |
| L | 719.8 | 6.00 | 0.400 | -15.9 | 0.708 |
| M | 737.9 | 5.98 | 0.083 | -19.3 | 0.658 |
| N | 720.8 | 6.00 | -0.817 | -18.4 | 0.508 |
| P | 703.8 | 6.04 | -0.500 | 10.3 | 0.331 |
| Q | 734.8 | 6.00 | -0.817 | -18.4 | 0.508 |
| R | 762.9 | 9.60 | -0.983 | -11.0 | 0.558 |
| S | 693.8 | 5.98 | -0.367 | -15.9 | 0.558 |
| T | 707.8 | 5.98 | -0.350 | -31.4 | 0.558 |
| V | 705.8 | 6.00 | 0.467 | -18.4 | 0.654 |
| W | 792.9 | 6.00 | -0.383 | -18.4 | 0.600 |
| Y | 769.8 | 6.00 | -0.450 | -32.2 | 0.650 |

### N-terminal vs C-terminal Variable Position

Paper 2 tests RGEFV-X (C-terminal variable) vs X-RGEFV (N-terminal variable).

**RGEFV-X** (228 peptides):
- MW: 726.4 +/- 30.7 Da
- pI: 6.24 +/- 1.10
- GRAVY: -0.341 +/- 0.485
- Exopeptidase susceptibility: 0.567

**X-RGEFV** (19 peptides):
- MW: 726.4 +/- 31.5 Da
- pI: 6.19 +/- 1.19
- GRAVY: -0.341 +/- 0.497
- Exopeptidase susceptibility: 0.645

## 6. Crosslinker Library Deep Dive (Paper 3)

**361 peptides** — KLVAD-X1X2-ASAE combinatorial library

These are MMP-cleavable crosslinker peptides for cell-responsive hydrogels. 
The X1X2 dipeptide at the cleavage site determines how fast cells can degrade the gel.

### MMP Cleavage Site Preferences

- Mean MMP cleavage score: 0.455
- Score range: 0.150 - 0.800

### Top 20 Most MMP-Cleavable Variants

| Rank | Dipeptide | Sequence | MMP Score | Bulk | Hydrophobicity |
|------|-----------|----------|-----------|------|----------------|
| 1 | LI | KLVADLIASAE | 0.800 | 0.70 | 4.15 |
| 2 | LL | KLVADLLASAE | 0.800 | 0.70 | 3.80 |
| 3 | IL | KLVADILASAE | 0.750 | 0.70 | 4.15 |
| 4 | II | KLVADIIASAE | 0.750 | 0.70 | 4.50 |
| 5 | ML | KLVADMLASAE | 0.700 | 0.70 | 2.85 |
| 6 | MI | KLVADMIASAE | 0.700 | 0.70 | 3.20 |
| 7 | FI | KLVADFIASAE | 0.700 | 0.80 | 3.65 |
| 8 | LF | KLVADLFASAE | 0.700 | 0.80 | 3.30 |
| 9 | LM | KLVADLMASAE | 0.700 | 0.70 | 2.85 |
| 10 | FL | KLVADFLASAE | 0.700 | 0.80 | 3.30 |
| 11 | LA | KLVADLAASAE | 0.650 | 0.50 | 2.80 |
| 12 | LS | KLVADLSASAE | 0.650 | 0.50 | 1.50 |
| 13 | LW | KLVADLWASAE | 0.650 | 0.85 | 1.45 |
| 14 | WI | KLVADWIASAE | 0.650 | 0.85 | 1.80 |
| 15 | WL | KLVADWLASAE | 0.650 | 0.85 | 1.45 |
| 16 | LT | KLVADLTASAE | 0.650 | 0.55 | 1.55 |
| 17 | AL | KLVADALASAE | 0.650 | 0.50 | 2.80 |
| 18 | AI | KLVADAIASAE | 0.650 | 0.50 | 3.15 |
| 19 | NI | KLVADNIASAE | 0.650 | 0.55 | 0.50 |
| 20 | NL | KLVADNLASAE | 0.650 | 0.55 | 0.15 |

### Top 20 Most MMP-Resistant Variants

| Rank | Dipeptide | Sequence | MMP Score | Bulk | Hydrophobicity |
|------|-----------|----------|-----------|------|----------------|
| 1 | PP | KLVADPPASAE | 0.150 | 0.50 | -1.60 |
| 2 | HP | KLVADHPASAE | 0.200 | 0.55 | -2.40 |
| 3 | DP | KLVADDPASAE | 0.200 | 0.45 | -2.55 |
| 4 | GP | KLVADGPASAE | 0.200 | 0.30 | -1.00 |
| 5 | EP | KLVADEPASAE | 0.250 | 0.50 | -2.55 |
| 6 | PG | KLVADPGASAE | 0.250 | 0.30 | -1.00 |
| 7 | KP | KLVADKPASAE | 0.250 | 0.55 | -2.75 |
| 8 | RP | KLVADRPASAE | 0.250 | 0.60 | -3.05 |
| 9 | PH | KLVADPHASAE | 0.250 | 0.55 | -2.40 |
| 10 | TP | KLVADTPASAE | 0.250 | 0.45 | -1.15 |
| 11 | PD | KLVADPDASAE | 0.250 | 0.45 | -2.55 |
| 12 | PE | KLVADPEASAE | 0.250 | 0.50 | -2.55 |
| 13 | SP | KLVADSPASAE | 0.250 | 0.40 | -1.20 |
| 14 | VP | KLVADVPASAE | 0.250 | 0.55 | 1.30 |
| 15 | HD | KLVADHDASAE | 0.300 | 0.50 | -3.35 |
| 16 | HG | KLVADHGASAE | 0.300 | 0.35 | -1.80 |
| 17 | PV | KLVADPVASAE | 0.300 | 0.55 | 1.30 |
| 18 | DG | KLVADDGASAE | 0.300 | 0.25 | -1.95 |
| 19 | NP | KLVADNPASAE | 0.300 | 0.45 | -2.55 |
| 20 | DD | KLVADDDASAE | 0.300 | 0.40 | -3.50 |

### Key Finding: KLVADLMASAE (Paper 3 Optimized Lead)

- **Dipeptide:** LM (Leu-Met)
- **MMP cleavage score:** 0.700
- **Rank:** 10 / 360
- **Cleavage site bulk:** 0.70
- **Dipeptide hydrophobicity:** 2.85
- **MW:** 1147.3 Da
- **pI:** 4.75
- **GRAVY:** 0.673

The LM dipeptide was identified via split-and-pool screening as optimal for 
cell-mediated hydrogel degradation — balancing MMP accessibility with gel stability.

### Benchmark: GPQGIWGQ (PanMMP crosslinker)

- **MW:** 841.9 Da
- **pI:** 5.53
- **GRAVY:** -0.775
- **Length:** 8 aa
- This is the standard MMP-cleavable crosslinker used in most hydrogel literature.
- Paper 3's KLVADLMASAE was designed to improve on this benchmark.

## 7. Amino Acid Composition Patterns

### Average Amino Acid Composition (all 618 peptides)

| Amino Acid | Avg Fraction | Category |
|------------|-------------|----------|
| A | 17.1137 | Hydrophobic, Small |
| V | 12.9783 | Hydrophobic, Branched |
| E | 12.8812 | Charged- |
| G | 7.8591 | Small |
| R | 7.6750 | Charged+ |
| F | 7.5709 | Hydrophobic, Aromatic |
| K | 6.4164 | Charged+ |
| D | 6.3440 | Charged- |
| S | 6.3117 | Polar, Small |
| L | 6.2594 | Hydrophobic, Branched |
| I | 1.0487 | Hydrophobic, Branched |
| Q | 0.9795 | Polar |
| P | 0.9731 | Hydrophobic |
| N | 0.9490 | Polar |
| W | 0.9445 | Hydrophobic, Aromatic |
| M | 0.9382 | Hydrophobic |
| T | 0.9382 | Polar |
| H | 0.9096 | Charged+ |
| Y | 0.9096 | Aromatic |
| C | 0.0000 |  |

### Composition: RGEFV Libraries vs Crosslinker Library

| AA | RGEFV Libraries | Crosslinker | Enrichment (XL/RGEFV) |
|----|-----------------|-------------|----------------------|
| A | 0.8772 | 28.2296 | 32.18x ** |
| C | 0.0000 | 0.0000 | infx ** |
| D | 0.8772 | 10.0478 | 11.45x ** |
| E | 17.5439 | 10.0478 | 0.57x ** |
| F | 17.5439 | 0.9569 | 0.05x ** |
| G | 17.5439 | 0.9569 | 0.05x ** |
| H | 0.8772 | 0.9569 | 1.09x |
| I | 0.8772 | 0.9569 | 1.09x |
| K | 0.8772 | 10.0478 | 11.45x ** |
| L | 0.8772 | 10.0478 | 11.45x ** |
| M | 0.8772 | 0.9569 | 1.09x |
| N | 0.8772 | 0.9569 | 1.09x |
| P | 0.8772 | 0.9569 | 1.09x |
| Q | 0.8772 | 0.9569 | 1.09x |
| R | 17.5439 | 0.9569 | 0.05x ** |
| S | 0.8772 | 10.0478 | 11.45x ** |
| T | 0.8772 | 0.9569 | 1.09x |
| V | 17.5439 | 10.0478 | 0.57x ** |
| W | 0.8772 | 0.9569 | 1.09x |
| Y | 0.8772 | 0.9569 | 1.09x |

## 8. Physicochemical Category Analysis

### Average Category Fractions by Library Type


**RGEFV-X** (228 peptides):

| Category | Mean Fraction |
|----------|---------------|
| Aromatic | 0.193 |
| Branched | 0.193 |
| Charged- | 0.184 |
| Charged+ | 0.193 |
| Hydrophobic | 0.404 |
| Polar | 0.035 |
| Small | 0.193 |

**X-RGEFV** (19 peptides):

| Category | Mean Fraction |
|----------|---------------|
| Aromatic | 0.193 |
| Branched | 0.193 |
| Charged- | 0.184 |
| Charged+ | 0.193 |
| Hydrophobic | 0.404 |
| Polar | 0.035 |
| Small | 0.193 |

**KLVAD-XX-ASAE** (361 peptides):

| Category | Mean Fraction |
|----------|---------------|
| Aromatic | 0.029 |
| Branched | 0.211 |
| Charged- | 0.201 |
| Charged+ | 0.120 |
| Hydrophobic | 0.531 |
| Polar | 0.129 |
| Small | 0.392 |

## 9. Summary Statistics by Library

| Library | n | Avg MW | Avg pI | Avg GRAVY | Avg II | Avg Exo Susc |
|---------|---|--------|--------|-----------|--------|-------------|
| Ac-RGEFV-X-COOH | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-RGEFV-X-amide | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-RGEFV-X-βA | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-βA-RGEFV-X-COOH | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-βA-RGEFV-X-amide | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-βA-RGEFV-X-βA | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Ac-βA-X-RGEFV-βA | 19 | 726 | 6.19 | -0.341 | -10.6 | 0.645 |
| Additional bioactive | 6 | 664 | 8.04 | -0.801 | 17.2 | 0.667 |
| Crosslinker split-and-pool | 361 | 1142 | 5.05 | 0.037 | -2.8 | 0.450 |
| Internal standards | 1 | 444 | 5.57 | 1.800 | 8.3 | 0.850 |
| NH2-RGEFV-X-COOH | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| NH2-RGEFV-X-amide | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| NH2-RGEFV-X-βA | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| NH2-βA-RGEFV-X-COOH | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| NH2-βA-RGEFV-X-amide | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| NH2-βA-RGEFV-X-βA | 19 | 726 | 6.24 | -0.341 | -20.4 | 0.567 |
| Named sequences | 3 | 1157 | 7.02 | -0.365 | -2.7 | 0.633 |

## 10. Property Correlations

| | mw_da | pI | gravy | instability_index | aromaticity | net_charge_ph7 | total_exopeptidase_susceptibility |
|---|---|---|---|---|---|---|---|
| mw_da | 1.0 | -0.454 | 0.309 | 0.43 | -0.718 | -0.543 | -0.622 |
| pI | -0.454 | 1.0 | -0.422 | -0.159 | 0.36 | 0.893 | 0.437 |
| gravy | 0.309 | -0.422 | 1.0 | 0.124 | -0.247 | -0.299 | 0.103 |
| instability_index | 0.43 | -0.159 | 0.124 | 1.0 | -0.443 | -0.234 | -0.3 |
| aromaticity | -0.718 | 0.36 | -0.247 | -0.443 | 1.0 | 0.474 | 0.597 |
| net_charge_ph7 | -0.543 | 0.893 | -0.299 | -0.234 | 0.474 | 1.0 | 0.519 |
| total_exopeptidase_susceptibility | -0.622 | 0.437 | 0.103 | -0.3 | 0.597 | 0.519 | 1.0 |

### Notable Correlations

- **mw_da** vs **aromaticity**: r = -0.718 (negative)
- **mw_da** vs **net_charge_ph7**: r = -0.543 (negative)
- **mw_da** vs **total_exopeptidase_susceptibility**: r = -0.622 (negative)
- **pI** vs **net_charge_ph7**: r = 0.893 (positive)
- **aromaticity** vs **total_exopeptidase_susceptibility**: r = 0.597 (positive)
- **net_charge_ph7** vs **total_exopeptidase_susceptibility**: r = 0.519 (positive)

## Methods

- Molecular weight, pI, GRAVY, instability index, aromaticity, and secondary structure fractions computed using BioPython ProteinAnalysis
- Hydrophobicity: Kyte-Doolittle scale
- Exopeptidase susceptibility: literature-derived scores for aminopeptidase (N-terminal) and carboxypeptidase (C-terminal) substrate preferences
- MMP cleavage scores: based on published MMP substrate specificity profiles (P1/P1' position preferences)
- Net charge: approximate at pH 7 (K, R = +1; H = +0.1; D, E = -1)
- Instability Index < 40 = predicted stable (Guruprasad et al., 1990)

## Data Sources

1. Rozans SJ et al. ACS Biomater Sci Eng 2024; 10:4916-4926 (RGEFV degradation libraries)
2. Rozans SJ et al. J Biomed Mater Res A 2025; e37864 (LC-MS assay optimization)
3. Wu Y, Rozans SJ et al. Adv Healthcare Mater 2025; e2501932 (crosslinker optimization)
