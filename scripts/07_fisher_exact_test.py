# -*- coding: utf-8 -*-
"""
Created on 2026-03-17

Constructs 2x2 contingency tables and runs Fisher's exact test to assess
whether glycosylation status and LiP structural alteration status are
independent categorical variables (per reviewer comments 19 & 20).

For each tissue, each detected protein is classified as:
  - LiP+  : has >= 1 conformotypic peptide (|log2FC| > 1 AND p < 0.05)
  - LiP-  : detected in LiP-MS but no conformotypic peptides
  - Glyco+: has >= 1 detected N-glycosite in glycoproteomics data
  - Glyco-: no detected N-glycosite

The 2x2 table is:
                 LiP+     LiP-
  Glyco+    |   a    |   b    |
  Glyco-    |   c    |   d    |

Universe = all proteins detected in the LiP-MS proteinGroups file.
"""

import pandas as pd
import numpy as np
import csv
import os
from scipy.stats import fisher_exact

# ============================================================
# USER CONFIGURATION
# Update paths and column names for each tissue before running.
# Run once per tissue (Lung, Heart, Kidney, Spleen).
# ============================================================

tissue_label = 'Lung'  # Change to 'Heart', 'Kidney', or 'Spleen' as needed

# Path to MaxQuant proteinGroups file for this tissue (LiP-MS experiment).
# This defines the full universe of detected proteins.
# Expected column: majority_protein_id_col (see below)
all_proteins_path = r"D:\path\to\proteinGroups_Lung.csv"

# Path to conformotypic peptides file for this tissue.
# These are the LiP+ proteins (already filtered: |log2FC|>1, p<0.05).
# This is the same file used as input to script 02.
# Expected column: lip_protein_id_col (see below)
conformotypic_peptides_path = r"D:\path\to\Conformotypic peptides match_Lung_glyco_short.csv"

# Path to glycoproteomics identifications file for this tissue.
# Should contain one row per identified glycopeptide or glycosite.
# Expected column: glyco_protein_id_col (see below)
glyco_data_path = r"D:\path\to\glyco_identifications_Lung.csv"

# Output directory for results
output_path = r"D:\path\to\output"

# ---- Column name configuration ----
# Column in proteinGroups file containing protein accessions.
# MaxQuant default is 'Majority protein IDs'; accessions are semicolon-separated.
majority_protein_id_col = 'Majority protein IDs'

# Column in conformotypic peptides file containing the protein accession.
lip_protein_id_col = 'Leading razor protein_P'

# Column in glycoproteomics file containing the protein accession.
# Adjust to match actual Byonic / other software output column name.
glyco_protein_id_col = 'Protein'

# ============================================================
# HELPER: extract the primary (first) accession from a field
# that may contain semicolon-separated entries (MaxQuant format)
# ============================================================

def extract_primary_accession(acc_string):
    if pd.isna(acc_string):
        return np.nan
    return str(acc_string).split(';')[0].strip()

# ============================================================
# STEP 1: Load universe of detected proteins from proteinGroups
# ============================================================

all_proteins_df = pd.read_csv(all_proteins_path)

# Filter out contaminants and reverse hits if these columns are present
for col in ['Potential contaminant', 'Reverse', 'Only identified by site']:
    if col in all_proteins_df.columns:
        all_proteins_df = all_proteins_df[all_proteins_df[col].isna() | (all_proteins_df[col] != '+')]

all_proteins_df['protein_id'] = all_proteins_df[majority_protein_id_col].apply(extract_primary_accession)
universe = set(all_proteins_df['protein_id'].dropna().unique())
print(f"[{tissue_label}] Total detected proteins (universe): {len(universe)}")

# ============================================================
# STEP 2: Load LiP+ proteins from conformotypic peptides file
# ============================================================

conformotypic_df = pd.read_csv(conformotypic_peptides_path)
lip_positive = set(
    conformotypic_df[lip_protein_id_col].dropna()
    .apply(extract_primary_accession)
    .unique()
)
print(f"[{tissue_label}] LiP+ proteins (conformotypic): {len(lip_positive)}")

# ============================================================
# STEP 3: Load Glyco+ proteins from glycoproteomics data
# ============================================================

glyco_df = pd.read_csv(glyco_data_path)
glyco_positive = set(
    glyco_df[glyco_protein_id_col].dropna()
    .apply(extract_primary_accession)
    .unique()
)
print(f"[{tissue_label}] Glyco+ proteins (N-glycosylated): {len(glyco_positive)}")

# ============================================================
# STEP 4: Build 2x2 contingency table
# Universe is restricted to proteins detected in the LiP-MS run.
# Glyco+ status is assessed against the glycoproteomics dataset.
# ============================================================

a = len([p for p in universe if p in lip_positive and p in glyco_positive])   # LiP+, Glyco+
b = len([p for p in universe if p not in lip_positive and p in glyco_positive]) # LiP-, Glyco+
c = len([p for p in universe if p in lip_positive and p not in glyco_positive]) # LiP+, Glyco-
d = len([p for p in universe if p not in lip_positive and p not in glyco_positive]) # LiP-, Glyco-

contingency_table = [[a, b],
                     [c, d]]

print(f"\n[{tissue_label}] 2x2 Contingency Table:")
print(f"                     LiP+        LiP-     Total")
print(f"  Glyco+         {a:>8}    {b:>8}    {a+b:>8}")
print(f"  Glyco-         {c:>8}    {d:>8}    {c+d:>8}")
print(f"  Total          {a+c:>8}    {b+d:>8}    {a+b+c+d:>8}")

# ============================================================
# STEP 5: Fisher's exact test
# ============================================================

odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

print(f"\n[{tissue_label}] Fisher's Exact Test:")
print(f"  Odds ratio : {odds_ratio:.4f}")
print(f"  p-value    : {p_value:.4e}")

if p_value < 0.05:
    print(f"  Interpretation: Glycosylation and LiP structural alteration are NOT independent (p < 0.05).")
else:
    print(f"  Interpretation: No significant dependence detected between glycosylation and LiP structural alteration (p >= 0.05).")

# ============================================================
# STEP 6: Save results
# ============================================================

results_df = pd.DataFrame({
    'Tissue':           [tissue_label],
    'a_LiP+_Glyco+':   [a],
    'b_LiP-_Glyco+':   [b],
    'c_LiP+_Glyco-':   [c],
    'd_LiP-_Glyco-':   [d],
    'Total_proteins':   [a + b + c + d],
    'Odds_Ratio':       [odds_ratio],
    'p_value_Fisher':   [p_value],
})

os.makedirs(output_path, exist_ok=True)
out_file = os.path.join(output_path, f'fisher_exact_results_{tissue_label}.csv')
with open(out_file, 'w', newline='') as f:
    results_df.to_csv(f, index=False)

print(f"\nResults saved to: {out_file}")
