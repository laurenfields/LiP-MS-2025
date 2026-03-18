# -*- coding: utf-8 -*-
"""
Created on 2026-03-18

Constructs a 2x2 contingency table and runs Fisher's exact test to assess
whether glycosylation status (UniProt-annotated) and LiP structural alteration
status are independent categorical variables (per reviewer comments 19 & 20).

Glycosylation is determined from UniProt annotations only (no experimental
glycoproteomics data required). A protein is Glyco+ if it has at least one
CARBOHYD entry in the UniProt Glycosylation field.

For each tissue, each detected protein is classified as:
  - LiP+  : has >= 1 conformotypic peptide (|log2FC| > 1 AND p < 0.05)
  - LiP-  : detected in LiP-MS but no conformotypic peptides
  - Glyco+: has >= 1 N-glycosylation annotation in UniProt
  - Glyco-: no N-glycosylation annotation in UniProt

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
import warnings
from scipy.stats import fisher_exact
from bioservices import UniProt
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# USER CONFIGURATION
# Update paths for each tissue before running.
# Run once per tissue (Lung, Heart, Kidney, Spleen).
# ============================================================

tissue_label = 'Lung'  # Change to 'Heart', 'Kidney', or 'Spleen' as needed

# Path to MaxQuant proteinGroups file for this tissue (LiP-MS experiment).
# This defines the full universe of detected proteins.
all_proteins_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\08_fisher_exact\proteinGroups_Lung 1.csv"

# Path to conformotypic peptides file for this tissue.
# These are the LiP+ proteins (already filtered: |log2FC|>1, p<0.05).
# This is the same file used as input to script 02.
conformotypic_peptides_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\08_fisher_exact\Conformotypic peptides match_Lung_glyco_short.csv"

# Output directory for results
output_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\08_fisher_exact"

# ---- Column name configuration ----
# Column in proteinGroups file containing protein accessions.
# MaxQuant default is 'Majority protein IDs'; accessions are semicolon-separated.
majority_protein_id_col = 'Majority protein IDs'

# Column in conformotypic peptides file containing the protein accession.
lip_protein_id_col = 'Leading razor protein_P'

# ---- UniProt cache ----
# If a cache file already exists at this path, UniProt will NOT be queried again.
# Delete or rename the cache file to force a fresh UniProt lookup.
# Leave as None to auto-generate a path in the output directory.
uniprot_cache_path = None  # e.g. r"D:\path\to\uniprot_glyco_cache_Lung.csv"

# ============================================================
# HELPER: extract the primary (first) accession from a field
# that may contain semicolon-separated entries (MaxQuant format)
# ============================================================

def extract_primary_accession(acc_string):
    if pd.isna(acc_string):
        return np.nan
    return str(acc_string).split(';')[0].strip()

# ============================================================
# HELPER: query UniProt for glycosylation status
# Returns True if the protein has any CARBOHYD annotation.
# ============================================================

def has_uniprot_glycosylation(prot_query):
    """
    Query UniProt for the Glycosylation field of a single protein accession.
    Returns True if at least one glycosylation (CARBOHYD) entry is annotated.
    Returns False if none found or if the query fails.
    """
    try:
        u = UniProt()
        res = u.get_df(prot_query.split())
        if res is None or res.empty:
            return False
        glyco = res['Glycosylation'].iloc[0]
        if isinstance(glyco, float) and np.isnan(glyco):
            return False
        # CARBOHYD entries are present if the field is not NaN
        return 'CARBOHYD' in str(glyco)
    except Exception as e:
        print(f"  Warning: UniProt query failed for {prot_query}: {e}")
        return False

# ============================================================
# STEP 1: Load universe of detected proteins from proteinGroups
# ============================================================

all_proteins_df = pd.read_csv(all_proteins_path)

# Filter out contaminants and reverse hits if these columns are present
for col in ['Potential contaminant', 'Reverse', 'Only identified by site']:
    if col in all_proteins_df.columns:
        all_proteins_df = all_proteins_df[
            all_proteins_df[col].isna() | (all_proteins_df[col] != '+')
        ]

all_proteins_df['protein_id'] = all_proteins_df[majority_protein_id_col].apply(
    extract_primary_accession
)
universe = sorted(set(all_proteins_df['protein_id'].dropna().unique()))
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
# STEP 3: Determine Glyco+ proteins via UniProt annotations
# Results are cached to avoid repeated queries on re-runs.
# ============================================================

os.makedirs(output_path, exist_ok=True)

if uniprot_cache_path is None:
    uniprot_cache_path = os.path.join(output_path, f'uniprot_glyco_cache_{tissue_label}.csv')

if os.path.exists(uniprot_cache_path):
    print(f"[{tissue_label}] Loading UniProt glyco cache from: {uniprot_cache_path}")
    cache_df = pd.read_csv(uniprot_cache_path)
    glyco_lookup = dict(zip(cache_df['protein_id'], cache_df['is_glyco_positive']))
    # Query any proteins not yet in the cache
    missing = [p for p in universe if p not in glyco_lookup]
else:
    glyco_lookup = {}
    missing = universe

if missing:
    print(f"[{tissue_label}] Querying UniProt for {len(missing)} proteins (this may take several minutes)...")
    for i, prot in enumerate(missing):
        glyco_lookup[prot] = has_uniprot_glycosylation(prot)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1} / {len(missing)} queried")

    # Save updated cache
    cache_df = pd.DataFrame({
        'protein_id': list(glyco_lookup.keys()),
        'is_glyco_positive': list(glyco_lookup.values())
    })
    cache_df.to_csv(uniprot_cache_path, index=False)
    print(f"[{tissue_label}] UniProt cache saved to: {uniprot_cache_path}")

glyco_positive = set(p for p in universe if glyco_lookup.get(p, False))
print(f"[{tissue_label}] Glyco+ proteins (UniProt-annotated): {len(glyco_positive)}")

# ============================================================
# STEP 4: Build 2x2 contingency table
# Universe is restricted to proteins detected in the LiP-MS run.
# ============================================================

a = len([p for p in universe if p in lip_positive and p in glyco_positive])      # LiP+, Glyco+
b = len([p for p in universe if p not in lip_positive and p in glyco_positive])   # LiP-, Glyco+
c = len([p for p in universe if p in lip_positive and p not in glyco_positive])   # LiP+, Glyco-
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

out_file = os.path.join(output_path, f'fisher_exact_results_{tissue_label}.csv')
with open(out_file, 'w', newline='') as f:
    results_df.to_csv(f, index=False)

print(f"\nResults saved to: {out_file}")
