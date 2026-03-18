# -*- coding: utf-8 -*-
"""
Created on 2026-03-18

Extension of 01_fold_change_assessment.py that adds peptide-level t-tests
with Benjamini-Hochberg (BH) multiple hypothesis testing correction.

Replaces the manual Excel t-test step used in the original pipeline.
Protein normalization logic (Sets A-D) is identical to script 01.

Additional output columns (appended to the same report):
  - 'peptide p-value (raw)'     : two-sided t-test on BH-normalized peptide LFQ intensities
  - 'peptide q-value (BH)'      : Benjamini-Hochberg adjusted p-value
  - 'Log2(peptide fold change normalized SetD)' : log2FC of protein-normalized peptide (Set D)

The recommended filter for conformotypic peptide selection is:
  |Log2(peptide fold change normalized SetD)| > 1  AND  peptide q-value (BH) < 0.05
"""

import pandas as pd
import csv
import math
import scipy
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np

# ============================================================
# USER CONFIGURATION
# ============================================================

peptide_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\peptides_Lung 1.csv" #Sample names associated with protein will end in T
protein_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\proteinGroups_Lung 1.csv" #Sample names associated with protein will end in P
output_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\01b_fold_change_assessment_BH_new"

experimental_sample_name_prefix = 'SRT_L'
control_sample_name_prefix      = 'WT_L'

lfq_prefix ='LFQ intensity ' # prefix before sample name in LFQ columns
protein_suffix = '_T'
peptide_suffix = '_P'
merge_col_1 = 'Protein names'
merge_col_2 = 'Gene names'

p_cutoff    = 0.05
min_log_FC  = 1

# ============================================================
# COLUMN NAME SETUP  (mirrors script 01)
# ============================================================

exp_1_name_protein = lfq_prefix + experimental_sample_name_prefix + '1' + protein_suffix
exp_2_name_protein = lfq_prefix + experimental_sample_name_prefix + '2' + protein_suffix
exp_3_name_protein = lfq_prefix + experimental_sample_name_prefix + '3' + protein_suffix
ctrl_1_name_protein = lfq_prefix + control_sample_name_prefix + '1' + protein_suffix
ctrl_2_name_protein = lfq_prefix + control_sample_name_prefix + '2' + protein_suffix
ctrl_3_name_protein = lfq_prefix + control_sample_name_prefix + '3' + protein_suffix
exp_1_name_peptide = lfq_prefix + experimental_sample_name_prefix + '1' + peptide_suffix
exp_2_name_peptide = lfq_prefix + experimental_sample_name_prefix + '2' + peptide_suffix
exp_3_name_peptide = lfq_prefix + experimental_sample_name_prefix + '3' + peptide_suffix
ctrl_1_name_peptide = lfq_prefix + control_sample_name_prefix + '1' + peptide_suffix
ctrl_2_name_peptide = lfq_prefix + control_sample_name_prefix + '2' + peptide_suffix
ctrl_3_name_peptide = lfq_prefix + control_sample_name_prefix + '3' + peptide_suffix

# ============================================================
# STEP 1: Protein-level fold change and normalization
# (identical to script 01)
# ============================================================

protein_report = pd.read_csv(protein_path)
protein_report = protein_report.replace(0, np.nan)

protein_report['Exp mean protein']  = protein_report[[exp_1_name_protein, exp_2_name_protein, exp_3_name_protein]].mean(axis=1, skipna=True)
protein_report['Ctrl mean protein'] = protein_report[[ctrl_1_name_protein, ctrl_2_name_protein, ctrl_3_name_protein]].mean(axis=1, skipna=True)
protein_report['protein fold change']        = protein_report['Exp mean protein'] / protein_report['Ctrl mean protein']
protein_report['Log2(protein fold change)']  = abs(np.log2(protein_report['protein fold change']))

peptide_report = pd.read_csv(peptide_path)
merged = peptide_report.merge(protein_report, on=['Protein names', 'Gene names'], how='left')

print('Number of protein entries: ', len(protein_report))
print('Number of peptide entries: ', len(peptide_report))
print('Number of entries post-merge: ', len(merged))

merged['protein p-value'] = ttest_ind(
    merged[[exp_1_name_protein, exp_2_name_protein, exp_3_name_protein]],
    merged[[ctrl_1_name_protein, ctrl_2_name_protein, ctrl_3_name_protein]],
    axis=1, equal_var=False, alternative='two-sided', nan_policy='omit'  # Welch's t-test
)[1]

# Sets A-D protein normalization factors (identical to script 01)
merged['unaltered fold change (SetA)']                           = merged['protein fold change']
merged['p-value filtered protein fold change (SetB)']           = merged['protein fold change']
merged['p-value OR log2(FC) filtered protein fold change (SetC)'] = merged['protein fold change']
merged['p-value AND log2(FC) filtered protein fold change (SetD)'] = merged['protein fold change']

merged.loc[merged['protein p-value'] > p_cutoff,  'p-value filtered protein fold change (SetB)'] = 1
merged.loc[merged['protein p-value'].isnull(),     'p-value filtered protein fold change (SetB)'] = 1

merged.loc[merged['protein p-value'] > p_cutoff,  'p-value OR log2(FC) filtered protein fold change (SetC)'] = 1
merged.loc[merged['protein p-value'].isnull(),     'p-value OR log2(FC) filtered protein fold change (SetC)'] = 1
merged.loc[abs(merged['Log2(protein fold change)']) < min_log_FC, 'p-value OR log2(FC) filtered protein fold change (SetC)'] = 1

merged.loc[
    abs(merged['Log2(protein fold change)'] < min_log_FC) & (merged['protein p-value'] > p_cutoff),
    'p-value AND log2(FC) filtered protein fold change (SetD)'
] = 1
merged.loc[
    abs(merged['Log2(protein fold change)'] < min_log_FC) & (merged['protein p-value'].isnull()),
    'p-value AND log2(FC) filtered protein fold change (SetD)'
] = 1

merged = merged.replace(0, np.nan)

# Normalized peptide intensities (all four sets)
for label, fc_col in [
    ('SetA', 'unaltered fold change (SetA)'),
    ('SetB', 'p-value filtered protein fold change (SetB)'),
    ('SetC', 'p-value OR log2(FC) filtered protein fold change (SetC)'),
    ('SetD', 'p-value AND log2(FC) filtered protein fold change (SetD)'),
]:
    merged[f'{label} (LFQ intensity Exp1 normalized-peptide)'] = merged[exp_1_name_peptide] / merged[fc_col]
    merged[f'{label} (LFQ intensity Exp2 normalized-peptide)'] = merged[exp_2_name_peptide] / merged[fc_col]
    merged[f'{label} (LFQ intensity Exp3 normalized-peptide)'] = merged[exp_3_name_peptide] / merged[fc_col]

merged = merged.replace(0, np.nan)

merged['Ctrl mean peptide'] = merged[[ctrl_1_name_peptide, ctrl_2_name_peptide, ctrl_3_name_peptide]].mean(axis=1, skipna=True)

for label in ['SetA', 'SetB', 'SetC', 'SetD']:
    exp_cols = [f'{label} (LFQ intensity Exp{i} normalized-peptide)' for i in [1, 2, 3]]
    merged[f'{label} exp mean intensity peptide'] = merged[exp_cols].mean(axis=1, skipna=True)
    merged[f'{label} exp mean fold change peptide normalized'] = (
        merged[f'{label} exp mean intensity peptide'] / merged['Ctrl mean peptide']
    )

# ============================================================
# STEP 2: Peptide-level t-test on Set D normalized intensities
# Set D is recommended: corrects for protein abundance only when
# both FC and p-value criteria are met at the protein level.
# ============================================================

setD_exp_cols  = [f'SetD (LFQ intensity Exp{i} normalized-peptide)' for i in [1, 2, 3]]
ctrl_pep_cols  = [ctrl_1_name_peptide, ctrl_2_name_peptide, ctrl_3_name_peptide]

raw_p = ttest_ind(
    merged[setD_exp_cols],
    merged[ctrl_pep_cols],
    axis=1, equal_var=False, alternative='two-sided', nan_policy='omit'  # Welch's t-test
)[1]

merged['peptide p-value (raw)'] = raw_p

# ============================================================
# STEP 3: Benjamini-Hochberg correction across all peptides
# NaN p-values (insufficient observations) are excluded from
# correction then re-inserted as NaN.
# ============================================================

valid_mask   = ~merged['peptide p-value (raw)'].isna()
p_valid      = merged.loc[valid_mask, 'peptide p-value (raw)'].values

_, q_valid, _, _ = multipletests(p_valid, alpha=0.05, method='fdr_bh')

merged['peptide q-value (BH)'] = np.nan
merged.loc[valid_mask, 'peptide q-value (BH)'] = q_valid

n_total    = valid_mask.sum()
n_sig_raw  = (merged.loc[valid_mask, 'peptide p-value (raw)'] < p_cutoff).sum()
n_sig_bh   = (merged.loc[valid_mask, 'peptide q-value (BH)']  < p_cutoff).sum()
print(f'\nPeptide-level statistics (Set D normalized):')
print(f'  Peptides with valid p-value : {n_total}')
print(f'  Significant (raw p < 0.05)  : {n_sig_raw}')
print(f'  Significant (BH q < 0.05)   : {n_sig_bh}')

# ============================================================
# STEP 4: Log2 fold change of normalized peptide (Set D)
# ============================================================

merged['Log2(peptide fold change normalized SetD)'] = np.log2(
    merged['SetD exp mean fold change peptide normalized']
)

# ============================================================
# STEP 5: Save output
# ============================================================

import os
os.makedirs(output_path, exist_ok=True)

file_out_path = os.path.join(output_path, 'Updated_report_BH_corrected.csv')
with open(file_out_path, 'w', newline='') as f:
    merged.to_csv(f, index=False)

print(f'\nOutput saved to: {file_out_path}')
print(f'\nRecommended conformotypic peptide filter:')
print(f'  |Log2(peptide fold change normalized SetD)| > {min_log_FC}  AND  peptide q-value (BH) < {p_cutoff}')
