# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:08:40 2022

@author: lawashburn
"""

import pandas as pd
import csv
import math
import scipy
from scipy.stats import ttest_ind
import numpy as np

peptide_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\peptides_Lung 1.csv" #Sample names associated with protein will end in T
protein_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\proteinGroups_Lung 1.csv" #Sample names associated with protein will end in P
output_path = r"F:\postDefenseDataTransfer\Backup_D\Manuscripts\2024_Haiyan_VitaminA\rebuttal\code_test\01_fold_change_assessment_old"
experimental_sample_name_prefix = 'SRT_L'
control_sample_name_prefix = 'WT_L'
lfq_prefix ='LFQ intensity ' # prefix before sample name in LFQ columns
protein_suffix = '_T'
peptide_suffix = '_P'
merge_col_1 = 'Protein names'
merge_col_2 = 'Gene names'

p_cutoff = 0.05
min_FC = 2
min_log_FC = 1

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

#Determine fold change based on protein level results
protein_report = pd.read_csv(protein_path)
print([c for c in protein_report.columns if 'LFQ' in c])
protein_report = protein_report.replace(0, np.nan) #replace empty values with NaN so mean can be taken without impact from 0s
protein_report['Exp mean protein'] = protein_report[[exp_1_name_protein, exp_2_name_protein,exp_3_name_protein]].mean(axis=1,skipna=True) #the mean is taken without considering 0s
                                                                                                                                        
protein_report['Ctrl mean protein'] = protein_report[[ctrl_1_name_protein, ctrl_2_name_protein,ctrl_3_name_protein]].mean(axis=1,skipna=True) #the mean is taken without considering 0s
                                                                                                                                        
protein_report['protein fold change'] = protein_report['Exp mean protein']/protein_report['Ctrl mean protein']
protein_report['Log2(protein fold change)'] = abs(np.log2(protein_report['protein fold change']))

#Determine p-value between ctrl and exp peptides
peptide_report = pd.read_csv(peptide_path)
merged_pep_prot_report = peptide_report.merge(protein_report, on=[merge_col_1, merge_col_2], how='left')

print('Number of protein entries: ',len(protein_report))
print('Number of peptide entries: ',len(peptide_report))
print('Number of entries post-protein/peptide merge: ',len(merged_pep_prot_report))

merged_pep_prot_report['protein p-value'] = ttest_ind(merged_pep_prot_report[[exp_1_name_protein, exp_2_name_protein,exp_3_name_protein]], 
                                merged_pep_prot_report[[ctrl_1_name_protein, ctrl_2_name_protein,ctrl_3_name_protein]], axis=1, equal_var=True, alternative='two-sided',nan_policy='omit')[1]


merged_pep_prot_report['unaltered fold change (SetA)'] = merged_pep_prot_report['protein fold change'] #in this column the FC will not be changed regardless of p-value
merged_pep_prot_report['p-value filtered protein fold change (SetB)'] = merged_pep_prot_report['protein fold change'] #in this column the FC will be equal to 1 if p-value is > 0.05
merged_pep_prot_report['p-value OR log2(FC) filtered protein fold change (SetC)'] = merged_pep_prot_report['protein fold change'] #in this column the FC will be equal to 1 if p-value is > 0.05 OR the FC is insignificant
merged_pep_prot_report['p-value AND log2(FC) filtered protein fold change (SetD)'] = merged_pep_prot_report['protein fold change'] #in this column the FC will be equal to 1 if p-value is > 0.05 AND the FC is insignificant

merged_pep_prot_report.loc[merged_pep_prot_report['protein p-value'] > p_cutoff, ['p-value filtered protein fold change (SetB)']] = 1
merged_pep_prot_report.loc[merged_pep_prot_report['protein p-value'].isnull(), ['p-value filtered protein fold change (SetB)']] = 1

merged_pep_prot_report.loc[merged_pep_prot_report['protein p-value'] > p_cutoff, ['p-value OR log2(FC) filtered protein fold change (SetC)']] = 1
merged_pep_prot_report.loc[merged_pep_prot_report['protein p-value'].isnull(), ['p-value OR log2(FC) filtered protein fold change (SetC)']] = 1
merged_pep_prot_report.loc[abs(merged_pep_prot_report['Log2(protein fold change)']) < min_log_FC, ['p-value OR log2(FC) filtered protein fold change (SetC)']] = 1


merged_pep_prot_report.loc[abs((merged_pep_prot_report['Log2(protein fold change)'])<min_log_FC) & (merged_pep_prot_report['protein p-value']> p_cutoff),
                    ['p-value AND log2(FC) filtered protein fold change (SetD)']] = 1
merged_pep_prot_report.loc[abs((merged_pep_prot_report['Log2(protein fold change)'])<min_log_FC) & (merged_pep_prot_report['protein p-value'].isnull()),
                    ['p-value AND log2(FC) filtered protein fold change (SetD)']] = 1


merged_pep_prot_report = merged_pep_prot_report.replace(0, np.nan) #replace empty values with NaN so mean can be taken without impact from 0s
merged_pep_prot_report['Ctrl mean peptide'] = merged_pep_prot_report[[ctrl_1_name_peptide, ctrl_2_name_peptide,ctrl_3_name_peptide]].mean(axis=1,
                                                                                                                                        skipna=True) #the mean is taken without considering 0s
merged_pep_prot_report = merged_pep_prot_report.replace(0, np.nan) #replace empty values with NaN so mean can be taken without impact from 0s
merged_pep_prot_report['SetA (LFQ intensity Exp1 normalized-peptide)'] = (merged_pep_prot_report[exp_1_name_peptide] / 
                                                                                   merged_pep_prot_report['unaltered fold change (SetA)'])
merged_pep_prot_report['SetA (LFQ intensity Exp2 normalized-peptide)'] = (merged_pep_prot_report[exp_2_name_peptide] / 
                                                                                   merged_pep_prot_report['unaltered fold change (SetA)'])
merged_pep_prot_report['SetA (LFQ intensity Exp3 normalized-peptide)'] = (merged_pep_prot_report[exp_3_name_peptide] / 
                                                                                   merged_pep_prot_report['unaltered fold change (SetA)'])


merged_pep_prot_report['SetB (LFQ intensity Exp1 normalized-peptide)'] = (merged_pep_prot_report[exp_1_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value filtered protein fold change (SetB)'])
merged_pep_prot_report['SetB (LFQ intensity Exp2 normalized-peptide)'] = (merged_pep_prot_report[exp_2_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value filtered protein fold change (SetB)'])
merged_pep_prot_report['SetB (LFQ intensity Exp3 normalized-peptide)'] = (merged_pep_prot_report[exp_3_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value filtered protein fold change (SetB)'])

merged_pep_prot_report['SetC (LFQ intensity Exp1 normalized-peptide)'] = (merged_pep_prot_report[exp_1_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value OR log2(FC) filtered protein fold change (SetC)'])
merged_pep_prot_report['SetC (LFQ intensity Exp2 normalized-peptide)'] = (merged_pep_prot_report[exp_2_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value OR log2(FC) filtered protein fold change (SetC)'])
merged_pep_prot_report['SetC (LFQ intensity Exp3 normalized-peptide)'] = (merged_pep_prot_report[exp_3_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value OR log2(FC) filtered protein fold change (SetC)'])

merged_pep_prot_report['SetD (LFQ intensity Exp1 normalized-peptide)'] = (merged_pep_prot_report[exp_1_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value AND log2(FC) filtered protein fold change (SetD)'])
merged_pep_prot_report['SetD (LFQ intensity Exp2 normalized-peptide)'] = (merged_pep_prot_report[exp_2_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value AND log2(FC) filtered protein fold change (SetD)'])
merged_pep_prot_report['SetD (LFQ intensity Exp3 normalized-peptide)'] = (merged_pep_prot_report[exp_3_name_peptide] / 
                                                                                   merged_pep_prot_report['p-value AND log2(FC) filtered protein fold change (SetD)'])
merged_pep_prot_report = merged_pep_prot_report.replace(0, np.nan) #replace empty values with NaN so mean can be taken without impact from 0s
merged_pep_prot_report['SetA exp mean intensity peptide'] = merged_pep_prot_report[['SetA (LFQ intensity Exp1 normalized-peptide)', 
                                                                                                            'SetA (LFQ intensity Exp2 normalized-peptide)',
                                                                                                            'SetA (LFQ intensity Exp3 normalized-peptide)']].mean(axis=1,
                                                                                                                                        skipna=True) #the mean is taken without considering 0s

merged_pep_prot_report['SetB exp mean intensity peptide'] = merged_pep_prot_report[['SetB (LFQ intensity Exp1 normalized-peptide)', 
                                                                                                            'SetB (LFQ intensity Exp2 normalized-peptide)',
                                                                                                            'SetB (LFQ intensity Exp3 normalized-peptide)']].mean(axis=1,
                                                                                                                                        skipna=True) #the mean is taken without considering 0s                                                                                                                                                              
    
merged_pep_prot_report['SetC exp mean intensity peptide'] = merged_pep_prot_report[['SetC (LFQ intensity Exp1 normalized-peptide)', 
                                                                                                            'SetC (LFQ intensity Exp2 normalized-peptide)',
                                                                                                            'SetC (LFQ intensity Exp3 normalized-peptide)']].mean(axis=1,
                                                                                                                                        skipna=True) #the mean is taken without considering 0s    
  
merged_pep_prot_report['SetD exp mean intensity peptide'] = merged_pep_prot_report[['SetD (LFQ intensity Exp1 normalized-peptide)', 
                                                                                                            'SetD (LFQ intensity Exp2 normalized-peptide)',
                                                                                                            'SetD (LFQ intensity Exp3 normalized-peptide)']].mean(axis=1,
                                                                                                                                        skipna=True) #the mean is taken without considering 0s                                                                                                                                                                 
    
merged_pep_prot_report['SetA exp mean fold change peptide normalized'] = merged_pep_prot_report['SetA exp mean intensity peptide']/merged_pep_prot_report['Ctrl mean peptide']                                                                                                                                                             
merged_pep_prot_report['SetB exp mean fold change peptide normalized'] = merged_pep_prot_report['SetB exp mean intensity peptide']/merged_pep_prot_report['Ctrl mean peptide'] 
merged_pep_prot_report['SetC exp mean fold change peptide normalized'] = merged_pep_prot_report['SetC exp mean intensity peptide']/merged_pep_prot_report['Ctrl mean peptide'] 
merged_pep_prot_report['SetD exp mean fold change peptide normalized'] = merged_pep_prot_report['SetD exp mean intensity peptide']/merged_pep_prot_report['Ctrl mean peptide'] 

                                                                                                                                                              
file_out_path = output_path + '\\Updated_report_20221206.csv'
with open(file_out_path,'w',newline='') as filec:
        writerc = csv.writer(filec)
        merged_pep_prot_report.to_csv(filec,index=False)