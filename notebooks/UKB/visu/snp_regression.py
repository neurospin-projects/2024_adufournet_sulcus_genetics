import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

"""
cd /home/ad279118
sshfs dufourna@irene-fr.ccc.cea.fr:/ccc/workflash/cont003/n4h00001/n4h00001/25irene_AD_UKB_TIV/results/Champollion_V1_32 tmp
"""

base_path = "/home/ad279118/tmp"
region_model = "SC-SPeC_left/name22-16-47_177/32PCs/White"
resid_pheno_name = "resid_pheno_IID.txt"
extracted_genotyping_data = "/volatile/ad279118/2024_adufournet_sulcus_genetics/notebooks/UKB/visu/rs2033939_A.raw"

"""
h2                           
maf-0.01_minp_perm.zmat.tsv  maf-0.01_most_perm.zmat.tsv
FUMA
maf-0.01.most_orig.sumstats
MAGMA
resid_pheno.txt
GWAS_uni
maf-0.01_minp_orig.zmat.tsv
maf-0.01_most_orig.zmat.tsv
manhattan_plot.png
"""
### PHENOTYPE
pre_residualized_bdd = pd.read_csv(f'{base_path}/{region_model}/{resid_pheno_name}', sep='\t')
pre_residualized_bdd = pre_residualized_bdd.drop(['FID', 'session_id'], axis=1)
pre_residualized_bdd = pre_residualized_bdd.set_index('IID')
print(pre_residualized_bdd.shape, '\n')
pre_residualized_bdd = pre_residualized_bdd[[f'dim{i}_res' for i in range(1,pre_residualized_bdd.shape[1]+1)]]
print(pre_residualized_bdd.iloc[:,:5].head(), '\n')

### GENOTYPE
genotype = pd.read_csv(extracted_genotyping_data, sep='\s+', engine="python") 
print(genotype.head(), '\n')
print(genotype[['IID','SEX','rs2033939_A']].head(), '\n')

###REGRESSION
bdd_geno = pd.merge(pre_residualized_bdd, genotype, on='IID', how='inner')
bdd_geno = bdd_geno.dropna()

# Features (latent dimensions) and target (genotype)
X = bdd_geno[[f'dim{i}_res' for i in range(1,pre_residualized_bdd.shape[1]+1)]]  # Select all latent dimensions

for SELECTED_SNP in genotype.drop(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], axis=1).keys():
    print(SELECTED_SNP, "\n")
    y = bdd_geno[SELECTED_SNP]         # Genotype values (0, 1, or 2)
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    print("R-squared (uncentered):", res.rsquared)
    print("Prob (F-statistic):", res.f_pvalue, "\n")
    merged = pd.DataFrame({"IID":bdd_geno.IID, SELECTED_SNP:y,"prediction":res.predict()}).sort_values('prediction')

    plt.figure(figsize=(8, 6))
    sns.violinplot(x=SELECTED_SNP, y='prediction', data=merged, hue=SELECTED_SNP)

    plt.xlabel('Number of A Alleles')
    plt.ylabel('prediction Value')
    plt.title(f'Violin Plot of prediction vs. Number of {SELECTED_SNP} Alleles')
    plt.legend(loc='upper right')
    plt.show()

umap_bool = False
if umap_bool:
    import umap

    reducer = umap.UMAP()
    umap_bdd = reducer.fit_transform(X)
    plt.scatter(
        umap_bdd[:, 0],
        umap_bdd[:, 1],
        c=[sns.color_palette()[int(g)] for g in y],
        s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the encoded sulcal shapes', fontsize=24)
    plt.show()
