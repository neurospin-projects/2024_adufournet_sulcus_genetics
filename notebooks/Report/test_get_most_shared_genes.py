import pandas as pd
from collections import Counter
import os

path_to_Champollion='/home/ad279118/tmp1'
folder='32PCs'
with open(f"{path_to_Champollion}/list_model.txt") as f:
    regions_models = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

gene_counter = Counter()
bonferroni_threshold = 0.05 / 19264

for region_model in regions_models:
    region, model = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{folder}/white.British.ancestry")
    magma_path = os.path.join(base_path, "MAGMA")
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    
    if os.path.exists(magma_genes_file):
        print(f"Processing MAGMA genes for {region_model}...")
        genes_df = pd.read_csv(magma_genes_file, sep="\s+", comment="#")
        
        # Filter genes by the Bonferroni-corrected p-value threshold
        significant_genes = genes_df[genes_df["P"] < bonferroni_threshold]
        
        # Update the gene counter with the significant genes from this region
        gene_counter.update(significant_genes["GENE"])

# Convert the Counter to a DataFrame for easier manipulation
gene_counts_df = pd.DataFrame(gene_counter.items(), columns=["Gene", "Count"])

# Sort the DataFrame by the count in descending order and select the top 30
top_shared_genes = gene_counts_df.sort_values(by="Count", ascending=False).head(30)

print(top_shared_genes)