# Antoine Dufournet
#
# Heuristic based on the correlation between the summary statistics to get 
# a first idea of which SNPs are among the lead SNPs.
# Seems to work pretty well, to find independant lead SNPs.

import argparse
import pandas as pd
import numpy as np
import glob
import os

def find_files(paths):
    """Find all files matching the pattern '.zmat.tsv' under the given base paths."""
    all_files = []
    for path in paths:
        if path.endswith('.zmat.tsv'):
            matched_files = glob.glob(path)
        else:
            search_pattern = f"{path.rstrip('/')}/**/*.zmat.tsv"
            matched_files = glob.glob(search_pattern, recursive=True)
        
        all_files.extend(matched_files)
    
    return sorted(all_files)

def get_lead_SNP(z_score_file, path_to_save, threshold=0.87):
    """Identify lead SNPs based on correlation and p-value."""
    
    # Load the z-score data
    z_score = pd.read_csv(z_score_file, sep='\t')

    # Drop non-numerical columns to calculate correlations
    numerical_data = z_score.drop(['CHR', 'SNP', 'PVAL', 'N', 'FREQ'], axis=1)
    correlations = np.abs(numerical_data.T.corr())  # Compute absolute correlation matrix
    
    # Identify SNP groups with correlations > threshold
    correlated_groups = []
    for idx, row in correlations.iterrows():
        correlated_snps = correlations.columns[row > threshold].tolist() 
        if len(correlated_snps) > 2:
            correlated_groups.append(set(correlated_snps))  

    # Deduplicate groups, to change because doesn't work
    unique_groups = []
    for group in correlated_groups:
        if not any(group.issubset(existing_group) for existing_group in unique_groups):
            unique_groups.append(group)

    # Debug: print unique groups for verification
    # print("Unique Groups:", unique_groups)

    # Find lead SNPs
    lead_snps = []
    for group in unique_groups:
        # Get rows corresponding to the group
        group_data = z_score.iloc[list(group)]
        
        # Skip empty groups
        if group_data.empty:
            continue

        # Find the SNP with the lowest p-value
        lead_snp = group_data.loc[group_data['PVAL'].idxmin()]
        lead_snps.append(lead_snp[['CHR', 'SNP', 'PVAL', 'N', 'FREQ']])

    # Save the lead SNPs to a file
    if lead_snps:
        lead_snps_df = pd.DataFrame(lead_snps)
        lead_snps_df = lead_snps_df.drop_duplicates()
        lead_snps_df.to_csv(path_to_save, sep='\t', index=False)
        print(f"Lead SNPs saved to {path_to_save}")
    else:
        print(f"No lead SNPs identified for file: {z_score_file}")

def main():
    parser = argparse.ArgumentParser(description="Identify lead SNPs based on correlation and p-value.")
    parser.add_argument('paths', nargs='+', type=str, help="List of base folder paths or file paths to the z-score files.")
    args = parser.parse_args()

    # Find all relevant files
    file_paths = find_files(args.paths)
    
    if not file_paths:
        print("No files found matching the pattern '.zmat.tsv'. Please check your paths.")
        return

    # Process each file
    for z_score_file in file_paths:
        base_path_to_save = os.path.dirname(z_score_file)
        path_to_save = os.path.join(base_path_to_save, "lead_SNP_heuristic.tsv")
        get_lead_SNP(z_score_file, path_to_save)

if __name__ == "__main__":
    main()

