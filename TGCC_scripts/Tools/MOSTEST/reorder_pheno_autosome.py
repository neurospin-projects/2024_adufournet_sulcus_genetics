# Antoine DUFOURNET

import sys
import os
import pandas as pd
from glob import glob

def check_and_reorder_pheno(pheno_file, fam_file):
    """
    Reorders the phenotype file accordingly the .fam file.
    Missing rows in the phenotype file are filled with 'NaN' values.

    Parameters:
    -----------
    pheno_file : str
        Path to the phenotype file with the IID column.
    fam_file : str
        Path to the .fam file to reorder against.

    Returns:
    --------
    None
    """
    # Step 1: Read the phenotype file
    pheno_df = pd.read_csv(pheno_file, sep='\t')
    if 'IID' not in pheno_df.columns:
        print(f"Error: 'IID' column not found in {pheno_file}")
        return
    iid_index = pheno_df.columns.get_loc('IID')
    pheno_iids = set(pheno_df['IID'].tolist())

    fam_df = pd.read_csv(fam_file, delim_whitespace=True, header=None)
    fam_df.columns = ['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype']
    fam_order = fam_df['IID'].tolist()

    # Step 2: Reorder the phenotype file according to the .fam file order
    reordered_pheno_df = pd.DataFrame(columns=pheno_df.columns)
    for iid in fam_order:
        if iid in pheno_iids:
            row = pheno_df[pheno_df['IID'] == iid]
        else:
            row = pd.DataFrame([['NaN'] * pheno_df.shape[1]], columns=pheno_df.columns)
            row.iloc[0, iid_index] = iid
        reordered_pheno_df = pd.concat([reordered_pheno_df, row], ignore_index=True)

    # Save the reordered phenotype file
    if "#FID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["#FID"], axis=1)
    if "FID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["FID"], axis=1)
    if "IID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["IID"], axis=1)

    output_pheno_file = pheno_file.replace('.txt', '_reordered.txt')
    reordered_pheno_df.to_csv(output_pheno_file, sep='\t', index=False)
    print(f"Reordered phenotype file saved as {output_pheno_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 check_and_reorder_pheno.py <pheno_file> <fam_file>")
        sys.exit(1)

    pheno_file = sys.argv[1]
    fam_file = sys.argv[2]

    check_and_reorder_pheno(pheno_file, fam_file)
