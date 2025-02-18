# Antoine DUFOURNET

"""
Summary:
    prepare_for_fuma.py is a script designed to prepare the summary statistics data from a GWAS (Genome-Wide Association Study) for analysis with FUMA (Functional Mapping and Annotation). 
    The script combines summary statistics from multiple chromosomes. (It computes effect sizes and standard errors if the lines for it are uncomment, but FUMA does it as well, therefore it's not necessary). It saves the combined data to a compressed file.

Arguments:
    --pheno: Path to the pre-residualized phenotype file.
    --sumstats_folder: Path to the folder containing summary statistics files. Be careful, the folder must only contain files from a same study.
    --output_folder: Path to the folder where you want the output.

Example:
    python3 prepare_for_fuma.py --pheno /path/to/pheno_pre_residualized.txt --sumstats_folder /path/to/sumstats_folder --output_folder path/to/output_folder
"""


import os
import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for FUMA')
    parser.add_argument('--pheno', type=str, required=True, help='Path to pre-residualized phenotype file')
    parser.add_argument('--sumstats_folder', type=str, required=True, help='Path to the folder containing summary statistics')
    parser.add_argument('--output_folder', type=str, required=False, help='Path to the folder where you want the output')

    args = parser.parse_args()

    pheno_path = args.pheno
    sumstats_folder = args.sumstats_folder
    output_folder = args.output_folder
 1

    # Extract the {ref} prefix from the phenotype file name
    # ref_prefix = os.path.basename(pheno_path).split('_pheno_pre_residualized.txt')[0]
    ref_prefix = os.path.basename(pheno_path).split('_pheno.phe')[0]

    #sumstats_all_chr = []
    #print("Files:")
    #for file in os.listdir(sumstats_folder):
    #    if file.endswith('.most_perm.sumstats'): 
    #        sumstats_path = os.path.join(sumstats_folder, file)
    #        mostest_output = pd.read_csv(sumstats_path, sep='\t')
    #        print(f"{file} loaded")
    #        sumstats_all_chr.append(mostest_output)

    #sumstats_all_chr_df = pd.concat(sumstats_all_chr)
    #output_path = f"{output_folder}/{ref_prefix}_mostest_all_chr.most_perm.sumstats.gz" 
    #sumstats_all_chr_df.to_csv(output_path, sep='\t', index=False, compression='gzip')

    sumstats_all_chr = []
    print("Files:")
    for file in os.listdir(sumstats_folder):
        if file.endswith('.most_orig.sumstats'): 
            sumstats_path = os.path.join(sumstats_folder, file)
            mostest_output = pd.read_csv(sumstats_path, sep='\t')
            print(f"{file} loaded")

            sumstats_all_chr.append(mostest_output)

    sumstats_all_chr_df = pd.concat(sumstats_all_chr)
    output_path = f"{output_folder}/{ref_prefix}_mostest_all_chr.most_orig.sumstats.gz" 
    print(f"File saved at {output_folder}/{ref_prefix}_mostest_all_chr.most_orig.sumstats.gz")
    sumstats_all_chr_df.to_csv(output_path, sep='\t', index=False, compression='gzip')

    sumstats_all_chr = []
    print("Files:")
    for file in os.listdir(sumstats_folder):
        if file.endswith('.minp_orig.sumstats'): 
            sumstats_path = os.path.join(sumstats_folder, file)
            mostest_output = pd.read_csv(sumstats_path, sep='\t')
            print(f"{file} loaded")

            sumstats_all_chr.append(mostest_output)

    sumstats_all_chr_df = pd.concat(sumstats_all_chr)
    output_path = f"{output_folder}/{ref_prefix}_all_chr.minp_orig.sumstats.gz" 
    sumstats_all_chr_df.to_csv(output_path, sep='\t', index=False, compression='gzip')

