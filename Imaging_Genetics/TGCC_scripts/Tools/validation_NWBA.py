# Antoine DUFOURNET

import glob
import pandas as pd
import argparse

def find_files(base_path, file_pattern="most_orig.sumstats"):
    """
    Find all files matching the specified pattern under the given base path.

    Args:
        base_path (str): The base directory or file path to search.
        file_pattern (str): The file pattern to match. Default is 'most_orig.sumstats'.

    Returns:
        list: Sorted list of file paths matching the pattern.
    """
    search_pattern = f"{base_path.rstrip('/')}/**/*{file_pattern}"
    matched_files = glob.glob(search_pattern, recursive=True)
    return sorted(matched_files)

def extract_significant_snps(file_path, pval_threshold=5e-8):
    """
    Extract SNPs from a .sumstats file with PVAL < threshold.

    Args:
        file_path (str): Path to the .sumstats file.
        pval_threshold (float): P-value threshold to filter SNPs. Default is 5e-8.

    Returns:
        set: A set of SNP IDs that meet the threshold.
    """
    df = pd.read_csv(file_path, sep='\t')
    significant_snps = df[df["PVAL"] < pval_threshold]["SNP"].unique()
    return set(significant_snps)

def filter_sumstats_by_snps(file_path, snps):
    """
    Filter rows in a .sumstats file based on a set of SNP IDs.

    Args:
        file_path (str): Path to the .sumstats file.
        snps (set): Set of SNP IDs to filter.

    Returns:
        pd.DataFrame: A dataframe containing rows with matching SNPs.
    """
    df = pd.read_csv(file_path, sep='\t')
    filtered_df = df[df["SNP"].isin(snps)]
    return filtered_df

def process_results(base_path):
    """
    Process results to extract significant SNPs from white.British.ancestry and
    create a filtered validation.sumstats file from non.white.British.ancestry.

    Args:
        base_path (str): Path to the repository containing results.

    Returns:
        None
    """
    # Define paths for ancestry results
    white_british_path = f"{base_path}/white.British.ancestry"
    non_white_british_path = f"{base_path}/non.white.British.ancestry"

    # Find most_orig.sumstats files
    # If the lead SNPs are already extracted, use only the lead SNPs, 
    # And not all the significant SNPs

    white_british_files = find_files(white_british_path, "lead_SNP_heuristic.tsv")
    if not white_british_files:
        print("No lead SNPs file was found, the summary statistic is then used to get significant SNPs")
        white_british_files = find_files(white_british_path, "most_orig.sumstats")
    non_white_british_files = find_files(non_white_british_path, "most_orig.sumstats")

    if not white_british_files or not non_white_british_files:
        print("Error: Could not find the required files.")
        return

    white_british_sumstats = white_british_files[0]
    non_white_british_sumstats = non_white_british_files[0]

    # Extract significant SNPs from white.British.ancestry
    print("Extracting significant SNPs from:", white_british_sumstats)
    significant_snps = extract_significant_snps(white_british_sumstats)

    # Filter non.white.British.ancestry sumstats file based on SNP mask
    print("Filtering non.white.British.ancestry sumstats:", non_white_british_sumstats)
    validation_df = filter_sumstats_by_snps(non_white_british_sumstats, significant_snps)

    # Save the filtered dataframe as validation.sumstats
    output_file = f"{base_path}/validation.sumstats"
    validation_df.to_csv(output_file, sep="\t", index=False)
    print(f"Filtered validation.sumstats saved to: {output_file}")

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Process ancestry results.")
    parser.add_argument("repo_path", help="Path to the repository containing results regarding ancestry.")
    args = parser.parse_args()

    # Process results
    process_results(args.repo_path)
