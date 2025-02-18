# Antoine DUFOURNET

import sys
import glob
import pandas as pd
import argparse
from collections import defaultdict

def find_files(paths):
    """Find all files matching the pattern 'most_orig.sumstats' under the given base paths."""
    all_files = []
    for path in paths:
        if path.endswith('most_orig.sumstats'):
            # If the path directly points to a file
            matched_files = glob.glob(path)
        else:
            # If the path is a directory or contains wildcards, search for matching files
            search_pattern = f"{path.rstrip('/')}/**/*most_orig.sumstats"
            matched_files = glob.glob(search_pattern, recursive=True)
        
        all_files.extend(matched_files)
    
    return sorted(all_files)

def process_files(file_paths):
    """Process each file to compute the union and intersection of significant SNPs, grouped by chromosome."""
    
    union_dict = defaultdict(set)
    intersection_dict = None  # This will be initialized with the first file's SNPs per chromosome

    for i, file in enumerate(file_paths):
        print(f"Working with file: {file}")
        # Read the file into a DataFrame (overwriting `df` to save memory)
        df = pd.read_csv(file, sep='\t')
        
        # Filter significant SNPs (PVAL < 5e-8)
        df = df[df["PVAL"] < 5e-8]
        
        # Create a dictionary of sets, grouping SNPs by chromosome
        current_snps = defaultdict(set)
        for _, row in df.iterrows():
            current_snps[row['CHR']].add(row['SNP'])
        
        # Update the union dictionary
        for chr_num, snps in current_snps.items():
            union_dict[chr_num].update(snps)
        
        # Initialize the intersection dictionary with the first file's SNPs
        if intersection_dict is None:
            intersection_dict = current_snps
        else:
            # Update the intersection dictionary by retaining only common SNPs per chromosome
            for chr_num in list(intersection_dict.keys()):
                intersection_dict[chr_num].intersection_update(current_snps.get(chr_num, set()))
                # Remove chromosome if no SNPs remain in the intersection
                if not intersection_dict[chr_num]:
                    del intersection_dict[chr_num]
    
    return union_dict, intersection_dict

def main():
    # Argument parsing to receive multiple base paths and file paths
    parser = argparse.ArgumentParser(description="Find and process SNPs from summary statistics files.")
    parser.add_argument('paths', nargs='+', help="List of base paths or file paths to process.")
    args = parser.parse_args()

    # Find all the valid file paths ending with 'most_orig.sumstats'
    file_paths = find_files(args.paths)

    if not file_paths:
        print("No files found matching *most_orig.sumstats")
        sys.exit(1)
    
    print(f"Found {len(file_paths)} files:")
    for path in file_paths:
        print(f" - {path}")

    # Process the files to calculate union and intersection of SNPs
    union_dict, intersection_dict = process_files(file_paths)

    # Display the results
    print("\nResults:")

    # Union results
    print("\nUnion of SNPs per Chromosome:")
    for chr_num, snps in sorted(union_dict.items()):
        print(f"Chromosome {chr_num}: {len(snps)} SNPs")
    
    # Intersection results
    print("\nIntersection of SNPs per Chromosome:")
    for chr_num, snps in sorted(intersection_dict.items()):
        print(f"Chromosome {chr_num}: {len(snps)} SNPs")
   
    # Calculate and display the total number of SNPs for the union
    total_union_snps = sum(len(snps) for snps in union_dict.values())
    print(f"\nTotal SNPs in the Union: {total_union_snps}")

    # Calculate and display the total number of SNPs for the intersection
    total_intersection_snps = sum(len(snps) for snps in intersection_dict.values())
    print(f"Total SNPs in the Intersection: {total_intersection_snps}")

if __name__ == "__main__":
    main()

