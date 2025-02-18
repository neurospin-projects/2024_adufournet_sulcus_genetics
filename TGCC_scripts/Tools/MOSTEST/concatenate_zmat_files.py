# Antoine DUFOURNET

import os
import pandas as pd
import glob
import argparse

def concatenate_zmat_files(input_dir, output_dir, output_filename):
    # Find all the .zmat.tsv files in the specified directory
    file_pattern = os.path.join(input_dir, 'mostest_chr*_decim_maf-0.05_most_orig.zmat.tsv')
    file_list = glob.glob(file_pattern)
    
    # Initialize an empty list to hold the DataFrames
    dataframes = []
    
    # Variable to hold the column names and order
    columns = None
    
    # Process each file
    for file_path in file_list:
        # Extract the chromosome number from the filename
        filename = os.path.basename(file_path)
        chr_number = filename.split('_')[1].replace('chr', '')
        
        # Read the TSV file into a DataFrame
        df = pd.read_csv(file_path, sep='\t')
        
        # Add a 'CHR' column
        df['CHR'] = chr_number
        
        # If columns are not set, set them based on the first DataFrame
        if columns is None:
            columns = df.columns
        else:
            # Ensure the columns are the same and in the same order
            if not df.columns.equals(columns):
                raise ValueError(f"Columns do not match in file {file_path}")
        
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined DataFrame to a TSV file
    output_file_path = os.path.join(output_dir, output_filename)
    combined_df.to_csv(output_file_path, sep='\t', index=False)
    
    print(f"Combined file saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate zmat TSV files")
    parser.add_argument("--input_dir", required=True, help="Input directory containing the .zmat.tsv files")
    parser.add_argument("--output_dir", required=True, help="Output directory to save the combined TSV file")
    parser.add_argument("--output_filename", required=True, help="Output TSV file name")

    args = parser.parse_args()

    concatenate_zmat_files(args.input_dir, args.output_dir, args.output_filename)
