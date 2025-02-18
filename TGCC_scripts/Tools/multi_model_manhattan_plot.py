# Antoine DUFOURNET
#
# To plot a summary statistic without having to wait for FUMA.
#
#

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os

def find_files(paths, file_name='most_orig.sumstats'):
    """Find all files matching the pattern 'most_orig.sumstats' under the given base paths."""
    all_files = []
    for path in paths:
        if path.endswith(file_name):
            # If the path directly points to a file
            matched_files = glob.glob(path)
        else:
            # If the path is a directory or contains wildcards, search for matching files
            search_pattern = f"{path.rstrip('/')}/**/*{file_name}"
            matched_files = glob.glob(search_pattern, recursive=True)
        
        all_files.extend(matched_files)
    
    return sorted(all_files)

def plot_manhattan(file_paths):
    # List of distinct base colors for each model
    base_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    plt.figure(figsize=(21, 10.5))

    # Initialize chrom_offsets for alignment
    chrom_offsets = None

    # Function to modify color brightness
    def adjust_color_brightness(color, factor):
        """Adjusts brightness of a color by blending with white (factor > 1) or black (factor < 1)."""
        color = mcolors.to_rgb(color)  # Convert color to RGB
        return tuple(min(1, max(0, c * factor)) for c in color)

    # Loop through each file and plot data
    for i, file in enumerate(file_paths):
        print(f"Working with file: {file}")
        model = file.split('/')[-2]
        
        df = pd.read_csv(file, sep='\t')
	
        if file.endswith('glm.linear'):
            df = df.rename(columns={"#CHROM":"CHR","POS":"BP", "ID":"rsID", "REF":"REF", "ALT":"A2", "A1":"A1", "TEST":"TEST", "OBS_CT":"OR", "P":"PVAL"})
        
        if i == 0:
            # Calculate cumulative position for x-axis alignment only once
            df.sort_values(by=['CHR', 'BP'], inplace=True)
            chrom_offsets = df.groupby('CHR')['BP'].max().cumsum().shift(fill_value=0)

        # Filter significant SNPs
        df = df[df["PVAL"] < 1e-1]
        df['neg_log_pval'] = -np.log10(df['PVAL'])
        df['x_val'] = df.apply(lambda row: row['BP'] + chrom_offsets.loc[row['CHR']], axis=1)

        # Loop through chromosomes to alternate colors
        chromosomes = sorted(df['CHR'].unique())
        for chrom_idx, chrom in enumerate(chromosomes):
            chrom_data = df[df['CHR'] == chrom]

            # Adjust color brightness based on chromosome index
            base_color = base_colors[i % len(base_colors)]  # Base color based on the model
            brightness_factor = 1.5 if chrom_idx % 2 == 0 else 0.5  # Stronger light or dark shades
            adjusted_color = adjust_color_brightness(base_color, brightness_factor)

            # Plot data for the chromosome
            plt.scatter(chrom_data['x_val'], chrom_data['neg_log_pval'], 
                        color=adjusted_color, s=4,
                        label=model if chrom_idx == 0 else "")  # Add label only once per model

    # Add a horizontal significance threshold line
    plt.axhline(y=-np.log10(0.05/1000000), color='r', linestyle='--')
    plt.axhline(y=-np.log10(0.05/10000), color='g', linestyle='--')

    # Customize plot labels and title
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(p-value)')
    plt.title('Manhattan Plot of Multiple Models')

    # Add chromosome labels at their midpoints
    chromosome_ticks = [chrom_offsets[chrom] + df[df['CHR'] == chrom]['BP'].max() / 2 for chrom in sorted(df['CHR'].unique())]
    chromosome_labels = [f"Chr {chrom}" for chrom in sorted(df['CHR'].unique())]
    plt.xticks(chromosome_ticks, chromosome_labels, rotation=45)

    # Add legend
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the plot to the current directory
    output_path = 'Multi_manhattan_plot.eps'
    #plt.savefig(output_path, format='eps')
    #print(f"Plot saved to: {output_path}")
    plt.show()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate multi-model Manhattan plots for given summary statistic files.")
    parser.add_argument('-p', '--paths', nargs='+', type=str, help="List of base folder paths or file paths to the summary statistic files.")
    parser.add_argument('-f', '--filename', type=str, default='most_orig.sumstats',  help="End of the name of the summary statistic files.")
    args = parser.parse_args()

    # Find all relevant files
    file_name = args.filename
    file_paths = find_files(args.paths, file_name)
    
    if not file_paths:
        print(f"No files found matching the pattern '{file_name}'. Please check your paths.")
        return

    # Call the plotting function with the found file paths
    plot_manhattan(file_paths)

if __name__ == "__main__":
    main()
