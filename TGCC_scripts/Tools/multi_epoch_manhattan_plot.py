# Antoine DUFOURNET

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

def main(base_path):
    # Split the path to handle 'epoch*' dynamically
    path_before_epoch, after_epoch = base_path.split('epoch*')

    # Get all matching file paths
    file_paths = glob.glob(base_path)
    file_paths = sorted(file_paths, key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))  

    # Choose a colormap and generate a color gradient
    colormap = plt.colormaps['Blues']  
    colors = colormap(np.linspace(0.3, 1.0, len(file_paths)))  # Gradient from light to dark

    plt.figure(figsize=(21, 10.5))

    # Loop through each file and plot data
    for i, file in enumerate(file_paths):
        print("Working with file", file)
        df = pd.read_csv(file, sep='\t')
        
        if i == 0:
            # Calculate cumulative position for x-axis alignment only once
            df.sort_values(by=['CHR', 'BP'], inplace=True)
            chrom_offsets = df.groupby('CHR')['BP'].max().cumsum().shift(fill_value=0)

        # Filter significant SNPs
        df = df[df["PVAL"] < 1e-5]
        df['neg_log_pval'] = -np.log10(df['PVAL'])
        df['x_val'] = df.apply(lambda row: row['BP'] + chrom_offsets.loc[row['CHR']], axis=1)

        # Plot the SNPs with progressively darker colors
        plt.scatter(df['x_val'], df['neg_log_pval'], 
                    color=colors[i], s=4,  # Use color from gradient
                    label=file.replace(path_before_epoch, '').replace(after_epoch, ''), alpha=0.7)

    # Add a horizontal significance threshold line
    plt.axhline(y=7.3, color='r', linestyle='--')

    # Customize plot labels and title
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(p-value)')
    plt.title('Manhattan Plot of Multiple Epochs')

    # Add chromosome labels at their midpoints
    chromosome_ticks = [chrom_offsets[chrom] + df[df['CHR'] == chrom]['BP'].max() / 2 for chrom in sorted(df['CHR'].unique())]
    chromosome_labels = [f"Chr {chrom}" for chrom in sorted(df['CHR'].unique())]
    plt.xticks(chromosome_ticks, chromosome_labels, rotation=45)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.08, 1))
    plt.tight_layout()
    plt.savefig(path_before_epoch + '/Multi_manhattan_plot.eps', format='eps')
    plt.show()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate a multi-epoch Manhattan plot.")
    parser.add_argument('base_path', type=str, help="Base path with 'epoch*' wildcard for locating summary statistic files.")
    args = parser.parse_args()

    # Call the main function with the provided base_path
    main(args.base_path)
