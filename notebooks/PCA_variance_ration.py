import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from sklearn.decomposition import PCA

base_path = '/home/ad279118/ukb/data/Champollion_V1_32/*/*/NOPCA/full_embeddings.csv'
output_base = '/neurospin/dico/adufournet/2026_PCA_on_latent/ChampollionV1_32'

verbose = False
standard_scaler = False

# Use glob to find all matching files
file_paths = glob.glob(base_path)

os.makedirs(output_base, exist_ok=True)

for file in file_paths:
    initial_path = file.replace('/full_embeddings.csv', '')
    
    print(f'Working with file: {file}')
    print("\n", file)
    
    # Extract region name from path
    # Path structure: .../Champollion_V1_32/<region>/<model>/NOPCA/full_embeddings.csv
    region_name = file.split('/')[-4]
    
    embeddings_UKB = pd.read_csv(file)

    if standard_scaler:
        list_columns = list(embeddings_UKB.columns)
        list_columns.remove('ID')
        embeddings_UKB[list_columns] = (
            (embeddings_UKB[list_columns] - embeddings_UKB[list_columns].mean(axis=0))
            / embeddings_UKB[list_columns].std(axis=0)
        )

    embeddings_UKB = embeddings_UKB.set_index('ID')
    
    # Fit PCA with 32 components
    pca = PCA(n_components=32)
    pca.fit(embeddings_UKB)

    # Save explained variance ratio table
    evr_df = pd.DataFrame({
        'pheno': [f'dim{i+1}' for i in range(32)],
        'explained_variance_ratio': pca.explained_variance_ratio_
    })
    
    output_path = os.path.join(output_base, f'{region_name}_explained_variance_ratio.csv')
    evr_df.to_csv(output_path, index=False)
    print(f'Saved explained variance ratio to: {output_path}')

    if verbose:
        print("Minimum std among the dimensions:")
        print(embeddings_UKB.std(axis=0).min(), "\n")
        print("Maximum std among the dimensions:")
        print(embeddings_UKB.std(axis=0).max(), "\n")
        print("Explained variance ratio for each PC:")
        print(pca.explained_variance_ratio_, "\n")
        print((np.cumsum(pca.explained_variance_ratio_) < 1.0).sum() + 1)