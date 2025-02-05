import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA



#base_path = '/neurospin/dico/adufournet/mycode/Output/2025-01-22/09-35-58_201/UKB40_trained_on_WBA_embeddings/full_embeddings.csv'
base_path = '/neurospin/dico/adufournet/mycode/Output/ORBITAL_*/*/trained_on_UKB36WBA_random_embeddings/full_embeddings.csv'

#to know if the PCA must be done with a fit only on the WBA subjects, and a transform for both the WBA and the nWBA subjects.
WBA_stratification = True
path_to_WBAstrat = '/volatile/ad279118/Irene/Stratification/list_ID_WBA.csv'


verbose = False 
# Use glob to find all matching files
file_paths = glob.glob(base_path)

variance = 0.999

var = str(variance).split('.')[1]

for file in file_paths:
    initial_path = file.replace('/full_embeddings.csv', '')
    if not glob.glob(f'{initial_path}/42433_{var}varpc.csv'):
        print(f'Working with file: {file}')
        print("\n", file)
        embeddings_UKB = pd.read_csv(file)

        if WBA_stratification:
            list_WBA_ID = pd.read_csv(path_to_WBAstrat,names=['ID'],  header=None)
            WBA_only = pd.merge(embeddings_UKB, list_WBA_ID, on='ID')

            # PCA only on the WBA cohort
            WBA_only = WBA_only.set_index('ID')
            embeddings_UKB = embeddings_UKB.set_index('ID')
            # Calculate the number of components 
            n_components = len(WBA_only.columns)
            pca = PCA(n_components=n_components)
            pca.fit(WBA_only)
            nb_dim_to_keep = (np.cumsum(pca.explained_variance_ratio_) < variance).sum()+1
            pca = PCA(n_components=nb_dim_to_keep)
            pca.fit(WBA_only)

        else:
            # PCA on the entire dataset
            embeddings_UKB = embeddings_UKB.set_index('ID')
            n_components = len(embeddings_UKB.columns)
            pca = PCA(n_components=n_components)
            pca.fit(embeddings_UKB)
            nb_dim_to_keep = (np.cumsum(pca.explained_variance_ratio_) < variance).sum()+1
            pca = PCA(n_components=nb_dim_to_keep)
            pca.fit(embeddings_UKB)

        if verbose:
            # Print basic statistics
            #print("\n", embeddings_UKB.describe(), "\n")

            print("Minimum std among the dimensions:")
            print(embeddings_UKB.std(axis=0).min(), "\n")

            print("Maximum std among the dimensions:")
            print(embeddings_UKB.std(axis=0).max(), "\n")

            print("Explained variance ratio for each PC:")
            print(pca.explained_variance_ratio_, "\n")

            # Number of components to explain $variance of the variance
            print((np.cumsum(pca.explained_variance_ratio_) < variance).sum()+1)

        if False:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
            plt.grid(True)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA on the Combined Embeddings')
            plt.show()
        
        # PCA trnsform the whole dataset (check if fit on WBA only)
        pca_embeddings_UKB = pd.DataFrame(pca.transform(embeddings_UKB), columns=[f'dim{i}' for i in range(1,nb_dim_to_keep+1)],  index=embeddings_UKB.index)
        print(pca_embeddings_UKB.head())
        nb_subjects = len(pca_embeddings_UKB)
        path_to_save=f'{initial_path}/{nb_subjects}_{var}varpc.csv'
        #pca_embeddings_UKB.to_csv(path_to_save)
        print('File saved:', path_to_save)