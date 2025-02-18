# Antoine DUFOURNET

import pandas as pd
import statsmodels.api as sm
import sys
import os
from scipy.stats import norm
import argparse


def process_files(pheno_file_path, covar_file_path, out_file_path):
    """
    Processes and pre-residualizes data from a pheno file and a covariate file.

    Parameters:
    -----------
    pheno_file_path : str
        Path to the pheno file (tab-separated) containing pheno 
        variables.
    covar_file_path : str
        Path to the covariate file (tab-separated) containing 
        covariates.

    Returns:
    --------
    None
        The function saves the processed data to a new file with '_pre_residualized' 
        suffix.
    """
    # Load data
    pheno_df = pd.read_csv(pheno_file_path, sep='\t')
    covar_df = pd.read_csv(covar_file_path, sep='\t')
    
    # Identify columns that are not of type int or float
    non_numeric_columns = covar_df.select_dtypes(exclude=['int', 'float']).columns

    # Convert boolean columns to 1 and 0
    for col in non_numeric_columns:
        if covar_df[col].dtype == bool:
            covar_df[col] = covar_df[col].astype(int)

    # Merge the dataframes on 'IID'
    if '#FID' in pheno_df.columns:
        merged_df = pd.merge(pheno_df.drop('#FID', axis=1), covar_df, on='IID', how='inner')
        list_cov = covar_df.drop(['IID', '#FID'], axis=1).columns
    elif 'FID' in pheno_df.columns:
        merged_df = pd.merge(pheno_df.drop('FID', axis=1), covar_df, on='IID', how='inner')
        list_cov = covar_df.drop(['IID', 'FID'], axis=1).columns

    print("\n","List of covariates:")
    print(list_cov,"\n")

    # Generate interaction terms
#    if 'I(Age*Age)' in list_cov and 'Age' in merged_df.columns:
#        merged_df['I(Age*Age)'] = merged_df['Age'] ** 2

    # Process each phenotype column
    phenotype_cols = [col for col in pheno_df.columns if col not in ['IID','#FID','FID']]

    print("\n","Pre-residualize with sm.OLS by keeping only the residuals for each dimension for each covariate.","\n")

    # Define the phenotype column
    print("Max correlation (between the phenotypes and the covariates) before pre-residualization:","\n")
    corr = merged_df.corr()
    print(abs(corr.loc[list_cov][phenotype_cols]).max(axis=1),"\n")

    # Pre-residualize and transform data
    for dim_i in phenotype_cols:
        X = merged_df[list_cov]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        y = merged_df[dim_i]
        model = sm.OLS(y, X, missing='drop').fit()

        # Get residuals
        residuals = model.resid
        merged_df[dim_i] = residuals

    print("Max correlation (between the phenotypes and the covariates) after pre-residualization:","\n")
    corr = merged_df.corr()
    print(abs(corr.loc[list_cov][phenotype_cols]).max(axis=1),"\n")

    print("Apply quantile normalization as advised.","\n")
    # Apply quantile normalization
    for dim_i in phenotype_cols:
        ecdf_values = merged_df[dim_i].rank(method='average') / len(merged_df[dim_i])
        qnorm_values = norm.ppf(ecdf_values - 0.5 / len(merged_df[dim_i]))
        merged_df[dim_i] = qnorm_values

    # Define the output file path if not provided in arg
    if not out_file_path:
        out_file_path = pheno_file_path.replace('.phe', '_pre_residualized.txt')

    print("\n",f"Save the pre-residualized file at {out_file_path} with a sep='\\t'.")
    # Save the pre_residualized file
    if '#FID' in pheno_df.columns:
        merged_df[['#FID','IID']+phenotype_cols].to_csv(out_file_path, sep='\t', index=False)
    elif 'FID' in pheno_df.columns:
        merged_df[['FID','IID']+phenotype_cols].to_csv(out_file_path, sep='\t', index=False)

    # Save the list of FID IID kept in the pre_residualized file as a .txt file
    dirname = os.path.dirname(pheno_file_path)
    list_name = os.path.join(dirname,'list_IID_kept.txt') 
    if '#FID' in pheno_df.columns:
        merged_df[['#FID','IID']].to_csv(list_name, index=False, header=False, sep=' ', quoting=3)
    elif 'FID' in pheno_df.columns:
        merged_df[['FID','IID']].to_csv(list_name, index=False, header=False, sep=' ', quoting=3)
    print("\n",f"Save the list of FID IID kept at {list_name}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Pre-residualize phenotype data using covariates and pheno variables.")
    parser.add_argument('pheno_file_path', type=str, help='Path to the pheno file (tab-separated) containing pheno variables.')
    parser.add_argument('covar_file_path', type=str, help='Path to the covariate file (tab-separated) containing covariates.')
    parser.add_argument('out_file_path', type=str, default=None, help='Optional: Path to the pre-residualized phenotype file (tab-separated).')

    args = parser.parse_args()

    process_files(args.pheno_file_path, args.covar_file_path, args.out_file_path)
