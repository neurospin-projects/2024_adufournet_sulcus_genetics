import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as stats
from scipy.linalg import svd
import h5py
import sys
import os

def correlation_heatmap(z_score, path_to_save):
    '''
    This function calculates and visualizes the correlation matrix of SNPs from a given region,
    creating a heatmap without using seaborn.
    
    Parameters:
    z_score (DataFrame): The input DataFrame containing SNP data. It should include 'CHR', 'SNP', 'PVAL', and additional dimensions (dim1, dim2, ...).

    Returns:
    correlation_matrix (DataFrame): The correlation matrix of SNPs based on their dimensions.
    
    Visualization:
    The function will plot a heatmap of the correlation matrix.
    '''

    # Drop non-numerical columns if necessary
    df_numerical = z_score.drop(['CHR', 'SNP', 'PVAL', 'N', 'FREQ'], axis=1) #['CHR', 'SNP', 'PVAL']
    df_transposed = df_numerical.T

    # Calculate correlation between rows
    correlation_matrix = df_transposed.corr()
    
    # Create labels for the axes
    labels = z_score['CHR'].astype(str) + " - " + z_score['SNP']
    correlation_matrix.columns = labels
    correlation_matrix.index = labels
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')

    # Add color bar with custom size and position
    cbar = fig.colorbar(cax, ax=ax, shrink=0.7, fraction=0.046, pad=0.04)

    # Set tick labels and adjust font size
    #ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Display every nth label to avoid clutter (adjust `step` based on the number of labels)
    step = max(1, len(labels) // 40)
    #ax.set_xticks(np.arange(0, len(labels), step))
    ax.set_yticks(np.arange(0, len(labels), step))

    # Update tick labels and alignment
    #ax.set_xticklabels(labels[::step], rotation=45, ha='center', fontsize=10)  # Center alignment
    ax.set_yticklabels(labels[::step], fontsize=10)

    # Adjust the spacing of the labels
    #ax.tick_params(axis='x', which='major', pad=10)  # Increase space between labels and axis

    # Title and axis labels
    ax.set_title("Correlation Matrix Between SNPs", pad=40, fontsize=14)
    ax.set_xlabel("SNP (CHR - SNP ID)", fontsize=12)
    ax.set_ylabel("SNP (CHR - SNP ID)", fontsize=12)

    # Adjust layout for better fit
    plt.tight_layout()
    clean_path = os.path.dirname(path_to_save)
    plt.savefig(f'{clean_path}/Correlation_Matrix_SNPs_MOSTest.eps', format='eps')
    #plt.show()
    
    return correlation_matrix

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('Usage: process_results_ext.py <bim> <fname> [<out>], where')
        print(' bim   - path to bim file (reference set of SNPs')
        print(' fname - prefix of .mat files output by mostest.m, ie. fname should be the same as "out" argument of the mostest.m')
        print(' out   - optional suffix for output files, by defautl fname will be used')
        sys.exit()

    bim_file = sys.argv[1] #'UKB26502_QCed_230519_maf0p005_chr21.bim'
    fname = sys.argv[2]    # 'all_chr21'
    out = sys.argv[3] if (len(sys.argv) > 3) else sys.argv[2]

    # read .bim file (reference set of SNPs)
    print('Load {}...'.format(bim_file))
    bim = pd.read_csv(bim_file, sep='\t', header=None, names='CHR SNP GP BP A1 A2'.split())
    del bim['GP']

    mat = sio.loadmat(fname + '.mat')
    bim['N'] = mat['nvec']

    # save matrix of z scores for SNPs passing 5e-08 threshold
    SNP_threshold = 5e-08
    print('Generate {}_***.zmat.tsv files...'.format(out))
    with h5py.File(fname + '_zmat.mat', 'r') as h5file:
        # Properly dereference the object references
        measures = []
        for i in range(h5file['measures'].shape[0]):
            ref = h5file['measures'][i, 0]
            measure = ''.join(chr(c[0]) for c in h5file[ref][:])
            measures.append(measure)
        #print(measures)
        freqvec = np.array(h5file['freqvec'])[0]
        maf_threshold = 0.005

        zmat_orig = np.array(h5file['zmat_orig']) #if you want to work with minp ad it to the list below
        for test in ['most_log10pval_orig']: #'minp_log10pval_orig', 
            pval = np.power(10, -mat[test].flatten())
            df_zmat_orig = pd.DataFrame(np.transpose(zmat_orig[:, pval < SNP_threshold]), columns=measures)
            df_zmat_orig.insert(0, 'FREQ', freqvec[pval < SNP_threshold])
            df_zmat_orig.insert(0, 'N', bim.N.values[pval < SNP_threshold])
            df_zmat_orig.insert(0, 'PVAL', pval[pval < SNP_threshold])
            df_zmat_orig.insert(0, 'SNP', bim.SNP.values[pval < SNP_threshold])
            df_zmat_orig.insert(0, 'CHR', bim.CHR.values[pval < SNP_threshold])
            #print(out + "_" + test.replace('_log10pval', '') + '.zmat.tsv')
            #print(df_zmat_orig.head())
            df_zmat_orig.to_csv(out + "_" + test.replace('_log10pval', '') + '.zmat.tsv', index=False, sep='\t')
            correlation_heatmap(df_zmat_orig, out)

	# save the z scores from the permuted genotype 
        zmat_perm = np.array(h5file['zmat_perm'])

        #for test in ['most_log10pval_perm']: #'minp_log10pval_orig',
            #df_zmat_perm = pd.DataFrame(np.transpose(zmat_perm[:,freqvec > maf_threshold]), columns=measures)
            #df_zmat_perm.insert(0, 'PVAL', pval[:])
            #df_zmat_perm.insert(0, 'SNP', bim.SNP.values[:])
            #df_zmat_perm.to_csv(out + "_" + test.replace('_log10pval', '') + '.zmat.csv', index=False, sep='\t')

        if False:
        # do the calculation of the correlation between the columns for the permuted genotpyes
            zmat_perm = np.transpose(zmat_perm)
            zmat_orig = np.transpose(zmat_orig)

            num_eigval_to_keep = 0
 
            ivec_snp_good = np.all(np.isfinite(zmat_orig) & np.isfinite(zmat_perm), axis=1)
            #print("ivec_snp_good:")
            #print(ivec_snp_good.shape)
            ivec_snp_good = ivec_snp_good & (freqvec > maf_threshold)
            #print("(freqvec > maf_threshold):")
            #print((freqvec > maf_threshold).shape)
            #print(f"ivec_snp_good: {len(ivec_snp_good)}")
            snps_weight_values = np.ones(zmat_perm.shape[0])

            def weightedcorrs(z_scores, weights):
               #Calculate weighted correlations. Replace this with the python implementation.

               # Placeholder: Return a correlation matrix using np.corrcoef for simplicity
               return np.corrcoef(z_scores, rowvar=False)

               # Compute correlation matrices
            C0 = weightedcorrs(zmat_perm[ivec_snp_good, :], snps_weight_values[ivec_snp_good])
            C1 = weightedcorrs(zmat_orig[ivec_snp_good, :], snps_weight_values[ivec_snp_good])
            print(C0)

             # Perform singular value decomposition (SVD)
            U, S, _ = svd(C0)
            s = np.diag(S)
            #print(U)
            print("Eigen values:")
            print(S)
             # Determine the maximum eigenvalue threshold for regularization
            if num_eigval_to_keep > 0:
                max_lambda = s[num_eigval_to_keep - 1]  # Python is zero-indexed
            else:
                max_lambda = 0
    
            #print("diag")
            #print(np.maximum(max_lambda, s))

             # Apply regularization
            C0_reg = U @ np.maximum(max_lambda, s) @ U.T
            #print(C0_reg)

             # Calculate the metric using the regularized matrix
            inv_C0_reg = np.linalg.inv(C0_reg)
            # To ge the final individual square Z score (with the correlation and the scale taken into account)
            # For each SNP
            most_coeff = (inv_C0_reg @ zmat_orig.T * zmat_orig.T)
            #print(most_coeff.shape)
            df_most_coeff = pd.DataFrame(np.transpose(most_coeff[:,pval < SNP_threshold ]), columns=measures)
            df_most_coeff.insert(0, 'PVAL', pval[pval < SNP_threshold])
            df_most_coeff.insert(0, 'SNP', bim.SNP.values[pval < SNP_threshold])
            df_most_coeff.to_csv(out + '_most.coeff.tsv', index=False, sep='\t')

#        # save individual GWAS results ('freqvec' is an indicator that we've saved individual GWAS beta's)
#        if 'freqvec' in h5file:
#            bim['FRQ'] = np.transpose(np.array(h5file['freqvec']))
#
#            beta_orig = np.array(h5file['beta_orig'])
#            se_orig = np.divide(beta_orig, zmat_orig)
#            pval_orig = stats.norm.sf(np.abs(zmat_orig)) * 2.0
#
#            for measure_index, measure in enumerate(measures):
#                fname = '{}.{}.orig.sumstats.gz'.format(out, measure)
#                print('Generate {}...'.format(fname))
#                bim['PVAL'] = np.transpose(pval_orig[measure_index, :])
#                bim['Z'] = np.transpose(zmat_orig[measure_index, :])
#                bim['BETA'] = np.transpose(beta_orig[measure_index, :])
#                bim['SE'] = np.transpose(se_orig[measure_index, :])
#                bim.to_csv(fname, compression='gzip', sep='\t', index=False)
#            del zmat_orig
#            del beta_orig
#            del se_orig
#            del pval_orig
#
#            beta_perm = np.array(h5file['beta_perm'])
#            zmat_perm = np.array(h5file['zmat_perm'])
#            se_perm = np.divide(beta_perm, zmat_perm)
#            pval_perm = stats.norm.sf(np.abs(zmat_perm)) * 2.0
#            for measure_index, measure in enumerate(measures):
#                fname = '{}.{}.perm.sumstats.gz'.format(out, measure)
#                print('Generate {}...'.format(fname))
#                bim['PVAL'] = np.transpose(pval_perm[measure_index, :])
#                bim['Z'] = np.transpose(zmat_perm[measure_index, :])
#                bim['BETA'] = np.transpose(beta_perm[measure_index, :])
#                bim['SE'] = np.transpose(se_perm[measure_index, :])
#                bim.to_csv(fname, compression='gzip', sep='\t', index=False)


