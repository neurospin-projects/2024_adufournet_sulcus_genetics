import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats import gamma


def mahalanobis_norm_perm_function(X, Y, corr, nb_perm=5000):
    """
    Input:
        X: np.array of shape (sample_size, features_nb)
        Y: np.array of shape (sample_size, 1)
        nb_perm: int, default 5000
    
    Output:
        list_mahalanobis_norm_perm: list of float
    """

    Xdf = pd.DataFrame(X)
    Ydf = pd.DataFrame(Y)
    list_mahalanobis_norm_perm = []

    for i in tqdm(range(nb_perm)):
        perm = np.random.permutation(Ydf)
        perm = sm.add_constant(perm)
        list_z_score_perm = []

        for i in range(Xdf.shape[1]):
            model_perm = sm.OLS(Xdf.iloc[:,i], perm)
            results_perm = model_perm.fit()
            list_z_score_perm.append(results_perm.tvalues.iloc[1])

        z_score_perm = np.array(list_z_score_perm)
        mahalanobis_norm_perm = (z_score_perm @ np.linalg.inv(corr)) @ z_score_perm.T

        list_mahalanobis_norm_perm.append(mahalanobis_norm_perm)
    return list_mahalanobis_norm_perm


def UniVar_reg(X, Y, corr):
    """
    Input:
        X: np.array of shape (sample_size, features_nb)
        Y: np.array of shape (sample_size, 1)
    
    Output:
        list_betas_orig
        list_z_score_orig
    """
    Xdf = pd.DataFrame(X)
    Ydf = pd.DataFrame(Y)

    Ydf = sm.add_constant(Ydf)
    list_betas_orig = []
    list_z_score_orig = []

    for i in range(Xdf.shape[1]):
        model_orig = sm.OLS(Xdf.iloc[:,i], Ydf)
        results_orig = model_orig.fit()

        print("Column:", i)
        print("Betas:", format(results_orig.params.iloc[1], '.4f'))
        print("t-values:", format(results_orig.tvalues.iloc[1], '.4f'))
        print("p-values:", results_orig.pvalues.iloc[1], '\n')

        list_betas_orig.append(results_orig.params.iloc[1])
        list_z_score_orig.append(results_orig.tvalues.iloc[1])

    return list_betas_orig, list_z_score_orig

def MOSTest(list_z_score_orig, corr, list_mahalanobis_norm_perm):
    z_score_orig = np.array(list_z_score_orig )

    print(z_score_orig)
    mahalanobis_norm_orig = (z_score_orig @ np.linalg.inv(corr)) @ z_score_orig.T
    print(mahalanobis_norm_orig)

    print('MOSTEST p-value')
    fit_alpha, fit_loc, fit_scale = gamma.fit(list_mahalanobis_norm_perm)
    p_value_orig = 1-gamma.cdf(mahalanobis_norm_orig, a=fit_alpha, loc=fit_loc, scale=fit_scale)
    print(p_value_orig)

    return p_value_orig


a = pd.read_csv('/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/SOr-SOlf_right/19-00-10_160_0/hcp_random_epoch80_embeddings/full_embeddings.csv', index_col=0)
b = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/participants.csv', index_col=0)
c = pd.merge(a, b, left_on='ID', right_on='Subject', how='inner')

def mapper(row):
    if row['Age'] == '26-30':
        return 28
    if row['Age'] == '31-35':
        return 33
    if row['Age'] == '22-25':
        return 23.5
    if row['Age'] == '36+':
        return 38

def mapper_sex(row):
    if row['Gender'] == 'M':
        return 1
    if row['Gender'] == 'F':
        return 0

X = c.drop(b.columns, axis=1)
Y = c.drop(a.columns, axis=1)
#Y = Y.apply(mapper_age, axis=1)
Y = Y.apply(mapper_sex, axis=1)


Xdf = pd.DataFrame(X)
Ydf = pd.DataFrame(Y)
Xdf = sm.add_constant(Xdf)

model = sm.OLS(Y, Xdf.iloc[:,:])
results = model.fit()
print("Estimated betas:", '\n', results.params, '\n') # to get the betas
print("t_values:", '\n', results.tvalues, '\n') # which is in fact also the z-score
print("P-values:", '\n', results.pvalues, '\n')
print(results.f_pvalue)


corr = np.corrcoef(X, rowvar=False)
list_mahalanobis_norm_perm = mahalanobis_norm_perm_function(X=X, Y=Y, corr=corr, nb_perm=10000)
list_betas_orig, list_z_score_orig = UniVar_reg(X=X, Y=Y, corr=corr)
print(MOSTest(list_z_score_orig, corr, list_mahalanobis_norm_perm))
