"""
@author: alessio ansuini (alessioansuini@gmail.com)
"""

from math import floor
import numpy as np
import torch
from sklearn import linear_model
from scipy.spatial.distance import pdist, squareform


def intrinsic_dimension_np(X, ret_dist=False, verbose=False):
    p_dist = pdist(X, metric='euclidean')
    dist = squareform(p_dist)
    id_est = estimate_np(dist, verbose=verbose)
    if ret_dist:
        return id_est, dist
    else:
        return id_est


def intrinsic_dimension_torch(X, ret_dist=False, verbose=False):
    if not torch.is_tensor(X) or not X.is_cuda:
        X = torch.tensor(X, device='cuda')
    dist = torch.cdist(X, X, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    id_est = estimate_torch(dist, verbose=verbose)
    if ret_dist:
        return id_est, dist
    else:
        return id_est


def estimate_np(X, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X

        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x

        (*) See cited paper for description

        Usage:

        _,_,reg,r,pval = estimate(X,fraction=0.85)

        The technique is described in :

        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y

    '''

    # sort distance matrix
    Y = np.sort(X, axis=1, kind='quicksort')

    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print('Fraction good points: {}'.format(good.shape[0] / Y.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0] * fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind='quicksort')
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints, np.newaxis], y[0:npoints, np.newaxis])
    # r, pval = pearsonr(x[0:npoints], y[0:npoints])
    return regr.coef_[0][0]


def estimate_torch(X, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X

        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x

        (*) See cited paper for description

        Usage:

        _,_,reg,r,pval = estimate(X,fraction=0.85)

        The technique is described in :

        "Estimating the intrinsic dimension of datasets by a
        minimal neighborhood information"
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y

    '''  # sort distance matrix
    X_torch = X
    Y_torch, _ = torch.sort(X_torch, dim=1)

    # clean data

    k1_torch = Y_torch[:, 1]
    k2_torch = Y_torch[:, 2]

    zeros_torch = torch.where(k1_torch == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros_torch.shape[0]))
        print(zeros_torch)

    degeneracies_torch = torch.where(k1_torch == k2_torch)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies_torch.shape[0]))
        print(degeneracies_torch)

    Y_range = torch.arange(Y_torch.shape[0], device=X_torch.device)
    good_torch = Y_range[(Y_range[:, None] != zeros_torch).all(dim=1)]

    good_torch = good_torch[(good_torch[:, None] != degeneracies_torch).all(dim=1)]

    if verbose:
        print('Fraction good points: {}'.format(good_torch.shape[0] / Y_torch.shape[0]))

    k1_torch = k1_torch[good_torch]
    k2_torch = k2_torch[good_torch]

    # n.of points to consider for the linear regression
    npoints_torch = int(floor(good_torch.shape[0] * fraction))

    # define mu and Femp
    N_torch = good_torch.shape[0]
    mu_torch, _ = torch.sort(torch.divide(k2_torch, k1_torch), dim=0)
    Femp_torch = (torch.arange(1, N_torch + 1, dtype=torch.float32, device=X_torch.device)) / N_torch

    # take logs (leave out the last element because 1-Femp is zero there)
    x_torch = torch.log(mu_torch[:-2])
    y_torch = -torch.log(1 - Femp_torch[:-2])

    # regression

    coef = x_torch[0:npoints_torch, None].pinverse() @ y_torch[0:npoints_torch, None]
    return coef.item()
