import torch
import numpy as np
from utils.analysis_utils import get_id
from sklearn.neighbors import NearestNeighbors





def torch_knn(X, K):
    if not torch.is_tensor(X) or not X.is_cuda:
        X = torch.tensor(X, device='cuda', dtype=torch.float32)

    c_dist = torch.cdist(X, X, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    t_k = torch.topk(c_dist, K + 1, dim=0, largest=False, sorted=True)

    return t_k.values.cpu().numpy().T, t_k.indices.cpu().numpy().T


# code is based on Curvature-aware Manifold Learning by Yangyang Li https://arxiv.org/abs/1706.07167
# we stop at computing Bi matrices and generating curvature information
def caml(X, K, d=None, dist_matrix=None, XK=None, use_gpu=False, batch_size=4, verbose=False):

    '''
        Estimates the principal curvatures of a system of points X

        Args:
        X : 2-D Matrix X (n,D) where n is the number of points and D is the extrinsic dimension
        K : number of nearest neighbors to consider
        d : intrinsic dimension of the data
        dist_matrix : precomputed distance matrix
        XK : 2-D Matrix X (n x K x D) precomputed nearest neighbors
        use_gpu : use gpu for computation
        batch_size : batch size for estimation
        verbose : print information

        Returns:
        p_curvs : principal curvatures of the data
    '''
    N,D = X.shape
    # 1. find K nearest neighbors for xi, i=1, ..., N
    if dist_matrix is None:
        metric = 'minkowski'
        data = X
    else:
        metric = 'precomputed'
        data = dist_matrix
    if XK is None:
        if use_gpu:
            I = torch_knn(X, K)[1]
        else:
            nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric=metric)
            nbrs.fit(data)
            I = nbrs.kneighbors(data, return_distance=False)[1]

        I = I[:, 1:]
        XK = X[I]

    # 2. determine the intrinsic dimension of the data
    if d is None:
        d = get_id(X)
        d = int(round(d))
    if verbose:
        print('caml: estimated intrinsic dimension {}'.format(d))

# Get back to this. also check if d=D
    if d == D:
        return np.zeros((X.shape[0], X.shape[1]))

    # 3. compute Bi for xi, i=1, ..., N
    # 3.1 compute XK, XK is in N x K x D (samples x neighbors x feats)

    p_curvs = estimate_curvature(X, XK, d, batch_size, use_gpu)

    return p_curvs


def estimate_curvature(X, XK, d, batch_size,use_gpu):
    p_curvs = []

    f_estimate = caml_inner_batched
    if use_gpu:
        f_estimate = caml_inner_batched_gpu

    for i in range(int(np.ceil(X.shape[0] / batch_size))):
        p_curv_i = calc_batch(X, XK, batch_size, i, f_estimate, d, use_gpu=use_gpu)
        p_curvs.append(p_curv_i)

    if use_gpu:
        p_curvs = torch.cat(p_curvs).detach().cpu().numpy()
    else:
        p_curvs = np.concatenate(p_curvs, axis=0)
    return p_curvs


def calc_batch(X, XK, b_s, i, f, d, use_gpu):
    act_bs = min(b_s, X.shape[0])
    X_i = X[i * act_bs:(i + 1) * act_bs]
    XK_i = XK[i * act_bs:(i + 1) * act_bs]
    if use_gpu:
        if not torch.is_tensor(X_i):
            X_i = torch.tensor(X_i, device='cuda')
        elif not X_i.is_cuda:
            X_i = torch.tensor(X_i, device='cuda')

        if not torch.is_tensor(XK_i):
            XK_i = torch.tensor(XK_i, device='cuda')
        elif not XK_i.is_cuda:
            XK_i = torch.tensor(XK_i, device='cuda')
    return f(X_i, XK_i, d)



def caml_inner_batched(X, XK, d):
    tidx = np.triu_indices(d, k=0)
    ones_mult = np.ones((d, d))
    np.fill_diagonal(ones_mult, .5)

    X_b = np.transpose(XK - X[:, None, :], (0, 2, 1))
    DUi_b, _, _ = np.linalg.svd(X_b, full_matrices=False)

    Ui_b = np.transpose(DUi_b, (0, 2, 1)) @ X_b
    Ui_d_b = Ui_b[:, :d]
    Ui_d_tr_b = np.transpose(Ui_d_b, (0, 2, 1))
    fi_b = np.transpose(Ui_b[:, d:], (0, 2, 1))  # fi is in K x (D-d)

    UUi_b = np.einsum('bki,bkj->bkij', Ui_d_tr_b, Ui_d_tr_b)
    UUi_b = UUi_b * ones_mult
    UUi_b = np.transpose(UUi_b[:, :, tidx[0], tidx[1]], (0, 2, 1))

    psii_b = np.concatenate((
        Ui_d_b,
        UUi_b
    ), axis=1)

    psii_b = np.transpose(psii_b, (0, 2, 1))
    Bi_b = (np.linalg.pinv(psii_b) @ fi_b)
    Bi_b = np.transpose(Bi_b, (0, 2, 1))

    Hi_b = np.zeros((X.shape[0], Bi_b.shape[1], d, d))
    Hi_b[:, :, tidx[0], tidx[1]] = Bi_b[:, :, d:]
    Hi_b[:, :, tidx[1], tidx[0]] = Bi_b[:, :, d:]

    eig_values = torch.linalg.eigvalsh(Hi_b)
    R_b = eig_values.reshape((eig_values.shape[0], -1))
    return R_b


def caml_inner_batched_gpu(X_torch, XK_torch, d):
    tidx_torch = torch.triu_indices(d, d)
    ones_mult_torch = torch.ones((d, d), device='cuda')
    ones_mult_torch.fill_diagonal_(.5)

    X_b_torch = (XK_torch - X_torch[:, None, :]).transpose(2, 1)
    DUi_b_torch, _, _ = torch.linalg.svd(X_b_torch, full_matrices=False)

    Ui_b_torch = DUi_b_torch.transpose(2, 1) @ X_b_torch
    del X_b_torch, DUi_b_torch
    Ui_d_b_torch = Ui_b_torch[:, :d]
    Ui_d_tr_b_torch = Ui_d_b_torch.transpose(2, 1)
    fi_b_torch = Ui_b_torch[:, d:].transpose(2, 1)
    del Ui_b_torch

    UUi_b_torch = torch.einsum('bki,bkj->bkij', Ui_d_tr_b_torch, Ui_d_tr_b_torch)
    UUi_b_torch = UUi_b_torch * ones_mult_torch
    UUi_b_torch = UUi_b_torch[:, :, tidx_torch[0], tidx_torch[1]].transpose(2, 1)

    psii_b_torch = torch.cat((
        Ui_d_b_torch,
        UUi_b_torch
    ), dim=1).transpose(2, 1)
    del UUi_b_torch, Ui_d_b_torch

    Bi_b_torch = (torch.linalg.pinv(psii_b_torch) @ fi_b_torch).transpose(2, 1)
    del psii_b_torch
    Hi_b_torch = torch.zeros((X_torch.shape[0], Bi_b_torch.shape[1], d, d), dtype=Bi_b_torch.dtype, device='cuda')
    Hi_b_torch[:, :, tidx_torch[0], tidx_torch[1]] = Bi_b_torch[:, :, d:]
    Hi_b_torch[:, :, tidx_torch[1], tidx_torch[0]] = Bi_b_torch[:, :, d:]

    del Bi_b_torch

    eig_values = torch.linalg.eigvalsh(Hi_b_torch)
    R_b_torch = eig_values.reshape((eig_values.shape[0], -1))


    return R_b_torch