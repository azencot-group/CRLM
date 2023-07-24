import itertools
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.intrinsic_dimension import intrinsic_dimension_np, intrinsic_dimension_torch
import torch.nn as nn


def get_id(x, use_gpu=True, verbose=False):
    if use_gpu:
        id_est = intrinsic_dimension_torch(x, verbose)
    else:
        id_est = intrinsic_dimension_np(x, verbose)
    return id_est


def get_neighbours(x, num_neighbours, use_gpu):
    if use_gpu:
        id_est = get_neighbours_torch(x, num_neighbours)
    else:
        id_est = get_neighbours_np(x, num_neighbours)
    return id_est


def get_layered_data(model, dict_out, layer_str=None):
    idx = 1
    handles = []
    for module in model:
        name = module.__class__.__name__
        if layer_str is not None:
            if layer_str in name:
                handles.append(module.register_forward_hook(get_activation(f'{name}_{idx}', dict_out)))
                idx += 1
        else:
            handles.append(module.register_forward_hook(get_activation(f'{name}', dict_out)))
    return handles, dict_out


def get_activation(name, dict_out):
    def hook(model, input, output):
        o_shape = output.shape
        out = output.detach().cpu().numpy().reshape(o_shape[0], -1)
        if name not in dict_out:
            dict_out[name] = out
        else:
            dict_out[name] = np.concatenate((dict_out[name], out), 0)

    return hook


def inspect_data(layers_dict, use_gpu=True):
    ids = []
    dims = []
    for key, value in layers_dict.items():
        id_est = get_id(value, use_gpu=use_gpu)
        id_pca = get_pca_id(value)
        ids.append(id_est)
        dims.append(id_pca)
    return ids, dims

def get_pca_id(x, th=0.9):
    # id given by the pca : 90 % of variance
    pca = PCA()
    scaler = StandardScaler()
    Out = scaler.fit_transform(x)
    pca.fit(Out)
    cs = np.cumsum(pca.explained_variance_ratio_)
    return np.argwhere(cs > th)[0][0] + 1
def get_neighbours_np(X, num_neighbours):
    x_np = X.detach().cpu().numpy()
    u, s, vh = np.linalg.svd(x_np, full_matrices=False)
    tol = np.max(s) * max(x_np.shape) * np.finfo(s.dtype).eps
    fzi = np.count_nonzero(s > tol, axis=-1)
    fzi = np.min(fzi)
    fzi = max(fzi, 1)
    u = u[..., :fzi]
    s = s[..., :fzi]
    vh = vh[..., :fzi, :]

    reps = min(fzi, num_neighbours)
    combs = np.array(list(map(list, itertools.product([0, 1], repeat=reps)))[:-1])
    combs_rep = np.tile(combs[:, None, None, :], (1, x_np.shape[0], x_np.shape[1], 1))
    l = len(combs)

    base_lst = np.ones(fzi - reps)
    s_rep = np.tile(s, (l, 1, 1, 1))
    base_lst_rep = np.tile(base_lst, (s_rep.shape[0], s_rep.shape[1], s_rep.shape[2], 1))
    masks = np.concatenate((base_lst_rep, combs_rep), axis=-1)

    s_masked = s_rep * masks

    s_eye = np.eye(fzi)
    s_eye_rep = np.tile(s_eye[None, None, :, :], (x_np.shape[0], x_np.shape[1], 1, 1))
    diag_tile = s_masked[..., None] * s_eye_rep
    u_tiled = np.tile(u, (l, 1, 1, 1, 1))
    vh_tiled = np.tile(vh, (l, 1, 1, 1, 1))

    res = u_tiled @ diag_tile @ vh_tiled
    return res


def get_neighbours_torch(X, num_neighbours):
    u, s, vh = torch.linalg.svd(X, full_matrices=False)
    tol = torch.max(s) * max(X.shape) * torch.finfo(s.dtype).eps
    fzi = torch.count_nonzero(s > tol, dim=-1)
    fzi = torch.min(fzi)
    fzi = max(fzi, 1)
    u = u[:, :, :, :fzi]
    s = s[:, :, :fzi]
    vh = vh[:, :, :fzi]

    reps = min(fzi, num_neighbours)
    combs = torch.tensor(list(map(list, itertools.product([0, 1], repeat=reps)))[:-1], device='cuda')
    combs_rep = torch.tile(combs[:, None, None, :], (1, X.shape[0], X.shape[1], 1))
    l = len(combs)

    base_lst = torch.ones(fzi - reps, device='cuda')
    s_rep = torch.tile(s, (l, 1, 1, 1))
    base_lst_rep = torch.tile(base_lst, (s_rep.shape[0], s_rep.shape[1], s_rep.shape[2], 1))
    masks = torch.cat((base_lst_rep, combs_rep), dim=-1)

    s_masked = s_rep * masks

    s_eye = torch.eye(fzi, device='cuda')
    s_eye_rep = torch.tile(s_eye[None, None, :, :], (X.shape[0], X.shape[1], 1, 1))
    diag_tile = s_masked[..., None] * s_eye_rep
    u_tiled = torch.tile(u, (l, 1, 1, 1, 1))
    vh_tiled = torch.tile(vh, (l, 1, 1, 1, 1))

    res = u_tiled @ diag_tile @ vh_tiled
    return res