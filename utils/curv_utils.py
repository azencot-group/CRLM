import torch
import os
import numpy as np
from utils.analysis_utils import inspect_data, get_neighbours, get_id
from utils.data_loader import get_cifar
from CAML.caml import caml
from os.path import join


def save_curvature(args, model, activations):
    id_dir = join(args.root_path, 'ids')
    curv_dir = join(args.root_path, 'curvs')

    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    if not os.path.exists(curv_dir):
        os.makedirs(curv_dir)

    input_id_path = join(id_dir, f'input_id_{args.data_set}.npy')
    latent_id_path = join(id_dir, f'ids_{args.data_set}_{args.model_name}.npy')

    input_dim_path = join(id_dir, f'input_dim_{args.data_set}.npy')
    latent_dim_path = join(id_dir, f'dims_{args.data_set}_{args.model_name}.npy')

    input_curv_path = join(curv_dir, f'input_curv_{args.data_set}.npy')
    latent_curv_path = join(curv_dir, f'curvs_{args.data_set}_{args.model_name}.npy')

    checkpoints_path = join(args.checkpoints_dir, f'ckpt_{args.model_name}_{args.data_set}.pth')

    checkpoint = torch.load(checkpoints_path)
    model.load_state_dict(checkpoint['net'])

    estimate_input_id_curv(args, input_id_path, input_curv_path)

    ids = est_ids(args, model, activations, latent_id_path, latent_dim_path)
    estimate_curvature(args, model, ids, activations, latent_curv_path, num_samples=args.num_samples)


def est_ids(args, model, activations, latent_id_path, latent_dim_path):
    ids = []
    dims = []
    if not os.path.exists(latent_id_path):
        for ii in range(args.n_runs):
            activations.clear()
            loader, _ = get_cifar(args)
            acquire_latent_rep(args, model, loader, num_samples=args.num_samples)
            id_i, pca_id_i = inspect_data(activations, use_gpu=True)
            ids.append(id_i)
            dims.append(pca_id_i)
        mean_id = np.mean(ids, axis=0)
        mean_pca_id = np.mean(dims, axis=0)
        np.save(latent_id_path, mean_id)
        np.save(latent_dim_path, mean_pca_id)
    else:
        mean_id = np.load(latent_id_path)
        mean_pca_id = np.load(latent_dim_path)
    mean_id = np.round(mean_id).astype(int)
    return mean_id, mean_pca_id


def estimate_input_id_curv(args, input_id_path, input_curv_path):
    loader, _ = get_cifar(args)
    data = loader.dataset.data.astype(np.float32)
    data = data.reshape(data.shape[0], -1)
    n = data.shape[0]
    m = min(n, args.num_samples)
    if not os.path.exists(input_id_path):
        ids = []
        for ii in range(args.n_runs):
            data_i = data[np.random.choice(n, m, replace=False), :]
            id_est_i = get_id(data_i, use_gpu=True)
            ids.append(id_est_i)

        input_d = np.mean(ids, axis=0)
        np.save(input_id_path, input_d)

    if not os.path.exists(input_curv_path):
        input_d = np.load(input_id_path).item()
        input_d = int(round(input_d))
        data_i = data[np.random.choice(n, m, replace=False), :]
        curv = caml(X=data_i, K=2 * input_d, d=input_d, batch_size=args.caml_batch_size, use_gpu=True)

        np.save(input_curv_path, curv)
    return


def acquire_latent_rep(args, model, loader, num_samples=5000):
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, _) in enumerate(loader):
            if batch_idx * loader.batch_size < num_samples:
                batch_x = batch_x.to(args.device)
                model(batch_x)
    return


def estimate_curvature(args, model, ids, activations, curv_path, num_samples):
    loader, _ = get_cifar(args)
    model.eval()
    curvs = []

    data_activations = {}
    neighbours_activations = {}
    with torch.no_grad():
        for batch_idx, (batch_x, _) in enumerate(loader):
            activations.clear()
            neighbours_activations.clear()
            if batch_idx * args.batch_size < num_samples:
                print(f"Samples {batch_idx * args.batch_size}, total {num_samples}")
                batch_x = batch_x.to(args.device)
                # aqcuire latent representation
                model(batch_x)
                data_activations = activations.copy()

                # acquire neighbours
                batch_x_neighbours = get_neighbours(batch_x, num_neighbours=10, use_gpu=True)
                # for idx, batch_x_n_i in enumerate(batch_x_neighbours):
                #     print(idx)
                #     model(batch_x_n_i)
                batch_x_neighbours = batch_x_neighbours.reshape(
                    [batch_x_neighbours.shape[0] * batch_x_neighbours.shape[1]] + list(batch_x_neighbours.shape[2:]))
                # acquire latent representation of neighbours
                model(batch_x_neighbours)
                neighbours_activations = activations.copy()
                curvs_i = est_curv_batch(args, data_activations, neighbours_activations, ids)
                curvs.append(curvs_i)

        curvs = [np.concatenate([curvs[j][i] for j in range(len(curvs))], axis=0) for i in
                 range(len(curvs[0]))]
        mapc = [np.mean(np.abs(curvs[ii])) for ii in range(len(curvs))]
    np.save(curv_path, mapc)
    return curvs


def est_curv_batch(args, data_activations, neighbours_activations, ids):
    curvs = np.empty(len(ids), dtype=object)
    for idx, (layer_name, layer_data) in enumerate(data_activations.items()):
        d = ids[idx]
        layer_neighbours = neighbours_activations[layer_name]
        layer_neighbours = layer_neighbours.reshape((-1, args.batch_size, layer_neighbours.shape[-1]))
        XK = layer_neighbours.transpose(1, 0, 2)
        crv_k = caml(X=layer_data, K=2 * d, d=d, XK=XK, batch_size=args.caml_batch_size, use_gpu=True)
        if curvs[idx] is None:
            curvs[idx] = crv_k
        else:
            curvs[idx] = np.concatenate((curvs[idx], crv_k), axis=0)
    return curvs
