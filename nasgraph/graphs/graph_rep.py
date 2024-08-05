import torch

import numpy as np
import networkx as nx
import torch.nn as nn
import multiprocessing as mp

from nasgraph.utils.net_util import init_network
from collections import defaultdict


def score_edge(model, cur_in_shape, cur_out_shape, parent_out_shape, cshift):
    batch_size = parent_out_shape[0]

    inputx = np.zeros(shape=(batch_size,)+cur_in_shape)
    for i in range(batch_size):
        inputx[i, i+cshift, :, :] = 1

    x = torch.from_numpy(inputx).float()
    y = model(x)
    outputy = y.detach().numpy()

    scores, sum_score = defaultdict(float), 0
    for i in range(batch_size):
        for j in range(cur_out_shape[0]):
            score = np.sum(outputy[i, j])
            if score != 0:
                scores[(i, j)] = score
                sum_score += score

    for edge in scores:
        scores[edge] /= sum_score

    return scores


def get_channel_shift(lid, pid, funs, parents, shapes):
    if funs[lid] != 'concat':
        return 0
    
    tshift = 0
    for p in parents[lid]:
        if p == pid:
            return tshift
        # shapes[p]: [(Cin, Hin, Win), (Cout, Hout, Wout)]
        tshift += shapes[p][1][0]

    raise RuntimeError('Something wrong with "parents" variable')


def score_layer(lid, layer, shapes, parents, funs, init, preprocessw):
    init_network(layer, init, preprocessw)
    lscores = {}
    for pid in parents[lid]:
        cshift = get_channel_shift(lid, pid, funs, parents, shapes)
        if funs[lid] == 'add':
            assert shapes[lid][0] == shapes[pid][1], \
                f'cur_in_shape = {shapes[lid][0]} while parent_out_shape = {shapes[pid][1]} for layer {lid}'
        escores = score_edge(layer, shapes[lid][0], shapes[lid][1], shapes[pid][1], cshift)
        lscores[(pid, lid)] = escores

    return lscores


def build_graph(lscores, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    # escores could be empty dictionary
    for (pid, lid), escores in lscores.items():
        if len(escores) == 0:
            continue
        for (super_node1, super_node2), score in escores.items():
            G.add_edge(f'{pid}_{super_node1}', f'{lid}_{super_node2}', weight=score)

    return G


def build_graph_representation(
    layers, shapes, parents, funs, init, prepw,
    ncpus=1, directed=True, verbose=False
):

    if verbose:
        print('The in-shape and out-shape are:')
        print(shapes)
        print('Parents relations are:')
        print(parents)
        print('Layers are:')
        print(layers)

    lscores = {}
    if ncpus == 1:
        for lid, layer in layers.items():
            ret = score_layer(lid, layer, shapes, parents, funs, init, prepw)
            lscores.update(ret)

    else:
        raise NotImplementedError

    graph = build_graph(lscores, directed)    

    return graph


def score_101_edge(model, cur_in_shape, cur_out_shape, parent_out_shape, cshift, fun):
    if fun == 'add':
        # truncate channels in NAS-Bench-101
        if parent_out_shape[0] < cur_in_shape[0]:
            raise ValueError(
                f'input channel ({cur_in_shape[0]}) < output channel'
                f' ({parent_out_shape[0]}) for truncate')
        elif parent_out_shape[0] == cur_in_shape[0]:
            batch_size = parent_out_shape[0]
        else:
            assert parent_out_shape[0] - cur_in_shape[0] == 1
            batch_size = cur_in_shape[0]
    else:
        batch_size = parent_out_shape[0]

    inputx = np.zeros(shape=(batch_size,)+cur_in_shape)
    for i in range(batch_size):
        inputx[i, i+cshift, :, :] = 1

    x = torch.from_numpy(inputx).float()
    y = model(x)
    outputy = y.detach().numpy()

    scores, sum_score = defaultdict(float), 0
    for i in range(batch_size):
        for j in range(cur_out_shape[0]):
            score = np.sum(outputy[i, j])
            if score != 0:
                scores[(i, j)] = score
                sum_score += score

    for edge in scores:
        scores[edge] /= sum_score

    return scores


def score_101_layer(lid, layer, shapes, parents, funs, init, preprocessw):
    init_network(layer, init, preprocessw)
    lscores = {}
    for pid in parents[lid]:
        cshift = get_channel_shift(lid, pid, funs, parents, shapes)
        # in 101 search space, there is truncation of channels, so parent_out_shape might not
        # be equal to cur_in_shape
        escores = score_101_edge(layer, shapes[lid][0], shapes[lid][1], shapes[pid][1], cshift, funs[lid])
        lscores[(pid, lid)] = escores

    return lscores


def build_101_graph_representation(
    layers, shapes, parents, funs, init, prepw,
    ncpus=1, directed=True, verbose=False 
):

    if verbose:
        print('The in-shape and out-shape are:')
        print(shapes)
        print('Parents relations are:')
        print(parents)
        print('Layers are:')
        print(layers)

    lscores = {}
    if ncpus == 1:
        for lid, layer in layers.items():
            ret = score_101_layer(lid, layer, shapes, parents, funs, init, prepw)
            lscores.update(ret)

    else:
        raise NotImplementedError

    graph = build_graph(lscores, directed)    

    return graph


def score_201_edge(model, cur_in_shape, cur_out_shape, parent_out_shape, cshift):
    def check_conv(model):
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                return True
        return False

    batch_size = parent_out_shape[0]

    inputx = np.zeros(shape=(batch_size,)+cur_in_shape)
    for i in range(batch_size):
        inputx[i, i+cshift, :, :] = 1

    x = torch.from_numpy(inputx).float()
    y = model(x)
    if check_conv(model):
        relu = nn.ReLU()
        y = relu(y)

    outputy = y.detach().numpy()

    scores, sum_score = defaultdict(float), 0
    for i in range(batch_size):
        for j in range(cur_out_shape[0]):
            score = np.sum(outputy[i, j])
            if score != 0:
                scores[(i, j)] = score
                sum_score += score

    for edge in scores:
        scores[edge] /= sum_score

    return scores


def score_201_layer(lid, layer, shapes, parents, init, preprocessw):
    init_network(layer, init, preprocessw)
    lscores = {}

    for pid in parents[lid]:
        escores = score_201_edge(layer, shapes[lid][0], shapes[lid][1], shapes[pid][1], 0)
        lscores[(pid, lid)] = escores

    return lscores


def build_201_graph_representation(
        layers, shapes, parents, init, preprocessw, ncpus=1,
        directed=False, verbose=False
    ):

    if verbose:
        print('The in-shape and out-shape are:')
        print(shapes)
        print('Parents relations are:')
        print(parents)
        print('Layers are:')
        print(layers)

    lscores = {}
    if ncpus == 1:
        for lid, layer in layers.items():
            ret = score_201_layer(lid, layer, shapes, parents, init, preprocessw)
            lscores.update(ret)
    else:
        num_cpus = min(ncpus, mp.cpu_count())
        pool = mp.Pool(processes=num_cpus)
        margs = [(lid, layer, shapes, parents, init, preprocessw) for lid, layer in layers.items()]
        tasks = [pool.apply_async(score_201_layer, args=marg) for marg in margs]
        result = [t.get() for t in tasks]
        for ls in result:
            lscores.update(ls)
        pool.close()

    graph = build_graph(lscores, directed)    

    return graph