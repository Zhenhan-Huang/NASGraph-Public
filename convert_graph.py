"""
Convert neural network in NAS space to graph
"""
import os, sys
import argparse
import time
import logging
import random
import torch
import json

import numpy as np
import pandas as pd
import nasgraph.utils.logging as lu
import multiprocessing as mp

from datetime import datetime
from nasbench import api as nb101api
from nas_201_api import NASBench201API as nb201api
from nasgraph.models.tnbapi import TransNASBenchAPI as tnb101api
from nasgraph.models.nasbench101_relation_cfg import NB101_Relation_Config
from nasgraph.models.nasbench201_relation_cfg import create_nasbench201
from nasgraph.models.transbench101_relation_cfg import TB101_Relation_Config
from nasgraph.models.nds_relation_cfg import (
    NDSCIFARNN
)
from nasgraph.graphs.graph_rep import (
    build_101_graph_representation, build_201_graph_representation,
    build_graph_representation
)
from nasgraph.utils.graph_util import save_graph

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert NAs to graphs')
    parser.add_argument('--wdir', type=str, default='.',
        help='working directory')
    parser.add_argument('--outdir', type=str, default='.',
        help='output directory')
    parser.add_argument('--agg-mode', type=str, default='depthwise',
        help='Aggregation mode')
    parser.add_argument('--sspace', type=str, default='nats_bench',
        help='Search space')
    parser.add_argument('--naspath', type=str, default=None,
        help='')
    parser.add_argument('--hashfile', type=str, default=None,
        help='')
    parser.add_argument('--index-st', type=int, default=None,
        help='')
    parser.add_argument('--index-ed', type=int, default=None,
        help='')
    parser.add_argument('--dataset', type=str, default='cifar10',
        help='')
    parser.add_argument('--index-multiple', type=int, default=1,
        help='index array [0*index-multiple, 1*index-multiple, ...]')
    parser.add_argument('--img-size', type=int, default=32,
        help='Image size')
    parser.add_argument('--prepw', type=str, default='none', choices=['none', 'minmax', 'absolute'],
        help='Preprocessing Conv2d weight')
    parser.add_argument('--nmodules', type=int, default=None,
        help='')
    parser.add_argument('--ncells', type=int, default=None,
        help='')
    parser.add_argument('--stemchannels', type=int, default=128,
        help='')
    parser.add_argument('--init', type=str, default='normal',
        help='')
    parser.add_argument('--seed', type=int, default=1,
        help='')
    parser.add_argument('--no-use-bn', action='store_true', default=False,
        help='')
    parser.add_argument('--directed', action='store_true', default=False,
        help='Convert neural network to directed graph (default undirected)')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='')
    parser.add_argument('--epsilon', type=float, default=1.0e-8)
    parser.add_argument('--ncpus', type=int, default=10,
        help='Number of CPUs')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def argument_check(args):
    assert os.path.isdir(args.wdir), \
        f'Working directory "{args.wdir}" not found'
    assert args.naspath is not None, \
        f'NASBench path is not specified, got "None"'
    assert os.path.isfile(args.naspath), \
        f'NASBench path ({args.naspath}) not found'
    assert args.index_multiple > 0, \
        f'Index multiple must be a positive integer'
    if args.sspace == 'nasbench101' or args.sspace == 'nasbench201':
        assert os.path.isfile(args.hashfile), \
            f'Hash file ({args.hashfile}) not found'


def convert_nasbench101(args, ind, adjmat, ops):
    fpath = f'{args.outdir}/model{ind:08d}_aggmode{args.agg_mode}_directed{args.directed}.gpickle'
    if os.path.isfile(fpath):
        print(f'File "{fpath}" already exists.')
        return

    nrc = NB101_Relation_Config(
        adjmat, ops, img_size=args.img_size, num_stacks=args.nmodules,
        num_modules_per_stack=args.ncells, stem_out_channels=args.stemchannels,
        use_bn=args.use_bn,
    )
    graph = build_101_graph_representation(
        nrc.layers, nrc.shapes, nrc.parents, nrc.funs, args.init, args.prepw,
        ncpus=args.ncpus, directed=args.directed, verbose=args.verbose,
    )

    save_graph(fpath, graph)


def convert_nasbench201(args, ind, arch_str):
    fpath = f'{args.outdir}/model{ind:08d}_aggmode{args.agg_mode}_directed{args.directed}.gpickle'
    if os.path.isfile(fpath):
        print(f'File "{fpath}" already exists.')
        return

    if args.verbose:
        print(f'The arch_str is {arch_str}')

    nrc = create_nasbench201(
        arch_str, img_size=args.img_size, num_stacks=args.nmodules, 
        num_modules_per_stack=args.ncells, stem_out_channels=args.stemchannels,
        use_bn=args.use_bn,
    )
    graph = build_201_graph_representation(
        nrc.layers, nrc.shapes, nrc.parents, args.init, args.prepw,
        ncpus=args.ncpus, directed=args.directed, verbose=args.verbose
    )
    save_graph(fpath, graph)


def convert_transbench101(args, ind, net_code, structure='backbone', task_name='none'):
    fpath = f'{args.outdir}/model{ind:08d}_task{task_name}_directed{args.directed}.gpickle'
    if os.path.isfile(fpath):
        print(f'File "{fpath}" already exists.')
        return

    if args.verbose:
        print(f'The net code is {net_code}')

    nrc = TB101_Relation_Config(
        net_code, structure, task_name, args.stemchannels,
        args.img_size, use_bn=args.use_bn
    )
    graph = build_201_graph_representation(
        nrc.layers, nrc.shapes, nrc.parents, args.init, args.prepw,
        ncpus=args.ncpus, directed=args.directed, verbose=args.verbose
    )
    save_graph(fpath, graph)


def convert_nds(args, ind, netinfo, network_relation_config):
    fpath = f'{args.outdir}/model{ind:08d}_aggmode{args.agg_mode}_directed{args.directed}.gpickle'
    if os.path.isfile(fpath):
        print(f'File "{fpath}" already exists.')
        return

    config = netinfo['net']
    ratio = 32 // args.stemchannels
    nrc = network_relation_config(config['width'] // ratio, config['depth'], config['genotype'])
    graph = build_graph_representation(
        nrc.layers, nrc.shapes, nrc.parents, nrc.funs, args.init, args.prepw,
        ncpus=args.ncpus, directed=args.directed, verbose=args.verbose
    )
    save_graph(fpath, graph)


def main():
    args = parse_arguments()
    argument_check(args)
    args.use_bn = not args.no_use_bn

    os.environ["KMP_WARNINGS"] = "FALSE"

    os.makedirs(args.outdir, exist_ok=True)
    fnm = f'ms_indexst{args.index_st}_indexed{args.index_ed}_ncpus{args.ncpus}.log'
    lu.setup_logging(args.outdir, fnm, args.verbose)

    logger.info(datetime.now().strftime('%Y%m%d_%H-%M-%S'))
    logger.info(f'Arguments: {args}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    inds = np.arange(args.index_st, args.index_ed) * args.index_multiple

    time_st = time.time()

    if args.sspace == 'nasbench101':
        # load hash file
        df = pd.read_csv(f'{args.hashfile}')
        # load NASBench101
        nb = nb101api.NASBench(args.naspath)
        params = []
        for ind in inds:
            if ind > 423623:
                break
            hash = df.iloc[ind]['module_hash']
            # fixed stat: 'module_adjacency', 'module_operations', 'trainable_parameters'
            # computed stat:
            # num : [{
            #  'halfway_training_time': <val>,
            #  'halfway_train_accuracy': <val>,
            #  'halfway_validation_accuracy': <val>,
            #  'halfway_test_accuracy': <val>,
            #  'final_training_time': <val>,
            #  'final_train_accuracy': <val>,
            #  'final_validation_accuracy': <val>,
            #  'final_test_accuracy': <val>}]
            f_stat, c_stat =  nb.get_metrics_from_hash(hash)
            ops = f_stat['module_operations']
            adjmat = f_stat['module_adjacency']
            if args.verbose:
                print('Operations: ', ops)
                print('Adjacency matrix: \n', adjmat)
            params.append((args, ind, adjmat, ops))

        if len(params) > 0:
            for param in params:
                convert_nasbench101(*param)

    elif args.sspace == 'nasbench201':
        num_archs = {
            'cifar10': 15625,
            'cifar100': 15625,
            'imagenet': 15625,
        }
        # load hash file
        df = pd.read_csv(f'{args.hashfile}')

        assert args.dataset.lower() in args.hashfile.lower(), \
            f'args.hashfile "{args.hashfile}" does not match dataset "{args.dataset}"'

        # load NASBench201
        nb = nb201api(args.naspath, verbose=False)
        params = []
        for ind in inds:
            if ind >= num_archs[args.dataset]:
                break
            arch_str = df.iloc[ind]['arch str']
            params.append([args, ind, arch_str])

        if len(params) > 0:
            for param in params:
                convert_nasbench201(*param)

    elif args.sspace == 'transbench101':
        num_archs = 7352
        nb = tnb101api(args.naspath)
        for ind in inds:
            if ind >= num_archs:
                break
            net_code = nb.index2arch(ind)
            # tasks 'class_object', 'segmentsemantic', 'autoencoder'
            for task in ['class_object', 'segmentsemantic', 'autoencoder']:
                convert_transbench101(args, ind, net_code, task_name=task)

    elif args.sspace == 'nds':
        # nds subspaces: nds_pnas, nds_enas, nds_darts, nds_darts_fix-w-d,
        #                nds_nasnet, nds_amoeba, nds_amoeba, nds_resnext-a, nds_resnext-a
        assert os.path.isfile(f'{args.hashfile}'), f'NDS json file "{args.hashfile}" not found'
        with open(args.hashfile, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data['top'] + data['mid']

        basenm = os.path.basename(args.hashfile)
        accuracies = {}

        for ind in inds:
            if ind > len(data) - 1:
                break
            netinfo = data[ind]
            config = netinfo['net']
            acc_min_test = 100 - netinfo['min_test_top1']
            acc_test = 100 - netinfo['test_ep_top1'][-1]
            accuracies[f'{ind}'] = [acc_min_test, acc_test]

            if 'genotype' in config:
                if '_in' in basenm:
                    raise NotImplementedError
                else:
                    network_relation_config = NDSCIFARNN
            else:
                raise NotImplementedError
            
            convert_nds(args, ind, netinfo, network_relation_config)

        with open(f'{args.outdir}/acc.json', 'w') as f:
            json.dump(accuracies, f)

    else:
        raise ValueError(f'Search space {args.sspace} not supported')

    time_ed = time.time()
    logger.info(f'It takes {(time_ed-time_st)/60:.2f} mins')
    print(f'It takes {(time_ed-time_st)/60:.2f} mins')


if __name__ == '__main__':
    main()
