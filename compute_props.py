import os, sys
import argparse
import logging
import time
import csv

import multiprocessing as mp
import networkx as nx
import nasgraph.utils.logging as lu

from nasgraph.graphs.graph_props import GraphProps

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute graph properties')
    parser.add_argument('--wdir', type=str, default='.',
        help='')
    parser.add_argument('--outdir', type=str, default='.',
        help='')
    parser.add_argument('--fptrn', type=str, default=None,
        help='')
    parser.add_argument('--index-st', type=int, default=None,
        help='')
    parser.add_argument('--index-ed', type=int, default=None,
        help='')
    parser.add_argument('--index-multiple', type=int, default=1,
        help='index array [0*index-multiple, 1*index-multiple, ...]')
    parser.add_argument('--weight', type=str, default=None,
        help='')
    parser.add_argument('--seed', type=int, default=1,
        help='')
    parser.add_argument('--ncpus', type=int, default=1,
        help='')
    parser.add_argument('--directed', action='store_true', default=False,
        help='Convert neural network to directed graph (default undirected)')
    parser.add_argument('--virtualss', action='store_true', default=False,
        help='')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='')
    if len(sys.argv) == 1:
        print(parser.print_help())
        sys.exit(1)
    return parser.parse_args()


def cal_graph(args, fid):
    fnm = args.fptrn.replace('ID', f'{fid:08d}')
    path = f'{args.wdir}/{fnm}'
    if not os.path.isfile(path):
        logger.info(f'File "{fnm}" not found, skip this file')
        return None
    # read graph object
    G = nx.read_gpickle(path)
    if args.directed:
        assert nx.is_directed(G), f'Graph is undirected, but args.directed={args.directed}'
    gp = GraphProps(
        G,
        directed=args.directed,
        weight=args.weight,
        virtual_source_sink=args.virtualss,
        verbose=args.verbose
    )
    features = gp.get_all_features()
    features += [fid]
    return features   


def argument_check(args):
    assert os.path.isdir(args.wdir), f'Working directory "{args.wdir}" not found'
    assert args.fptrn is not None, f'File pattern (args.fptrn) not specified, got "{args.fptrn}"'
    assert 'ID' in args.fptrn, f'File pattern (args.fptrn) must contain "<ID>", got "{args.fptrn}"'


def main():
    args = parse_arguments()
    argument_check(args)

    os.makedirs(args.outdir, exist_ok=True)
    fnm = f'ms_indexst{args.index_st}_indexed{args.index_ed}_ncpus{args.ncpus}.log'
    lu.setup_logging(args.outdir, fnm, args.verbose)

    logger.info(f'Graph directed: "{args.directed}", weight: "{args.weight}"')
    logger.info(f'Arguments: {args}')

    time_st = time.time()
    ncpus = min(args.ncpus, mp.cpu_count())

    args_all = [(args, fid*args.index_multiple) for fid in range(args.index_st, args.index_ed)]

    if ncpus == 1:
        results = []
        for i, argument in enumerate(args_all):
            logger.info(f'Process {i}th graph')            
            ret = cal_graph(*argument)
            if ret is not None:
                results.append(ret)
    else:
        pool = mp.Pool(processes=ncpus)
        rets = [pool.apply_async(cal_graph, argument) for argument in args_all]
        results = [ret.get() for ret in rets if ret.get() is not None]
    
    logger.info(f'Number of graphs computed is {len(results)}')
    dummy_graph = nx.DiGraph() if args.directed else nx.Graph()
    dummy_gp = GraphProps(dummy_graph, directed=args.directed, dummy=True)
    feature_nms = dummy_gp.get_feature_names() + ['file id']
    sfnm = f'model_indexst{args.index_st}_indexed{args.index_ed}_directed{args.directed}.csv'
    with open(f'{args.outdir}/{sfnm}', 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter=',')
        writer.writerow(feature_nms)
        for result in results:
            writer.writerow(result)
    
    time_ed = time.time()
    logger.info(f'It takes {(time_ed - time_st)/60:.2f} mins')


if __name__ == '__main__':
    main()
