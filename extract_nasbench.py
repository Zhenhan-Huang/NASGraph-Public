import os, sys
import argparse

import pandas as pd

from nasbench import api as nb101api
from nas_201_api import NASBench201API as nb201api
from nasgraph.models.tnbapi import TransNASBenchAPI as tnb101api


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract NASBench Information')
    parser.add_argument('--outdir', type=str, default='.',
        help='output directory')
    parser.add_argument('--sspace', type=str, default='nasbench101',
        help='Search space')
    parser.add_argument('--naspath', type=str, default=None,
        help='')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def extract_nasbench101(args):
    columns = [
        'file id', 'module_hash', 'trainable params',
        'train accuracy', 'validation accuracy', 'test accuracy'
    ]
    result = []
    nas101 = nb101api.NASBench(args.naspath)
    for idx, model_hash in enumerate(nas101.hash_iterator()):
        # {'module_adjacency': array([[0, 1, 0, 0, 1, 1, 0],
        # [0, 0, 1, 0, 0, 0, 0],
        # [0, 0, 0, 1, 0, 0, 1],
        # [0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 0, 0, 1],
        # [0, 0, 0, 0, 0, 0, 0]], dtype=int8),
        # 'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'],
        # 'trainable_parameters': 8555530}
        # {108: [{'halfway_training_time': 883.4580078125,
        #         'halfway_train_accuracy': 0.8282251358032227,
        #         'halfway_validation_accuracy': 0.7776442170143127,
        #         'halfway_test_accuracy': 0.7740384340286255,
        #         'final_training_time': 1769.1279296875,
        #         'final_train_accuracy': 1.0,
        #         'final_validation_accuracy': 0.9241786599159241,
        #         'final_test_accuracy': 0.9211738705635071},
        #        {'halfway_training_time': 883.6810302734375,
        #         'halfway_train_accuracy': 0.8796073794364929,
        #         'halfway_validation_accuracy': 0.8291265964508057,
        #         'halfway_test_accuracy': 0.8204126358032227,
        #         'final_training_time': 1768.2509765625,
        #         'final_train_accuracy': 1.0,
        #         'final_validation_accuracy': 0.9245793223381042,
        #         'final_test_accuracy': 0.9190705418586731},
        #        {'halfway_training_time': 883.4569702148438,
        #         'halfway_train_accuracy': 0.8634815812110901,
        #         'halfway_validation_accuracy': 0.811598539352417,
        #         'halfway_test_accuracy': 0.8058894276618958,
        #         'final_training_time': 1768.9759521484375,
        #         'final_train_accuracy': 1.0,
        #         'final_validation_accuracy': 0.9304887652397156,
        #         'final_test_accuracy': 0.9215745329856873}]}
        fstat, cstat = nas101.get_metrics_from_hash(model_hash)
        train_accs, valid_accs, test_accs = [], [], []
        for dictobj in cstat[108]:
            train_accs.append(dictobj['final_train_accuracy'])
            valid_accs.append(dictobj['final_validation_accuracy'])
            test_accs.append(dictobj['final_test_accuracy'])

        data = [
            idx, model_hash, fstat['trainable_parameters'],
            sum(train_accs)/len(train_accs), sum(valid_accs)/len(valid_accs),
            sum(test_accs)/len(test_accs)
        ]

        df = pd.DataFrame([data], columns=columns)
        result.append(df)
    
    result = pd.concat(result, ignore_index=True)
    result.to_csv(f'{args.outdir}/nasbench_only108.csv', index=False)


def extract_nasbench201(args, dataset):
    columns = [
        'file id', 'arch str', 'cfg name', 'C', 'N',
        'train accuracy', 'validation accuracy', 'test accuracy'
    ]
    nas201 = nb201api(args.naspath, verbose=False)
    result = []
    for idx, arch_str in enumerate(nas201):
        # arch_str: |avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|
        # info:
        # {
        #   'train-loss': 0.31902812266031905,
        #   'train-accuracy': 88.91866665852865,
        #   'train-per-time': 7.221092998981476,
        #   'train-all-time': 1444.2185997962952',
        #   'valid-loss': 0.5658507757886251,
        #   'valid-accuracy': 81.98266665690103,
        #   'valid-per-time': 2.54897923696609,
        #   'valid-all-time': 509.795847393218,
        #   'test-loss': 0.5760996529261272,
        #   'test-accuracy': 81.52,
        #   'test-per-time': 1.0195916947864396,
        #   'test-all-time': 203.91833895728792
        # }
        info = nas201.get_more_info(
            idx, 'cifar10-valid' if dataset == 'cifar10' else dataset,
            iepoch=None, hp='200', is_random=False
        )
        cfg = nas201.get_net_config(idx, 'cifar10-valid' if dataset == 'cifar10' else dataset)

        data = [
            idx, arch_str, cfg['name'], cfg['C'], cfg['N'],
            info['train-accuracy'], info['valid-accuracy'], info['test-accuracy']
        ]

        df = pd.DataFrame([data], columns=columns)
        result.append(df)

    result = pd.concat(result, ignore_index=True)
    result.to_csv(f'{args.outdir}/nasbench201_{dataset}.csv', index=False)


def extract_transbench101(args):
    trans101 = tnb101api(args.naspath)
    for xtask in trans101.task_list:
        result = []
        for uid, xarch in enumerate(trans101.arch2space.keys()):
            arch_dict = {}
            for xmetric in trans101.metrics_dict[xtask]:
                arch_dict[xmetric] = trans101.get_single_metric(xarch, xtask, xmetric, mode='best')
            arch_dict['file_id'] = uid
            result.append(pd.DataFrame(arch_dict, index=[uid]))

        result = pd.concat(result, ignore_index=True)
        result.to_csv(f'{args.outdir}/transbench101_{xtask}.csv', index=False)


def main():
    args = parse_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    if args.sspace == 'nasbench101':
        extract_nasbench101(args)
    elif args.sspace == 'nasbench201':
        for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
            extract_nasbench201(args, dataset)
    elif args.sspace == 'transbench101':
        extract_transbench101(args)
    else:
        raise ValueError(f'NAS space {args.sspace} not supported')


if __name__ == '__main__':
    main()