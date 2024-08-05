import os, sys

from .nasbench201_base_ops import *
from collections import defaultdict, deque


def get_genotype(arch_str):
    nodestrs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
        inputs = list(filter(lambda x: x != "", node_str.split("|")))
        for xinput in inputs:
            assert len(xinput.split("~")) == 2, \
                "invalid input length : {:}".format(xinput)
        inputs = (xi.split("~") for xi in inputs)
        input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
        genotypes.append(input_infos)
    return genotypes


def create_nasbench201(arch_str, img_size, num_stacks, num_modules_per_stack,
        stem_out_channels, use_bn=False):
    genotype = get_genotype(arch_str)
    nrc = NB201_Relation_Config(genotype, img_size=img_size, stem_out_channels=stem_out_channels,
        num_stacks=num_stacks, num_modules_per_stack=num_modules_per_stack, use_bn=use_bn)
    return nrc


class NB201_Relation_Config(object):
    def __init__(self, genotype, img_size=32, stem_out_channels=16,
        num_stacks=3, num_modules_per_stack=5, use_bn=False,
        affine=True, track_running_stats=True
    ):
        """
        Default input image has three channels
        """
        cur_size = img_size
        self._parents = defaultdict(list)
        self._oidx = {}
        self._shapes = {}
        self.layers = {}
        loc2glob = defaultdict(list)

        # initial stem
        lid = 0
        
        minput = nn.Identity()
        self.layers[lid] = minput
        self._shapes[lid] = [(3, cur_size, cur_size), (3, cur_size, cur_size)]
        lid += 1
        
        out_channels = stem_out_channels
        stem = Stem(in_channels=3, out_channels=out_channels, use_bn=use_bn)
        self.layers[lid] = stem
        self._shapes[lid] = [(3, cur_size, cur_size), (out_channels, cur_size, cur_size)]
        self._oidx[lid] = 0
        self._parents[lid].append(lid - 1)
        lid += 1

        loc2glob[(0, 0, 0)].append(lid - 1)

        # stack modules
        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                cell_inid = loc2glob[(stack_num, 0, 0)]
                conv_a = ConvBN(in_channels, 2*in_channels, 3, 2, 1, 1, affine, track_running_stats, use_bn=use_bn)
                self.layers[lid] = conv_a
                self._shapes[lid] = [(in_channels, cur_size, cur_size), (2*in_channels, cur_size//2, cur_size//2)]
                self._oidx[lid] = 1
                self._parents[lid] = cell_inid
                lid += 1

                conv_b = ConvBN(2*in_channels, 2*in_channels, 3, 1, 1, 1, affine, track_running_stats, use_bn=use_bn)
                self.layers[lid] = conv_b
                self._shapes[lid] = [(2*in_channels, cur_size//2, cur_size//2), (2*in_channels, cur_size//2, cur_size//2)]
                self._oidx[lid] = 1
                self._parents[lid].append(lid - 1)
                lid += 1

                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                    #     stride=1, padding=0, bias=False)
                    nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1,
                        stride=1, padding=0, bias=False)
                )
                self.layers[lid] = downsample
                self._shapes[lid] = [(in_channels, cur_size, cur_size), (2*in_channels, cur_size//2, cur_size//2)]
                self._oidx[lid] = 0
                self._parents[lid] = cell_inid
                lid += 1

                loc2glob[(stack_num, 0, 0)] = [lid-1, lid-2]

                out_channels *= 2
                in_channels = out_channels
                cur_size = cur_size // 2

            for module_num in range(num_modules_per_stack):
                for i in range(1, len(genotype)+1):
                    node_info = genotype[i - 1]
                    for (op_name, op_in) in node_info:
                        if op_in == 0:
                            op = OPS[op_name](
                                in_channels, out_channels, 1, affine, track_running_stats, use_bn=use_bn
                            )
                            shapes = [(in_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]
                        else:
                            op = OPS[op_name](
                                out_channels, out_channels, 1, affine, track_running_stats, use_bn=use_bn
                            )
                            shapes = [(out_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]

                        coord = self._get_coord(stack_num, module_num, num_modules_per_stack, i, len(genotype))
                        loc2glob[coord].append(lid)
                        
                        self.layers[lid] = op
                        self._shapes[lid] = shapes
                        self._oidx[lid] = 1 if 'conv' in op_name else 0
                        self._parents[lid] += loc2glob[(stack_num, module_num, op_in)]
                        lid += 1

        lastact = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.layers[lid] = lastact
        self._shapes[lid] = [(out_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]
        self._oidx[lid] = 1
        self._parents[lid] += loc2glob[(num_stacks, 0, 0)]
        lid += 1

        self.sanity_check(self._parents.copy())


    def _get_coord(self, stack_num, module_num, num_modules_per_stack, i, num_vertices):
        # i starts from 1
        if i < num_vertices:
            return (stack_num, module_num, i)
        # module_num starts from 0
        elif module_num+1 < num_modules_per_stack:
            return (stack_num, module_num+1, 0)
        return (stack_num+1, 0, 0)


    def sanity_check(self, parents):
        # node 0 is not included, so +1
        nnodes = len(parents) + 1
        nodequeue = deque()
        nodequeue.append(nnodes-1)
        visited = [nnodes-1]

        while len(nodequeue) != 0:
            u = nodequeue.popleft()
            for v in parents[u]:
                if v in visited:
                    continue
                nodequeue.append(v)
                visited.append(v)

        nodes_not_reachable = set(range(nnodes-1)) - set(visited)
        assert len(nodes_not_reachable) == 0, \
            f'Number of reacheable nodes is {len(visited)} while total number' +\
            f' of nodes is {nnodes}. Unreachable Nodes are {nodes_not_reachable}'

    @property
    def parents(self):
        return self._parents

    @property
    def shapes(self):
        return self._shapes
