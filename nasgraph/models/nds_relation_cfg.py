import torch.nn as nn

from collections import defaultdict, deque
from .nds_base_ops import *
from nasgraph.utils.layer_recorder import LayerRecorder


class NDSCIFARNN(object):
    def __init__(self, C, nlayers, genotype, img_size=32, stem_multiplier=3, use_bn=False):
        #TODO: add support the jigsaw
        cur_size = img_size
        lshape = [(3, cur_size, cur_size), (3, cur_size, cur_size)]
        lrc = LayerRecorder(nn.Identity(), lshape)

        C_curr = stem_multiplier * C
        stem = NDS_CIFAR_HEAD(3, C_curr, use_bn=use_bn)
        lshape = [(3, img_size, img_size), (C_curr, img_size, img_size)]
        lrc.append_layer(stem, lshape, [lrc.get_last_layer_id()])

        cell_inputs = [[lrc.get_last_layer_id()], [lrc.get_last_layer_id()]]

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        for i in range(nlayers):
            if i in [nlayers//3, 2*nlayers//3]:
                # apply reduction
                C_curr *= 2
                reduction = True
                concat = genotype['normal_concat']
                op_names, indices = zip(*genotype['reduce'])
            else:
                # no reduction
                reduction = False
                concat = genotype['reduce_concat']
                op_names, indices = zip(*genotype['normal'])
                
            if reduction_prev:
                preprocess0 = FactorizedReduce(C_prev_prev, C_curr)
            else:
                preprocess0 = ConvBN(C_prev_prev, C_curr, 1, 1, 0)
            lshape = [(C_prev_prev, cur_size, cur_size), (C_curr, cur_size, cur_size)]
            lrc.append_layer(preprocess0, lshape, cell_inputs[0], fun='concat')
            lid_pre1 = lrc.get_last_layer_id()

            preprocess1 = ConvBN(C_prev, C_curr, 1, 1, 0)
            lshape = [(C_prev, cur_size, cur_size), (C_curr, cur_size, cur_size)]
            lrc.append_layer(preprocess1, lshape, cell_inputs[1], fun='concat')
            lid_pre2 = lrc.get_last_layer_id()

            nsteps = len(op_names) // 2
            loc2glob = {0: lid_pre1, 1: lid_pre2}
            for i in range(nsteps):
                stride1 = 2 if (reduction and indices[2*i] < 2) else 1
                stride2 = 2 if (reduction and indices[2*i+1] < 2) else 1
                if reduction:
                    if stride1 == 2:
                        lshape1 = [(C_curr, cur_size, cur_size), (C_curr, cur_size//2, cur_size//2)]
                    else:
                        lshape1 = [(C_curr, cur_size//2, cur_size//2), (C_curr, cur_size//2, cur_size//2)]
                    
                    if stride2 == 2:
                        lshape2 = [(C_curr, cur_size, cur_size), (C_curr, cur_size//2, cur_size//2)]
                    else:
                        lshape2 = [(C_curr, cur_size//2, cur_size//2), (C_curr, cur_size//2, cur_size//2)]
                    sum_shape = [(C_curr, cur_size//2, cur_size//2), (C_curr, cur_size//2, cur_size//2)]
                else:
                    lshape1 = lshape2 = [(C_curr, cur_size, cur_size), (C_curr, cur_size, cur_size)]
                    sum_shape = [(C_curr, cur_size, cur_size), (C_curr, cur_size, cur_size)]

                op1 = OPS[op_names[2*i]](C_curr, stride1, False, False)
                lrc.append_layer(op1, lshape1, [loc2glob[indices[2*i]]])
                lid1 = lrc.get_last_layer_id()
                

                op2 = OPS[op_names[2*i+1]](C_curr, stride2, False, False)
                lrc.append_layer(op2, lshape2, [loc2glob[indices[2*i+1]]])
                lid2 = lrc.get_last_layer_id()

                op_sum = nn.Identity()
                lrc.append_layer(op_sum, sum_shape, [lid1, lid2])
                lid_sum = lrc.get_last_layer_id()

                loc2glob[i+2] = lid_sum

            cur_size = cur_size // 2 if reduction else cur_size
            preprocess0_input = cell_inputs[1]
            preprocess1_input = [loc2glob[j] for j in concat]
            cell_inputs = [preprocess0_input, preprocess1_input]
            multiplier = len(concat)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        head = nn.Identity()
        lshape = [(C_prev, cur_size, cur_size), (C_prev, cur_size, cur_size)]
        lrc.append_layer(head, lshape, cell_inputs[1], fun='concat')

        self.layers = lrc.layers
        self._shapes = lrc.shapes
        self._parents = lrc.parents
        self.funs = lrc.funs
        # In case where all ops only connect s0 or s1, sanity_check fails
        #self.sanity_check(self._parents)

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


class NDSImageNetNN(object):
    def __init__(self):
        raise NotImplementedError
