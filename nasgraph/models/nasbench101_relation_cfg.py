import torch
import copy

import numpy as np

from .nasbench101_base_ops import *
from collections import defaultdict, deque


class NB101_Relation_Config(object):
    def __init__(self, adjmat, ops, img_size=32, num_labels=10, in_channels=3,
        stem_out_channels=128, num_stacks=3, num_modules_per_stack=3, use_bn=True
    ):
        cur_size = img_size
        spec = ModelSpec(adjmat, ops)
        self.matrix = spec.matrix
        self._parents = defaultdict(list)
        self._funs = {}
        self._shapes = {}
        self.layers = {}

        # initial stem convolution
        lid = 0

        minput = nn.Identity()
        self.layers[lid] = minput
        self._shapes[lid] = [(in_channels, cur_size, cur_size), (in_channels, cur_size, cur_size)]
        self._funs[lid] = 'add'
        lid += 1
        
        out_channels = stem_out_channels
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1, use_bn=use_bn)
        self.layers[lid] = stem_conv
        self._shapes[lid] = [(in_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]
        self._parents[lid].append(lid-1)
        self._funs[lid] = 'add'
        lid += 1

        # stack modules
        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers[lid] = downsample
                self._shapes[lid] = [(in_channels, cur_size, cur_size), (in_channels, cur_size//2, cur_size//2)]
                self._parents[lid].append(lid-1)
                self._funs[lid] = 'add'
                lid += 1

                out_channels *= 2
                cur_size = cur_size // 2

            for module_num in range(num_modules_per_stack):
                cell_inpt_id = lid - 1
                num_vertices = np.shape(self.matrix)[0]
                vertex_channels = compute_vertex_channels(in_channels, out_channels, self.matrix)

                # operation for input on each vertex
                edge_map = {}
                for t in range(1, num_vertices):
                    if self.matrix[0, t]:
                        op = projection(in_channels, vertex_channels[t], use_bn=use_bn)
                        self.layers[lid] = op
                        self._shapes[lid] = [(in_channels, cur_size, cur_size), (vertex_channels[t], cur_size, cur_size)]
                        edge_map[(0, t)] = lid
                        self._parents[lid].append(cell_inpt_id)
                        self._funs[lid] = 'add'
                        lid += 1

                # operation for each node
                node_map = {}
                for t in range(1, num_vertices-1):
                    op = OP_MAP[spec.ops[t]](vertex_channels[t], vertex_channels[t], use_bn=use_bn)
                    self.layers[lid] = op
                    self._shapes[lid] = [(vertex_channels[t], cur_size, cur_size), (vertex_channels[t], cur_size, cur_size)]
                    node_map[t] = lid
                    p_tmp = [edge_map[(0, t)]] if self.matrix[0, t] else []
                    for s in range(1, t):
                        if self.matrix[s, t]:
                            p_tmp.append(node_map[s])
                    self._parents[lid] += p_tmp
                    self._funs[lid] = 'add'
                    lid += 1

                # manually add output nodew (identity mapping)
                # output node for concatenation
                self.layers[lid] = nn.Identity()
                self._shapes[lid] = [(out_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]
                p_tmp = []
                for s in range(1, num_vertices-1):
                    if self.matrix[s, num_vertices-1]:
                        p_tmp.append(node_map[s])
                self._parents[lid] += p_tmp
                self._funs[lid] = 'concat'
                lid += 1
                # output node for addition
                self.layers[lid] = nn.Identity()
                self._shapes[lid] = [(out_channels, cur_size, cur_size), (out_channels, cur_size, cur_size)]
                p_tmp = [lid-1] if not self.matrix[0, num_vertices-1] else [edge_map[0, num_vertices-1], lid-1]
                self._parents[lid] += p_tmp
                self._funs[lid] = 'add'
                lid += 1

                in_channels = out_channels

        # classifier = nn.Linear(out_channels, num_labels)
        # self.layers[lid] = classifier
        # self._shapes[lid] = [(out_channels,), (num_labels,)]
        # self._parents[lid].append(lid-1)
        # self._funs[lid] = 'add'
        # lid += 1
        self.sanity_check(self._parents.copy())

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

    @property
    def funs(self):
        return self._funs


def projection(in_channels, out_channels, use_bn=True):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1, use_bn=use_bn)


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


class ModelSpec(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format='channels_last'):
        """Initialize the module spec.
        Args:
          matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
          ops: V-length list of labels for the base ops used. The first and last
            elements are ignored because they are the input and output vertices
            which have no operations. The elements are retained to keep consistent
            indexing.
          data_format: channels_last or channels_first.
        Raises:
          ValueError: invalid matrix or ops
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()

        self.matrix = torch.tensor(self.matrix)

        self.data_format = data_format

    def _prune(self):
        """Prune the extraneous parts of the graph.
        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.
        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]


def compute_vertex_channels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Code from https://github.com/google-research/nasbench/
    Returns:
        list of channel counts, in order of the vertices.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
          vertex_channels[v] = interior_channels
          if correction:
              vertex_channels[v] += 1
              correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return [int(v) for v in vertex_channels]
