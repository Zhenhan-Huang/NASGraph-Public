import hashlib
import itertools

import numpy as np
import networkx as nx

from collections import defaultdict


def save_graph(fpath, graph):
    # reading pickle using nx.read_gpickle(fnm)
    nx.write_gpickle(graph, fpath)


class AggregatorConv1D(object):
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def get_super_nodes_cluster(self, dim):
        size = 1 + (dim - self.kernel) // self.stride
        cluster = defaultdict(set)
        for b in range(size):
            offset = self.stride * b
            cluster[(b,)] = set([(i,) for i in range(offset, offset+self.kernel)])
        return cluster


class AggregatorConv2D(object):
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.strides = (stride, stride)

    def get_super_nodes_cluster(self, input_shape, kc=1, padding=0):
        '''
        Obtain the members' indices of the super nodes according to convolutional operation
        
        Parameters:
        input_shape:    shape of the inputs
        kc:             kernel depth has two options, (1) the same as the number of channels of the inputs
                        or (2) kc = 1, i.e., depthwise separable
        '''
        _, h1, w1 = input_shape
        kh, kw = self.kernel
        sh, sw = self.strides
        h2, w2 = 1 + (h1 + 2*padding - kh) // sh, 1 + (w1 + 2*padding - kw) // sw

        cluster = defaultdict(set)
        super_indices = np.indices(dimensions=[h2, w2]).reshape(2, -1).T

        for i2, j2 in super_indices:
            mask, pairs = self.get_mask_convolve_anchors_output(input_shape, i2, j2, padding)
            indices_x_in = pairs[mask]
            # key: index of one output neuron
            # val: indices of associated input neurons
            if kc == input_shape[0]:
                key = (i2, j2)
                for i1, j1 in indices_x_in:
                    cluster[key] |= set([(c, i1, j1) for c in range(input_shape[0])])
            elif kc == 1:
                for c in range(input_shape[0]):
                    key = (c, i2, j2)
                    cluster[(key)] |= set([(c, i1, j1) for i1, j1 in indices_x_in])
                    #print(f'key is {(i2, j2, c)}, value is:\n', sorted(cluster[key]))
        return cluster

    def get_mask_convolve_anchors_output(self, shape1, i2, j2, padding):
        '''
        Obtain mask of valid convolve anchors for given cell in the output patch.
        The positions of valid convolve anchors will be marked as True, otherwise False.
        Padded zeros are not considered as valid convolve anchors.
        Parameters:
        -----------
        shape1:          shape of the input patch
        i2, j2:          coordinate of the output cell
        paddings:        number of rows and columns to pad zeros (top, bottom, left, right)
        Returns:
        -------
        binary mask matrix of the valid anchors
        '''
        _, h1, w1 = shape1
        T, L = [padding] * 2

        sh, sw = self.strides

        # (0,0) - upper left corner
        indices = np.indices(dimensions=self.kernel).reshape(2, -1).T
        pairs = np.array([(-T + i + i2 * sh, -L + j + j2 * sw) for i, j in indices])
        col = pairs[:,0]
        cond1 = (col >= 0) & (col < h1)
        col = pairs[:,1]
        cond2 = (col >= 0) & (col < w1)
        mask = cond1 & cond2
        return mask, pairs


def collect_supernode(inout_shapes, agg_mode):
    clusters = {}
    for lid, shapes in inout_shapes.items():
        # shapes: [input_shape, output_shape]
        # convolution input shape: (C, H, W)
        # attention input shape: (sequence length, hidden size)

        if len(shapes[1]) == 1:
            out_shape = shapes[1]
            if agg_mode == 'depthwise':
                super_kernel = out_shape[0]
                super_stride = 1
            else:
                raise NotImplementedError
            aggr = AggregatorConv1D(kernel=super_kernel, stride=super_stride)
            cluster = aggr.get_super_nodes_cluster(dim=out_shape[0])
        
        elif len(shapes[1]) == 2:
            out_shape = shapes[1]
            if agg_mode == 'depthwise':
                super_kernel = out_shape[0]
                super_stride = 1
            else:
                raise NotImplementedError
            aggr = AggregatorConv1D(kernel=super_kernel, stride=super_stride)
            cluster = aggr.get_super_nodes_cluster(dim=out_shape[0])

        else:   
            out_shape = shapes[1]
            if agg_mode == 'depthwise':
                super_kernel = out_shape[1:]
                super_stride = 1
                kc = 1
            else:
                raise NotImplementedError
            aggr = AggregatorConv2D(kernel=super_kernel, stride=super_stride)
            cluster = aggr.get_super_nodes_cluster(input_shape=out_shape, kc=kc)
        clusters[lid] = cluster

    return clusters


def gen_is_edge_fn(bits):
    """Generate a boolean function for the edge connectivity.
    Given a bitstring FEDCBA and a 4x4 matrix, the generated matrix is
      [[0, A, B, D],
       [0, 0, C, E],
       [0, 0, 0, F],
       [0, 0, 0, 0]]
    Note that this function is agnostic to the actual matrix dimension due to
    order in which elements are filled out (column-major, starting from least
    significant bit). For example, the same FEDCBA bitstring (0-padded) on a 5x5
    matrix is
      [[0, A, B, D, 0],
       [0, 0, C, E, 0],
       [0, 0, 0, F, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]]
    Args:
      bits: integer which will be interpreted as a bit mask.
    Returns:
      vectorized function that returns True when an edge is present.
    """
    def is_edge(x, y):
        """Is there an edge from x to y (0-indexed)?"""
        if x >= y:
            return 0
        # Map x, y to index into bit string
        index = x + (y * (y - 1) // 2)
        return (bits >> index) % 2 == 1

    return np.vectorize(is_edge)


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).
    i.e. no disconnected or "hanging" vertices.
    It is sufficient to check for:
      1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
      2) no cols of 0 except for col 0 (only input vertex has no in-edges)
    Args:
      matrix: V x V upper-triangular adjacency matrix
    Returns:
      True if the there are no dangling vertices.
    """
    shape = np.shape(matrix)

    rows = matrix[:shape[0]-1, :] == 0
    rows = np.all(rows, axis=1)     # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)     # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.
    Args:
      matrix: np.ndarray square upper-triangular adjacency matrix.
      labeling: list of int labels of length equal to both dimensions of
        matrix.
    Returns:
      MD5 hash of the matrix and labeling.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w]
                             for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.
    Args:
      graph: np.ndarray adjacency matrix.
      label: list of labels of same length as graph dimensions.
      permutation: a permutation list of ints of same length as graph dimensions.
    Returns:
      np.ndarray where vertex permutation[v] is vertex v from the original graph
    """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    def edge_fn(x, y): return graph[inverse_perm[x], inverse_perm[y]] == 1
    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def is_isomorphic(graph1, graph2):
    """Exhaustively checks if 2 graphs are isomorphic."""
    matrix1, label1 = np.array(graph1[0]), graph1[1]
    matrix2, label2 = np.array(graph2[0]), graph2[1]
    assert np.shape(matrix1) == np.shape(matrix2)
    assert len(label1) == len(label2)

    vertices = np.shape(matrix1)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(0, vertices)):
        pmatrix1, plabel1 = permute_graph(matrix1, label1, perm)
        if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
            return True

    return False


def compute_score(graph, weighted=None, metric='average degree', verbose=False):    
    score = None
    if metric == 'average degree':
        degs = graph.degree(weight=weighted)
        score = np.mean([v for v in dict(degs).values()])
    elif metric == 'density':
        assert weighted is None, 'Density property calculation does not support graphs other than unweighted graph'
        score = nx.density(graph)
    else:
        raise ValueError(f'metric "{metric}" not supported')
    return score
