
from collections import defaultdict


class LayerRecorder(object):
    """
    LayerRecorder records all preceeding layers connecting to the current layer
    """

    def __init__(self, input_op, input_shape):
        self.layers = {0: input_op}
        self.parents = defaultdict(list)
        self.loc2glob = defaultdict(list)
        self.shapes = {0: input_shape}
        self.funs = {0: 'add'}
        self.lid = 1

    def append_layer(self, lop, lshape, parents, fun='add'):
        """
        args:
            lop (module)
            lshape (list of tuple): [(Cin, Hin, Win), (Cout, Hout, Wout)]
            parents (list of int)
            fun (str): add or concat
        """
        assert isinstance(parents, list), \
            f'Expected "parents" parameter has a type list, but got {parents} (type {type(parents)})'
        self.layers[self.lid] = lop
        self.parents[self.lid] += parents
        self.shapes[self.lid] = lshape
        self.funs[self.lid] = fun
        self.lid += 1

    def record_loc_glob(self, loc, glob):
        self.loc2glob[(loc)] += glob

    def get_glob_lid(self, loc):
        return self.loc2glob[loc]

    def get_last_layer_id(self):
        return self.lid - 1
