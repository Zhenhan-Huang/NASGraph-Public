import math

import torch.nn as nn

from collections import deque
from .transbench101_base_ops import *
from nasgraph.utils.layer_recorder import LayerRecorder


class TB101_Relation_Config(object):
    def __init__(self, net_code, structure='backbone', task_name='none', 
        stem_channels=64, img_size=224, input_dim=(224, 224), chratio=1, use_bn=False    
    ):
        self.structure = structure
        self._read_net_code(net_code)
        chratio = self.base_channels // stem_channels
        self.base_channels = stem_channels
        self.inplanes = self.base_channels

        imratio = input_dim[0] // img_size
        self.cur_size = img_size
        lshape = [(3, self.cur_size, self.cur_size), (3, self.cur_size, self.cur_size)]
        lrc = LayerRecorder(nn.Identity(), lshape)

        stem1 = ConvBN(3, self.base_channels//2, 3, 2, 1, use_bn=use_bn)
        lshape = [(3, self.cur_size, self.cur_size), (self.base_channels//2, self.cur_size//2, self.cur_size//2)]
        lrc.append_layer(stem1, lshape, [lrc.get_last_layer_id()])

        stem2 = ConvBN(self.base_channels//2, self.base_channels, 3, 2, 1, use_bn=use_bn)
        lshape = [(self.base_channels//2, self.cur_size//2, self.cur_size//2), (self.base_channels, self.cur_size//4, self.cur_size//4)]
        lrc.append_layer(stem2, lshape, [lrc.get_last_layer_id()])

        self.cur_size = self.cur_size // 4

        llids = [lrc.get_last_layer_id()]
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)
            target_channel = self.inplanes * 2 if layer_type % 2 == 0 else self.inplanes
            stride = 2 if layer_type > 2 else 1
            llids = self.cell(lrc, llids, target_channel, stride, use_bn=use_bn)
            self.cur_size = self.cur_size // stride

        # virtual layers to combine layers
        # virlayer = nn.Identity()
        # lshape = [(self.inplanes, self.cur_size, self.cur_size), (self.inplanes, self.cur_size, self.cur_size)]
        # lrc.append_layer(virlayer, lshape, llids)

        if task_name == 'segmentsemantic':
            self.create_model_segmentsemantic(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'class_object':
            # encoder -> decoder, target dimension: 100
            self.create_model_class_object(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'class_scene':
            # encoder -> decoder, target dimension: 63
            self.create_model_class_scene(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'jigsaw':
            self.create_model_jigsaw(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'room_layout':
            # encoder -> decoder, target dimension: 9
            self.create_model_room_layout(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'autoencoder':
            self.create_model_autoencoder(lrc, llids, chratio, imratio, use_bn=use_bn)
        elif task_name == 'normal':
            self.create_model_normal(lrc, llids, chratio, imratio, use_bn=use_bn)
        else:
            raise ValueError(f'Task name "{task_name}" not supported')

        self.layers = lrc.layers
        self.shapes = lrc.shapes
        self.parents = lrc.parents
        self.sanity_check(self.parents)
        
        # print(net_code)
        # print(self.parents)
        # print(lrc.layers)
        # print(lrc.shapes)

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

    def convert_resnetbb(self, lrc, llids, planes, stride, affine=True, track_running_stats=True, use_bn=False):
        ltmp = []
        if stride == 2:
            img_size = self.cur_size // 2
        else:
            img_size = self.cur_size
        conv_a = ConvBNReLU(self.inplanes, planes, 3, stride, 1, 1, affine, track_running_stats, use_bn)
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (planes, img_size, img_size)]
        lrc.append_layer(conv_a, lshape, llids)

        conv_b = ConvBN(planes, planes, 3, 1, 1, 1, affine, track_running_stats, use_bn)
        lshape = [(planes, img_size, img_size), (planes, img_size, img_size)]
        lrc.append_layer(conv_b, lshape, [lrc.get_last_layer_id()])

        ltmp.append(lrc.get_last_layer_id())

        if stride != 1 or self.inplanes != planes:
            if use_bn:
                skipdownsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(planes, affine, track_running_stats)
                )
            else:
                skipdownsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
                )
        else:
            skipdownsample = nn.Identity()
        
        img_size = self.cur_size // 2 if stride !=1 else self.cur_size
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (planes, img_size, img_size)]
        lrc.append_layer(skipdownsample, lshape, llids)

        ltmp.append(lrc.get_last_layer_id())
        llids = ltmp
        self.inplanes = planes

        conv_a = ConvBNReLU(self.inplanes, planes, 3, 1, 1, 1, affine, track_running_stats, use_bn)
        lshape = [(self.inplanes, img_size, img_size), (planes, img_size, img_size)]
        lrc.append_layer(conv_a, lshape, ltmp)

        conv_b = ConvBN(planes, planes, 3, 1, 1, 1, affine, track_running_stats, use_bn)
        lshape = [(planes, img_size, img_size), (planes, img_size, img_size)]
        lrc.append_layer(conv_b, lshape, [lrc.get_last_layer_id()])

        ltmp_ret = [lrc.get_last_layer_id()]

        skipdownsample = nn.Identity()
        lshape = [(planes, img_size, img_size), (planes, img_size, img_size)]
        lrc.append_layer(skipdownsample, lshape, ltmp)

        ltmp_ret.append(lrc.get_last_layer_id())

        return ltmp_ret


    def convert_microcell(self, lrc, llids, planes, stride, affine=True,
        track_running_stats=True, use_bn=False):
        if stride == 2:
            img_size = self.cur_size // 2
        else:
            img_size = self.cur_size
        C_in = self.inplanes
        C_out = planes
        node_num = len(self.micro_code)
        from_nodes = [list(range(i)) for i in range(node_num)]
        node_dict = {0: llids}
        for node in range(1, node_num):
            node_dict[node] = []
            for op_idx, from_node in zip(self.micro_code[node], from_nodes[node]):
                if from_node == 0:
                    op = OPS[op_idx](C_in, C_out, stride, affine, track_running_stats, use_bn)
                    lshape = [(C_in, self.cur_size, self.cur_size), (C_out, img_size, img_size)]
                else:
                    op = OPS[op_idx](C_out, C_out, 1, affine, track_running_stats, use_bn)
                    lshape = [(C_out, img_size, img_size), (C_out, img_size, img_size)]
                
                lrc.append_layer(op, lshape, node_dict[from_node])
                
                node_dict[node].append(lrc.get_last_layer_id())

        self.inplanes = C_out

        return node_dict[node_num-1]

    def _read_net_code(self, net_code):
        net_code_list = net_code.split('-')
        self.base_channels = int(net_code_list[0])
        self.macro_code = net_code_list[1]
        if net_code_list[-1] == 'basic':
            self.cell_type = 'resnetbb'
            self.micro_code = 'basic'
            self.cell = self.convert_resnetbb
        else:
            self.cell_type = 'microcell'
            self.micro_code = [''] + net_code_list[2].split('_')
            self.cell = self.convert_microcell


    def create_model_segmentsemantic(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        out_width = out_width // imratio
        num_upsample = int(math.log2(out_width / self.cur_size))
        assert num_upsample in [2, 3, 4, 5, 6], f"invalid num_upsample: {num_upsample}"

        convlayer = ConvBNReLU(self.inplanes, 1024//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (1024//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(convlayer, lshape, llids)

        tmp_in_C = 1024//chratio
        for i in range(6-num_upsample):
            convlayer = ConvBNReLU(tmp_in_C, tmp_in_C//2, 3, 1, 1, use_bn=use_bn)
            lshape = [(tmp_in_C, self.cur_size, self.cur_size), (tmp_in_C//2, self.cur_size, self.cur_size)]
            lrc.append_layer(convlayer, lshape, [lrc.get_last_layer_id()])
            tmp_in_C = tmp_in_C // 2

        for i in range(num_upsample-1):
            deconvlayer = DeconvLayer(tmp_in_C, tmp_in_C//2, 3, 2, 1, use_bn=use_bn)
            lshape = [(tmp_in_C, self.cur_size, self.cur_size), (tmp_in_C//2, self.cur_size*2, self.cur_size*2)]
            lrc.append_layer(deconvlayer, lshape, [lrc.get_last_layer_id()])

            self.cur_size *= 2
            tmp_in_C = tmp_in_C // 2

        deconvlayer = DeconvLayer(tmp_in_C, tmp_in_C, 3, 2, 1)
        lshape = [(tmp_in_C, self.cur_size, self.cur_size), (tmp_in_C, self.cur_size*2, self.cur_size*2)]
        lrc.append_layer(deconvlayer, lshape, [lrc.get_last_layer_id()])

        self.cur_size *= 2

        convlayer = ConvBNReLU(tmp_in_C, out_width, 3, 1, 1, use_bn=use_bn)
        lshape = [(tmp_in_C, self.cur_size, self.cur_size), (out_width, self.cur_size, self.cur_size)]
        lrc.append_layer(convlayer, lshape, [lrc.get_last_layer_id()])

    def create_model_class_object(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        # decoder
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (self.inplanes, self.cur_size, self.cur_size)]
        lrc.append_layer(avgpool, lshape, llids)

    def create_model_class_scene(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        # decoder
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (self.inplanes, self.cur_size, self.cur_size)]
        lrc.append_layer(avgpool, lshape, llids)

    def create_model_jigsaw(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        raise NotImplementedError
    
    def create_model_room_layout(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        # decoder
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (self.inplanes, self.cur_size, self.cur_size)]
        lrc.append_layer(avgpool, lshape, llids)

    def create_model_autoencoder(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        out_width = out_width // imratio
        num_upsample = int(math.log2(out_width / self.cur_size))
        assert num_upsample in [2, 3, 4, 5, 6], f"invalid num_upsample: {num_upsample}"

        conv1 = ConvBNReLU(self.inplanes, 1024//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(self.inplanes, self.cur_size, self.cur_size), (1024//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv1, lshape, llids)

        conv2 = ConvBNReLU(1024//chratio, 1024//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(1024//chratio, self.cur_size, self.cur_size), (1024//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv2, lshape, [lrc.get_last_layer_id()])

        if num_upsample == 6:
            conv3 = DeconvLayer(1024//chratio, 512//chratio, 3, 2, 1, use_bn=use_bn)
            lshape = [(1024//chratio, self.cur_size, self.cur_size), (512//chratio, self.cur_size*2, self.cur_size*2)]
            lrc.append_layer(conv3, lshape, [lrc.get_last_layer_id()])

            self.cur_size = self.cur_size * 2
        else:
            conv3 = ConvBNReLU(1024//chratio, 512//chratio, 3, 1, 1, use_bn=use_bn)
            lshape = [(1024//chratio, self.cur_size, self.cur_size), (512//chratio, self.cur_size, self.cur_size)]
            lrc.append_layer(conv3, lshape, [lrc.get_last_layer_id()])

        conv4 = ConvBNReLU(512//chratio, 512//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(512//chratio, self.cur_size, self.cur_size), (512//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv4, lshape, [lrc.get_last_layer_id()])

        if num_upsample >= 5:
            conv5 = DeconvLayer(512//chratio, 256//chratio, 3, 2, 1, use_bn=use_bn)
            lshape = [(512//chratio, self.cur_size, self.cur_size), (256//chratio, self.cur_size*2, self.cur_size*2)]
            lrc.append_layer(conv5, lshape, [lrc.get_last_layer_id()])

            self.cur_size = self.cur_size * 2
        else:
            conv5 = ConvBNReLU(512//chratio, 256//chratio, 3, 1, 1, use_bn=use_bn)
            lshape = [(512//chratio, self.cur_size, self.cur_size), (256//chratio, self.cur_size, self.cur_size)]
            lrc.append_layer(conv5, lshape, [lrc.get_last_layer_id()])

        conv6 = ConvBNReLU(256//chratio, 128//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(256//chratio, self.cur_size, self.cur_size), (128//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv6, lshape, [lrc.get_last_layer_id()])
        
        if num_upsample >= 4:
            conv7 = DeconvLayer(128//chratio, 64//chratio, 3, 2, 1, use_bn=use_bn)
            lshape = [(128//chratio, self.cur_size, self.cur_size), (64//chratio, self.cur_size*2, self.cur_size*2)]
            lrc.append_layer(conv7, lshape, [lrc.get_last_layer_id()])

            self.cur_size = self.cur_size * 2
        else:
            conv7 = ConvBNReLU(128//chratio, 64//chratio, 3, 1, 1, use_bn=use_bn)
            lshape = [(128//chratio, self.cur_size, self.cur_size), (64//chratio, self.cur_size, self.cur_size)]
            lrc.append_layer(conv7, lshape, [lrc.get_last_layer_id()])

        conv8 = ConvBNReLU(64//chratio, 64//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(64//chratio, self.cur_size, self.cur_size), (64//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv8, lshape, [lrc.get_last_layer_id()])

        if num_upsample >= 3:
            conv9 = DeconvLayer(64//chratio, 32//chratio, 3, 2, 1, use_bn=use_bn)
            lshape = [(64//chratio, self.cur_size, self.cur_size), (32//chratio, self.cur_size*2, self.cur_size*2)]
            lrc.append_layer(conv9, lshape, [lrc.get_last_layer_id()])

            self.cur_size = self.cur_size * 2
        else:
            conv9 = ConvBNReLU(64//chratio, 32//chratio, 3, 1, 1, use_bn=use_bn)
            lshape = [(64//chratio, self.cur_size, self.cur_size), (32//chratio, self.cur_size, self.cur_size)]
            lrc.append_layer(conv9, lshape, [lrc.get_last_layer_id()])

        conv10 = ConvBNReLU(32//chratio, 32//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(32//chratio, self.cur_size, self.cur_size), (32//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv10, lshape, [lrc.get_last_layer_id()])

        conv11 = DeconvLayer(32//chratio, 16//chratio, 3, 2, 1, use_bn=use_bn)
        lshape = [(32//chratio, self.cur_size, self.cur_size), (16//chratio, self.cur_size*2, self.cur_size*2)]
        lrc.append_layer(conv11, lshape, [lrc.get_last_layer_id()])

        self.cur_size = self.cur_size * 2

        conv12 = ConvBNReLU(16//chratio, 32//chratio, 3, 1, 1, use_bn=use_bn)
        lshape = [(16//chratio, self.cur_size, self.cur_size), (32//chratio, self.cur_size, self.cur_size)]
        lrc.append_layer(conv12, lshape, [lrc.get_last_layer_id()])

        conv13 = DeconvLayer(32//chratio, 16//chratio, 3, 2, 1, use_bn=use_bn)
        lshape = [(32//chratio, self.cur_size, self.cur_size), (16//chratio, self.cur_size*2, self.cur_size*2)]
        lrc.append_layer(conv13, lshape, [lrc.get_last_layer_id()])

        self.cur_size = self.cur_size * 2
        # conv14 is the same for every model, so it is removed
        

    def create_model_normal(self, lrc, llids, chratio, imratio, use_bn=False, out_width=256):
        self.create_model_autoencoder(lrc, llids, chratio, imratio, use_bn, out_width)
