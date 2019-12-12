import os
import torch
import argparse
from Configs.base_options import BaseOptions

import pdb

#######################################################################################################################
# This script is based on the CycleGan & Pix2Pix code repository
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#######################################################################################################################

class ShapeNetPartOptions(BaseOptions):
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--indir', type=str, default='/mnt/Data/ShapenetPart/', help='input folder')
        parser.add_argument('--num_classes', type=int, default=16, help='number of object categories')
        parser.add_argument('--num_parts', type=int, default=50, help='number of object categories')
        parser.add_argument('--num_points_training', type=int, default=2048, help='number of points to use for training')
        parser.add_argument('--in_channel_x_complete', type=int, default=3, help='dimension of the features (normals, ...)')
        parser.add_argument('--in_channel', type=int, default=3, help='dimension of the spatial input')
        self.network = parser.parse_known_args()[0].network
        if self.network == 'PointNet++':
            parser = self.initialize_pointnet2(parser)
        elif self.network == 'mRes':
            parser = self.initialize_mRes(parser)
        elif self.network == 'mResX':
            parser = self.initialize_mResX(parser)
        elif self.network == 'convPN':
            parser = self.initialize_convPN(parser)
        elif self.network == 'deepConvPN':
            parser = self.initialize_deepConvPN(parser)
        self.isTrain = True
        return parser
    
    def initialize_pointnet2(self, parser):
        parser.add_argument('--name', type=str, default='training_pointnet2_shapenetpart', help='training run name')
        parser.add_argument('--desc', type=str, default='My training on ShapeNet-Part with Pointnet++.', help='description')
        parser.add_argument('--refine', type=str, default='', help='refine model at this path')
        parser.add_argument('--batch_size', type=int, default=8, help='maximum number of samples within a batch')
        parser.add_argument('--nb_subsampled_points', type=lambda x: eval(x), default=[512, 128, 128], help='resolution of the point cloud at each step of the encoding (3 non increasing integers)')
        parser.add_argument('--nb_neighbours', type=lambda x: eval(x), default=[[32, 64, 128],
                                                                                [64, 96, 128],
                                                                                [None]], help='number of neighbours to consider for each resolution and each of the encoding steps')
        parser.add_argument('--sampling_method', type=lambda x: eval(x), default=[['query_ball'] * 3,
                                                                                  ['query_ball'] * 3,
                                                                                  [None]], help='sampling method to use for each resolution and each of the encoding steps')
        parser.add_argument('--patch_radius', type=lambda x: eval(x), default=[[0.1, 0.2, 0.4],
                                                                               [0.2, 0.4, 0.8],
                                                                               [None]], help='radius of the query ball (if chosen) for each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding1', type=lambda x: eval(x), default=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                                                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                                                              [[256, 512, 1024]]], help='kernel size of the mlp applied before the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--use_x', type=lambda x: eval(x), default=[[True, True, True],
                                                                        [True, True, True],
                                                                        [True]], help='whether to use x as additional features for the linear layers at each resolution and each of the encoding steps')
        parser.add_argument('--pooling_operation', type=lambda x: eval(x), default=[['max']*3,
                                                                                    ['max']*3,
                                                                                    ['max']], help='pooling layer to use at each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding2', type=lambda x: eval(x), default=[[[]]*3,
                                                                                     [[]]*3,
                                                                                     [[]]], help='kernel size of the mlp applied after the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--intermediate_size_fc', type=lambda x: eval(x), default=[512, 256], help='dimension of the outout linear layers')
        parser.add_argument('--dropout_rate', type=lambda x: eval(x), default=[0.7], help='dropout rate to use')
        parser.add_argument('--weight_decay', default=5*1e-4, help='weight decay')
        parser.add_argument('--nb_interpolating_points', type=lambda x: eval(x), default=[3, 3, 3], help='number of points to use for interpolation at each of the decoding steps')
        parser.add_argument('--use_x_complete_unsampled', type=lambda x: eval(x), default=[True, True, True], help='whether to use skiplinks at each of the decoding steps')
        parser.add_argument('--list_dim_channels_decoding', type=lambda x: eval(x), default=[[256, 256],
                                                                                             [256, 128],
                                                                                             [128, 128]], help='kernel size of the mlp applied after the pooling at each of the decoding steps')
        return parser
    
    def initialize_mRes(self, parser):
        parser.add_argument('--name', type=str, default='training_mres_shapenetpart', help='training run name')
        parser.add_argument('--desc', type=str, default='My training on ShapeNet-Part with mRes.', help='description')
        parser.add_argument('--refine', type=str, default='', help='refine model at this path')
        parser.add_argument('--batch_size', type=int, default=8, help='maximum number of samples within a batch')
        parser.add_argument('--nb_subsampled_points', type=lambda x: eval(x), default=[[512, 256, 128],
                                                                                       [128, 96, 64],
                                                                                       [128]], help='resolution of the point cloud at each of the encoding steps (3 non increasing lists)')
        parser.add_argument('--nb_neighbours', type=lambda x: eval(x), default=[[32, 32, 32],
                                                                                [64, 64, 64],
                                                                                [None]], help='number of neighbours to consider for each resolution and each of the encoding steps')
        parser.add_argument('--sampling_method', type=lambda x: eval(x), default=[['query_ball', 'query_ball', 'query_ball'],
                                                                                  ['query_ball', 'query_ball', 'query_ball'],
                                                                                  [None]], help='sampling method to use for each resolution and each of the encoding steps')
        parser.add_argument('--patch_radius', type=lambda x: eval(x), default=[[0.1, 0.2, 0.4],
                                                                               [0.2, 0.4, 0.8],
                                                                               [None]], help='radius of the query ball (if chosen) for each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding1', type=lambda x: eval(x), default=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                                                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                                                              [[256, 512, 1024]]], help='kernel size of the mlp applied before the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--use_x', type=lambda x: eval(x), default=[[True, True, True],
                                                                        [True, True, True],
                                                                        [True]], help='whether to use x as additional features for the linear layers at each resolution and each of the encoding steps')
        parser.add_argument('--cross_connection', type=lambda x: eval(x), default=[[False, False],
                                                                                   [False, False],
                                                                                   [None, None]], help='whether to use x-link at each resolution and each of the encoding steps')
        parser.add_argument('--pooling_operation', type=lambda x: eval(x), default=[['max', 'max', 'max'],
                                                                                    ['max', 'max', 'max'],
                                                                                    ['max']], help='pooling layer to use at each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding2', type=lambda x: eval(x), default=[[[]] * 3,
                                                                                     [[]] * 3,
                                                                                     [[]]], help='kernel size of the mlp applied after the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--intermediate_size_fc', type=lambda x: eval(x), default=[512, 256], help='dimension of the outout linear layers')
        parser.add_argument('--dropout_rate', type=lambda x: eval(x), default=[0.7], help='dropout rate to use')
        parser.add_argument('--weight_decay', default=5*1e-4, help='weight decay')
        parser.add_argument('--nb_interpolating_points', type=lambda x: eval(x), default=[3, 3, 3], help='number of points to use for interpolation at each of the decoding steps')
        parser.add_argument('--use_x_complete_unsampled', type=lambda x: eval(x), default=[True, True, True], help='whether to use skiplinks at each of the decoding steps')
        parser.add_argument('--list_dim_channels_decoding', type=lambda x: eval(x), default=[[256, 256],
                                                                                             [256, 128],
                                                                                             [128, 128]], help='kernel size of the mlp applied after the pooling at each of the decoding steps')
        parser.add_argument('--dropout_rate_cross', type=int, default=0, help='dropout rate to use for all of the crosslinks')
        parser.add_argument('--nb_interpolating_points_encoding', type=lambda x: eval(x), default=[[8, 8],
                                                                                                   [8, 8],
                                                                                                   [None]], help='number of points to use for interpolation at each resolution and at each of the encoding steps')
        parser.add_argument('--nb_interpolating_points_crossconnection', type=lambda x: eval(x), default=[[[8, 8], [8, 8]],
                                                                                                          [[8, 8], [8, 8]],
                                                                                                          [None]], help='number of points to use for interpolation for each crosslinks at each resolution and at each of the encoding steps')
        return parser
    
    def initialize_mResX(self, parser):
        parser.add_argument('--name', type=str, default='training_mresx_shapenetpart', help='training run name')
        parser.add_argument('--desc', type=str, default='My training on ShapeNet-Part with mResX.', help='description')
        parser.add_argument('--refine', type=str, default='', help='refine model at this path')
        parser.add_argument('--batch_size', type=int, default=8, help='maximum number of samples within a batch')
        parser.add_argument('--nb_subsampled_points', type=lambda x: eval(x), default=[[512, 256, 128],
                                                                                       [128, 96, 64],
                                                                                       [128]], help='resolution of the point cloud at each of the encoding steps (3 non increasing lists)')
        parser.add_argument('--nb_neighbours', type=lambda x: eval(x), default=[[32, 32, 32],
                                                                                [64, 64, 64],
                                                                                [None]], help='number of neighbours to consider for each resolution and each of the encoding steps')
        parser.add_argument('--sampling_method', type=lambda x: eval(x), default=[['query_ball', 'query_ball', 'query_ball'],
                                                                                  ['query_ball', 'query_ball', 'query_ball'],
                                                                                  [None]], help='sampling method to use for each resolution and each of the encoding steps')
        parser.add_argument('--patch_radius', type=lambda x: eval(x), default=[[0.1, 0.2, 0.4],
                                                                               [0.2, 0.4, 0.8],
                                                                               [None]], help='radius of the query ball (if chosen) for each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding1', type=lambda x: eval(x), default=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                                                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                                                              [[256, 512, 1024]]], help='kernel size of the mlp applied before the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--use_x', type=lambda x: eval(x), default=[[True, True, True],
                                                                        [True, True, True],
                                                                        [True]], help='whether to use x as additional features for the linear layers at each resolution and each of the encoding steps')
        parser.add_argument('--cross_connection', type=lambda x: eval(x), default=[[True, True],
                                                                                   [True, True],
                                                                                   [None, None]], help='whether to use x-link at each resolution and each of the encoding steps')
        parser.add_argument('--pooling_operation', type=lambda x: eval(x), default=[['max', 'max', 'max'],
                                                                                    ['max', 'max', 'max'],
                                                                                    ['max']], help='pooling layer to use at each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding2', type=lambda x: eval(x), default=[[[]] * 3,
                                                                                     [[]] * 3,
                                                                                     [[]]], help='kernel size of the mlp applied after the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--intermediate_size_fc', type=lambda x: eval(x), default=[512, 256], help='dimension of the outout linear layers')
        parser.add_argument('--dropout_rate', type=lambda x: eval(x), default=[0.7], help='dropout rate to use')
        parser.add_argument('--weight_decay', default=5*1e-4, help='weight decay')
        parser.add_argument('--nb_interpolating_points', type=lambda x: eval(x), default=[3, 3, 3], help='number of points to use for interpolation at each of the decoding steps')
        parser.add_argument('--use_x_complete_unsampled', type=lambda x: eval(x), default=[True, True, True], help='whether to use skiplinks at each of the decoding steps')
        parser.add_argument('--list_dim_channels_decoding', type=lambda x: eval(x), default=[[256, 256],
                                                                                             [256, 128],
                                                                                             [128, 128]], help='kernel size of the mlp applied after the pooling at each of the decoding steps')
        parser.add_argument('--dropout_rate_cross', type=int, default=0, help='dropout rate to use for all of the crosslinks')
        parser.add_argument('--nb_interpolating_points_encoding', type=lambda x: eval(x), default=[[8, 8],
                                                                                                   [8, 8],
                                                                                                   [None]], help='number of points to use for interpolation at each resolution and at each of the encoding steps')
        parser.add_argument('--nb_interpolating_points_crossconnection', type=lambda x: eval(x), default=[[[8, 8], [8, 8]],
                                                                                                          [[8, 8], [8, 8]],
                                                                                                          [None]], help='number of points to use for interpolation for each crosslinks at each resolution and at each of the encoding steps')
        return parser
    
    def initialize_convPN(self, parser):
        parser.add_argument('--name', type=str, default='training_convpn_shapenetpart', help='training run name')
        parser.add_argument('--desc', type=str, default='My training on ShapeNet-Part with convPN.', help='description')
        parser.add_argument('--refine', type=str, default='', help='refine model at this path')
        parser.add_argument('--batch_size', type=int, default=8, help='maximum number of samples within a batch')
        parser.add_argument('--nb_subsampled_points', type=lambda x: eval(x), default=[[512, 256, 128],
                                                                                       [128, 96, 64]], help='resolution of the point cloud at each of the encoding steps (3 non increasing lists)')
        parser.add_argument('--nb_neighbours', type=lambda x: eval(x), default=[[32, 32, 32, 32],
                                                                                [64, 64, 64, 64]], help='number of neighbours to consider for each resolution and each of the encoding steps')
        parser.add_argument('--sampling_method', type=lambda x: eval(x), default=[['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                                                                                  ['query_ball', 'query_ball', 'query_ball', 'query_ball']], help='sampling method to use for each resolution and each of the encoding steps')
        parser.add_argument('--patch_radius', type=lambda x: eval(x), default=[[0.1, 0.2, 0.4, 0.4],
                                                                               [0.2, 0.4, 0.8, 0.8]], help='radius of the query ball (if chosen) for each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding', type=lambda x: eval(x), default=[[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                                                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                                                             [256, 512, 1024]], help='kernel size of the mlp applied before the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--use_x', type=lambda x: eval(x), default=[[True, True, True, True],
                                                                        [True, True, True, True],
                                                                        True], help='whether to use x as additional features for the linear layers at each resolution and each of the encoding steps')
        parser.add_argument('--use_crosslinks', type=lambda x: eval(x), default=[False, False], help='whether to use x-link at each of the encoding steps')
        parser.add_argument('--use_reslinks', type=lambda x: eval(x), default=[True, True], help='whether to use residual-link at each of the encoding steps')
        parser.add_argument('--sequence', type=lambda x: eval(x), default=[['C', 'C', 'C', 'S'],
                                                                           ['C', 'C', 'C', 'S']], help='sequence of block to select (C or S) to use for each of the encoding steps')
        parser.add_argument('--pooling_operation', type=lambda x: eval(x), default=[['max', 'max', 'max', 'max'],
                                                                                    ['max', 'max', 'max', 'max'],
                                                                                    'max'], help='pooling layer to use at each resolution and each of the encoding steps')
        parser.add_argument('--residuallinks_input', type=lambda x: eval(x), default=[], help='input of the residual links for the third encoding block')
        parser.add_argument('--residuallinks_output', type=lambda x: eval(x), default=[], help='output of the residual links for the third encoding block')
        parser.add_argument('--intermediate_size_fc', type=lambda x: eval(x), default=[512, 256], help='dimension of the outout linear layers')
        parser.add_argument('--dropout_rate', type=lambda x: eval(x), default=[0.7], help='dropout rate to use')
        parser.add_argument('--weight_decay', default=5*1e-4, help='weight decay')
        parser.add_argument('--nb_interpolating_points', type=lambda x: eval(x), default=[[8, 8, 8], [8, 8, 8], None, 3, 3, 3], help='number of points to use for interpolation at each of the decoding steps')
        parser.add_argument('--use_x_complete_unsampled', type=lambda x: eval(x), default=[True, True, True], help='whether to use skiplinks at each of the decoding steps')
        parser.add_argument('--list_dim_channels_decoding', type=lambda x: eval(x), default=[[256, 256],
                                                                                             [256, 128],
                                                                                             [128, 128]], help='kernel size of the mlp applied after the pooling at each of the decoding steps')
        parser.add_argument('--blockout_rate', type=lambda x: eval(x), default=[[0,0,0,0,0,0],
                                                                                [0,0,0,0,0,0]], help='dropout rate of each of the convolutional block at each of the encoding steps')
        return parser
    
    def initialize_deepConvPN(self, parser):
        parser.add_argument('--name', type=str, default='training_deepconvpn_shapenetpart', help='training run name')
        parser.add_argument('--desc', type=str, default='My training on ShapeNet-Part with deepConvPN.', help='description')
        parser.add_argument('--refine', type=str, default='', help='refine model at this path')
        parser.add_argument('--batch_size', type=int, default=8, help='maximum number of samples within a batch')
        parser.add_argument('--nb_subsampled_points', type=lambda x: eval(x), default=[[512, 256, 128],
                                                                                       [128, 96, 64]], help='resolution of the point cloud at each of the encoding steps (3 non increasing lists)')
        parser.add_argument('--nb_neighbours', type=lambda x: eval(x), default=[[32, 32, 32, 32],
                                                                                [64, 64, 64, 64]], help='number of neighbours to consider for each resolution and each of the encoding steps')
        parser.add_argument('--sampling_method', type=lambda x: eval(x), default=[['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                                                                                  ['query_ball', 'query_ball', 'query_ball', 'query_ball']], help='sampling method to use for each resolution and each of the encoding steps')
        parser.add_argument('--patch_radius', type=lambda x: eval(x), default=[[0.1, 0.2, 0.4, 0.4],
                                                                               [0.2, 0.4, 0.8, 0.8]], help='radius of the query ball (if chosen) for each resolution and each of the encoding steps')
        parser.add_argument('--list_dim_channels_encoding', type=lambda x: eval(x), default=[[[32, 32, 32, 32, 64, 64], [64, 64, 64, 64, 128, 128], [64, 64, 96, 96, 128, 128]],
                                                                                             [[64, 64, 64, 64, 128, 128], [128, 128, 128, 128, 256, 256], [128, 128, 128, 128, 256, 256]],
                                                                                             [256, 128, 256, 512, 128, 512, 1024, 128, 1024]], help='kernel size of the mlp applied before the pooling at each resolution and each of the encoding steps')
        parser.add_argument('--use_x', type=lambda x: eval(x), default=[[True, True, True, True],
                                                                        [True, True, True, True],
                                                                        True], help='whether to use x as additional features for the linear layers at each resolution and each of the encoding steps')
        parser.add_argument('--use_crosslinks', type=lambda x: eval(x), default=[False, False], help='whether to use x-link at each of the encoding steps')
        parser.add_argument('--use_reslinks', type=lambda x: eval(x), default=[True, True], help='whether to use residual-link at each of the encoding steps')
        parser.add_argument('--sequence', type=lambda x: eval(x), default=[['C', 'C', 'C', 'S', 'C', 'C', 'C'],
                                                                           ['C', 'C', 'C', 'S', 'C', 'C', 'C']], help='sequence of block to select (C or S) to use for each of the encoding steps')
        parser.add_argument('--pooling_operation', type=lambda x: eval(x), default=[['max', 'max', 'max', 'max'],
                                                                                    ['max', 'max', 'max', 'max'],
                                                                                    'max'], help='pooling layer to use at each resolution and each of the encoding steps')
        parser.add_argument('--residuallinks_input', type=lambda x: eval(x), default=[1, 4, 7], help='input of the residual links for the third encoding block')
        parser.add_argument('--residuallinks_output', type=lambda x: eval(x), default=[2, 5, 8], help='output of the residual links for the third encoding block')
        parser.add_argument('--intermediate_size_fc', type=lambda x: eval(x), default=[512, 256], help='dimension of the outout linear layers')
        parser.add_argument('--dropout_rate', type=lambda x: eval(x), default=[0.7], help='dropout rate to use')
        parser.add_argument('--weight_decay', default=5*1e-4, help='weight decay')
        parser.add_argument('--nb_interpolating_points', type=lambda x: eval(x), default=[[8, 8, 8], [8, 8, 8], None, 3, 3, 3], help='number of points to use for interpolation at each of the decoding steps')
        parser.add_argument('--use_x_complete_unsampled', type=lambda x: eval(x), default=[True, True, True], help='whether to use skiplinks at each of the decoding steps')
        parser.add_argument('--list_dim_channels_decoding', type=lambda x: eval(x), default=[[256, 256],
                                                                                             [256, 128],
                                                                                             [128, 128]], help='kernel size of the mlp applied after the pooling at each of the decoding steps')
        parser.add_argument('--blockout_rate', type=lambda x: eval(x), default=[[0,0,0,0,0,0],
                                                                                [0,0,0,0,0,0]], help='dropout rate of each of the convolutional block at each of the encoding steps')
        return parser