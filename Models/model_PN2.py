# Importation of librairies
import math
import torch
import numpy as np
from Models.model_utils import *

import pdb

#######################
#######################
class PointNetSetAbstraction(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, in_channel_x_complete, in_channel_x, list_dim_channels1,\
                 pooling_operation, use_x, list_dim_channels2):
        super(PointNetSetAbstraction, self).__init__()
        self.nb_subsampled_points = nb_subsampled_points
        self.nb_neighbours = nb_neighbours
        self.sampling_method = sampling_method
        self.patch_radius = patch_radius
        self.nb_scales = len(self.patch_radius)
        self.pooling_operation = pooling_operation
        self.use_x = use_x
        self.length_conv1_bank = []
        self.length_conv2_bank = []
        self.last_channel = 0
        self.flag_nofeatures = in_channel_x_complete is None
        if self.flag_nofeatures:
            in_channel_x_complete = in_channel_x
        if self.patch_radius[0] is not None:
            self.sampling = Sampling(self.nb_subsampled_points)
        self.groupinglist = torch.nn.ModuleList()
        self.convlist1 = torch.nn.ModuleList()
        self.bnlist1 = torch.nn.ModuleList()
        self.convlist2 = torch.nn.ModuleList()
        self.bnlist2 = torch.nn.ModuleList()
        self.intermediate_channel = []
        for i in range(self.nb_scales):
            if self.patch_radius[i] is None:
                self.groupinglist.append(Grouping_all(self.nb_subsampled_points, self.use_x[i]))
            else:
                self.groupinglist.append(Grouping(self.nb_neighbours[i], self.sampling_method[i],\
                                                  self.use_x[i], batch_size, self.patch_radius[i]))
            last_channel = in_channel_x_complete
            if self.use_x[i] and not self.flag_nofeatures:
                last_channel += in_channel_x
            for out_channel in list_dim_channels1[i]:
                self.convlist1.append(torch.nn.Conv1d(last_channel, out_channel, 1, bias=True))
                self.bnlist1.append(torch.nn.BatchNorm1d(out_channel))
                last_channel = out_channel
            self.intermediate_channel += [last_channel]
            if self.pooling_operation[i] == 'max_and_avg':
                last_channel *= 2
            if len(list_dim_channels2[i]) > 0:
                for out_channel in list_dim_channels2[i]:
                    self.convlist2.append(torch.nn.Conv1d(last_channel, out_channel, 1, bias=True))
                    self.bnlist2.append(torch.nn.BatchNorm1d(out_channel))
                    last_channel = out_channel
            self.length_conv1_bank += [len(list_dim_channels1[i])]
            self.length_conv2_bank += [len(list_dim_channels2[i])]
            self.last_channel += last_channel
    def update_decay(self, bn_decay_value):
        for i in range(len(self.bnlist1)):
            self.bnlist1[i].momentum = bn_decay_value
        for i in range(len(self.bnlist2)):
            self.bnlist2[i].momentum = bn_decay_value
    def forward(self, x, x_complete, bn_decay_value=None):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        batch_size, num_points, dim_channels_x = x.size()
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(dim_channels_x) == torch.Tensor:
            dim_channels_x = dim_channels_x.item()
        # Sampling
        if self.patch_radius[0] is not None:
            indexes_downsampling = self.sampling(x)
            subsampled_centroids = self.sampling.extract_values(x, indexes_downsampling)
        cpt_conv1_bank = 0
        cpt_conv2_bank = 0
        list_output_channel_wise = []
        for i in range(self.nb_scales):
            # Grouping
            if self.patch_radius[i] is None:
                subsampled_centroids, indexes = self.groupinglist[i](x)
                x_complete_ = self.groupinglist[i].extract_values(x, x_complete)
            else:
                indexes = self.groupinglist[i](x, subsampled_centroids)
                x_, x_complete_ = self.groupinglist[i].extract_values(x, x_complete, indexes)
                if self.use_x[i] and (not self.flag_nofeatures):
                    x_complete_ = torch.cat((x_complete_, x_), dim=3)
            _, _, _, dim_channels_x_complete = x_complete_.size()
            if type(dim_channels_x_complete) == torch.Tensor:
                dim_channels_x_complete = dim_channels_x_complete.item()
            if self.patch_radius[i] is not None:
                x_complete_ = x_complete_.view(batch_size, self.nb_subsampled_points * self.nb_neighbours[i],\
                                               dim_channels_x_complete).transpose(1, 2)
            else:
                x_complete_ = x_complete_.view(batch_size, num_points, dim_channels_x_complete).transpose(1, 2)
            # MLP layers
            for j in range(self.length_conv1_bank[i]):
                x_complete_ = torch.nn.functional.relu(self.bnlist1[cpt_conv1_bank + j](self.convlist1[cpt_conv1_bank + j](x_complete_)))

            cpt_conv1_bank += self.length_conv1_bank[i]
            if self.patch_radius[i] is not None:
                x_complete_ = x_complete_.contiguous().view(batch_size, self.intermediate_channel[i],\
                                                            self.nb_subsampled_points, self.nb_neighbours[i])
            else:
                x_complete_ = x_complete_.contiguous().view(batch_size, self.intermediate_channel[i], 1, num_points)
            # Pooling operation
            mask = indexes.unsqueeze(1).repeat(1, self.intermediate_channel[i], 1, 1) == -1
            if self.pooling_operation[i] == 'max':
                x_complete_[mask] = -np.inf
                x_complete_, _ = torch.max(x_complete_, 3)
            elif self.pooling_operation[i] == 'weighted_avg':
                norms = torch.sqrt(torch.sum(x_**2, dim=3, keepdim=True))
                norms = torch.exp(-norms * 5)
                mask = indexes.unsqueeze(3) == -1
                norms[mask] = 0
                normalization = torch.sum(norms, dim=2, keepdim=True).repeat(1, 1, self.nb_neighbours[i], 1)
                weights = norms / normalization
                weights = weights.repeat(1, 1, 1, self.intermediate_channel).permute(0, 3, 1, 2)
                x_complete_ = x_complete_ * weights
                x_complete_ = torch.sum(x_complete_, dim=3)
            else:
                nb_true = torch.sum(1 - mask, dim=3).float()
                if self.pooling_operation[i] == 'avg':
                    x_complete_[mask] = 0
                    x_complete_ = torch.sum(x_complete_, dim=3) / nb_true
                elif self.pooling_operation[i] == 'max_and_avg':
                    x_complete_[mask] = -np.inf
                    x1, _ = torch.max(x_complete_, 3)
                    x_complete_[mask] = 0
                    x2 = torch.sum(x_complete_, dim=3) / nb_true
                    x_complete_ = torch.cat([x1, x2], dim=1)
            # Additional MLP layers
            for j in range(self.length_conv2_bank[i]):
                x_complete_ = torch.nn.functional.relu(self.bnlist2[cpt_conv2_bank + j](self.convlist2[cpt_conv2_bank + j](x_complete_)))
            cpt_conv2_bank += self.length_conv2_bank[i]
            x_complete_ = x_complete_.transpose(1, 2)
            list_output_channel_wise.append(x_complete_)
        x_complete = torch.cat(list_output_channel_wise, dim=2)
        return subsampled_centroids, x_complete

class PointNet2(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, \
                 in_channel_x_complete, in_channel, list_dim_channels1, use_x,pooling_operation, list_dim_channels2,\
                 intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled, \
                 list_dim_channels, num_classes, num_parts):
        super(PointNet2, self).__init__()
        self.pointnet_downsampling1 = PointNetSetAbstraction(batch_size, nb_subsampled_points[0], nb_neighbours[0], sampling_method[0], \
                                      patch_radius[0], in_channel_x_complete, in_channel, list_dim_channels1[0],\
                                      pooling_operation[0], use_x[0], list_dim_channels2[0])
        in_channel_x_complete1 = self.pointnet_downsampling1.last_channel
        self.pointnet_downsampling2 = PointNetSetAbstraction(batch_size, nb_subsampled_points[1], nb_neighbours[1], sampling_method[1], \
                                      patch_radius[1], in_channel_x_complete1, in_channel, list_dim_channels1[1], \
                                      pooling_operation[1], use_x[1], list_dim_channels2[1])
        in_channel_x_complete2 = self.pointnet_downsampling2.last_channel
        self.pointnet_downsampling3 = PointNetSetAbstraction(batch_size, nb_subsampled_points[2], nb_neighbours[2], sampling_method[2], \
                                      patch_radius[2], in_channel_x_complete2, in_channel, list_dim_channels1[2], \
                                      pooling_operation[2], use_x[2], list_dim_channels2[2])
        if use_x_complete_unsampled[0]:
            in_channel_x_complete3 = self.pointnet_downsampling3.last_channel
        self.pointnet_upsampling1 = PointNetFeaturePropagation(nb_interpolating_points[0], use_x_complete_unsampled[0], \
                                    in_channel_x_complete3, in_channel_x_complete2, list_dim_channels[0], batch_size)
        if use_x_complete_unsampled[1]:
            in_channel_x_complete4 = self.pointnet_upsampling1.last_channel
        self.pointnet_upsampling2 = PointNetFeaturePropagation(nb_interpolating_points[1], use_x_complete_unsampled[1], \
                                    in_channel_x_complete4, in_channel_x_complete1, list_dim_channels[1], batch_size)
        in_channel_x_complete5 = self.pointnet_upsampling2.last_channel
        if use_x_complete_unsampled[2]:
            if in_channel_x_complete is not None:
                in_channel += in_channel_x_complete
        if num_classes is None:
            num_classes = 0
        self.pointnet_upsampling3 = PointNetFeaturePropagation(nb_interpolating_points[2], use_x_complete_unsampled[2], \
                                    in_channel_x_complete5, in_channel + num_classes, list_dim_channels[2], batch_size)
        in_channel_x_complete6 = self.pointnet_upsampling3.last_channel
        self.conv1d1 = torch.nn.Conv1d(in_channel_x_complete6, intermediate_size_fc[0], 1, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(intermediate_size_fc[0])
        self.dropout = torch.nn.Dropout(p=dropout_rate[0])
        self.conv1d2 = torch.nn.Conv1d(intermediate_size_fc[0], num_parts, 1, bias=True)
        if num_classes > 0:
            self.one_hot = torch.nn.Parameter(torch.FloatTensor(1, num_classes).zero_(), requires_grad=False)
    def update_decay(self, bn_decay_value):
        self.bn1.momentum = bn_decay_value
    def forward(self, x, x_complete, labels_cat, bn_decay_value=None):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        num_batch, num_points, _ = x.size()
        if type(num_batch) == torch.Tensor:
            num_batch = num_batch.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        # Abstraction layers
        subsampled_centroids1, x1 = self.pointnet_downsampling1(x, x_complete, bn_decay_value)
        subsampled_centroids2, x2 = self.pointnet_downsampling2(subsampled_centroids1, x1, bn_decay_value)
        subsampled_centroids3, x3 = self.pointnet_downsampling3(subsampled_centroids2, x2, bn_decay_value)
        # Feature Propagation layers
        x4 = self.pointnet_upsampling1(subsampled_centroids2, subsampled_centroids3, x2, x3, bn_decay_value)
        x5 = self.pointnet_upsampling2(subsampled_centroids1, subsampled_centroids2, x1, x4, bn_decay_value)
        if labels_cat is not None:
            target = self.one_hot.repeat(num_batch, 1)
            target = target.scatter_(1, labels_cat.data, 1).unsqueeze(1).repeat(1, num_points, 1)
            x0 = torch.cat([target, x],dim=2)
        else:
            x0 = x
        if x_complete is not None:
            x0 = torch.cat([x0, x_complete], dim=2)
        x6 = self.pointnet_upsampling3(x[:,:,:3], subsampled_centroids1, x0, x5, bn_decay_value)
        # Fully connected layers
        x = x6
        x = x.transpose(1, 2)
        x = torch.nn.functional.relu(self.bn1(self.conv1d1(x)))
        x = self.dropout(x)
        x = self.conv1d2(x)
        x = x.transpose(1, 2)
        return x

if __name__ == '__main__':
    print('Testing the whole network for ShapeNet-Part')
    batch_size = 8
    nb_subsampled_points = [512, 128, 128]
    nb_neighbours = [[32, 64, 128], [64, 96, 128], [None]]
    sampling_method = [['query_ball']*3, ['query_ball']*3, [None]]
    patch_radius = [[0.1,0.2,0.4],[0.2,0.4,0.8],[None]]
    in_channel_x_complete = 3
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]], [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    pooling_operation = [['max']*3, ['max']*3, ['max']]
    list_dim_channels2 = [[[]]*3,[[]]*3,[[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = 16
    num_parts = 50
    model = PointNet2(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, \
                      in_channel_x_complete, in_channel, list_dim_channels1, use_x,pooling_operation, list_dim_channels2,\
                      intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled, \
                      list_dim_channels, num_classes, num_parts)

    batch_size = 4
    num_points_training = 2048
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = torch.randn(batch_size, num_points_training, in_channel_x_complete)
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on PointNet++ for ShapeNet-Part: Success')

if __name__ == '__main__':
    print('Testing the whole network for ScanNet')
    batch_size = 8
    nb_subsampled_points = [512, 128, 128]
    nb_neighbours = [[32, 64, 128], [64, 96, 128], None]
    sampling_method = [['query_ball', 'query_ball', 'query_ball'],
                       ['query_ball', 'query_ball', 'query_ball'], None]
    patch_radius = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], [None]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                          [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    pooling_operation = [['max', 'max', 'max'], ['max', 'max', 'max'], ['max']]
    list_dim_channels2 = [[[]] * 3, [[]] * 3, [[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = None
    num_parts = 21
    model = PointNet2(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, \
                      in_channel_x_complete, in_channel, list_dim_channels1, use_x, pooling_operation, list_dim_channels2, \
                      intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled, \
                      list_dim_channels, num_classes, num_parts)

    batch_size = 4
    num_points_training = 8192
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = None
    output = model(x, x_complete, labels_cat)
    print('Test on PointNet++ for ScanNet: Success')

if __name__ == '__main__':
    print('Testing the whole network for PartNet')
    batch_size = 8
    nb_subsampled_points = [512, 128, 128]
    nb_neighbours = [[32, 64, 128], [64, 96, 128], [None]]
    sampling_method = [['query_ball'] * 3, ['query_ball'] * 3, [None]]
    patch_radius = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], [None]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                          [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    pooling_operation = [['max'] * 3, ['max'] * 3, ['max']]
    list_dim_channels2 = [[[]] * 3, [[]] * 3, [[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = 17
    num_parts = 251
    model = PointNet2(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, \
                      in_channel_x_complete, in_channel, list_dim_channels1, use_x, pooling_operation, list_dim_channels2,\
                      intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled, \
                      list_dim_channels, num_classes, num_parts)

    batch_size = 4
    num_points_training = 10000
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on PointNet++ for PartNet: Success')