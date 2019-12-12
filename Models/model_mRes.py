# Importation of librairies
import torch
import numpy as np
from Models.model_utils import *

import pdb

class PointNetSetAbstraction(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, in_channel_x_complete, in_channel_x, list_dim_channels1, \
                 pooling_operation, use_x, cross_connection, list_dim_channels2, dropout_rate_cross, nb_interpolating_points_encoding):
        # Rq: nb_subsampled_points is now a list
        super(PointNetSetAbstraction, self).__init__()
        self.batch_size = batch_size
        self.nb_subsampled_points = nb_subsampled_points
        self.nb_scales = len(self.nb_subsampled_points)
        self.nb_neighbours = nb_neighbours
        self.sampling_method = sampling_method
        self.patch_radius = patch_radius
        self.in_channel_x = in_channel_x
        self.in_channel_x_complete = in_channel_x_complete
        self.pooling_operation = pooling_operation
        self.use_x = use_x
        self.cross_connection = cross_connection
        self.length_conv1_bank = []
        self.length_conv2_bank = []
        self.dropout_rate_cross = dropout_rate_cross
        self.flag_nofeatures = self.in_channel_x_complete is None
        if self.flag_nofeatures:
            self.in_channel_x_complete = self.in_channel_x
        self.samplinglist = torch.nn.ModuleList()
        self.groupinglist = torch.nn.ModuleList()
        self.interpolatinglist = torch.nn.ModuleList()
        self.convlist1 = torch.nn.ModuleList()
        self.bnlist1 = torch.nn.ModuleList()
        self.convlist2 = torch.nn.ModuleList()
        self.bnlist2 = torch.nn.ModuleList()
        self.intermediate_channel = []
        for i in range(self.nb_scales):
            # Step 1: Upsampling in each of the branch (with increasing downsampling)
            # Step 2: Grouping the points
            if self.patch_radius[i] is None:
                self.samplinglist.append(None)
                self.groupinglist.append(Grouping_all(nb_subsampled_points[i], self.use_x[i]))
            else:
                self.samplinglist.append(Sampling(self.nb_subsampled_points[i]))
                self.groupinglist.append(Grouping(self.nb_neighbours[i], self.sampling_method[i], self.use_x[i], batch_size, self.patch_radius[i]))
            # Step 3: Convolution filter bank 1
            last_channel = self.in_channel_x_complete
            if self.use_x[i] and not self.flag_nofeatures:
                last_channel += in_channel_x
            for j in range(len(list_dim_channels1[i])):
                self.convlist1.append(torch.nn.Conv1d(last_channel, list_dim_channels1[i][j], 1, bias=True))
                self.bnlist1.append(torch.nn.BatchNorm1d(list_dim_channels1[i][j]))
                last_channel = list_dim_channels1[i][j]
            self.intermediate_channel += [last_channel]
            # Step 4: Pooling operation
            if self.pooling_operation[i] == 'max_and_avg':
                last_channel *= 2
            # Step 5: Convolution filter bank 2
            if len(list_dim_channels2[i]) > 0:
                for out_channel in list_dim_channels2[i]:
                    self.convlist2.append(torch.nn.Conv1d(last_channel, out_channel, 1, bias=True))
                    self.bnlist2.append(torch.nn.BatchNorm1d(out_channel))
                    last_channel = out_channel
            # Step 6: Interpolating the points back to the original resolution
            if i > 0:
                self.interpolatinglist.append(Interpolating(nb_interpolating_points_encoding[i - 1], False, batch_size))
            else:
                self.interpolatinglist.append(None)
            self.length_conv1_bank += [len(list_dim_channels1[i])]
            self.length_conv2_bank += [len(list_dim_channels2[i])]
            if i == 0:
                self.last_channel = last_channel
        # Additional convolutions for SumRes purpose
        self.convlist_sumres = torch.nn.ModuleList()
        for i in range(1, self.nb_scales):
            self.convlist_sumres.append(torch.nn.Conv1d(list_dim_channels1[i][-1], list_dim_channels1[0][-1], 1, bias=True))
        self.list_dim_channels1 = list_dim_channels1
        self.cumsum_conv1_bank = np.cumsum([0] + self.length_conv1_bank[:-1])
        self.cumsum_conv2_bank = np.cumsum([0] + self.length_conv2_bank[:-1])
        self.tensor_batch = torch.nn.Parameter(torch.arange(self.batch_size), requires_grad=False)
        self.zeros = torch.nn.Parameter(torch.zeros([1]).type(torch.ByteTensor), requires_grad=False)
        self.ones = torch.nn.Parameter(torch.ones([1]), requires_grad=False)
    def add_cross_connection(self, batch_size, nb_interpolating_points_crossconnection):
        '''
        Adding the cross connections to the current Pointnet++ network
        '''
        # Adding the linear layers for cross connection
        self.linearlist_downsampling = torch.nn.ModuleList()
        self.linearlist_upsampling = torch.nn.ModuleList()
        for j in range(self.length_conv1_bank[0] - 1):
            if self.cross_connection[j]:
                for i in range(self.nb_scales):
                    if i < self.nb_scales - 1:
                        self.linearlist_downsampling.append(
                            torch.nn.Conv1d(self.list_dim_channels1[i][j], self.list_dim_channels1[i + 1][j], 1, bias=True))
                    if i > 0:
                        self.linearlist_upsampling.append(torch.nn.Conv1d(self.list_dim_channels1[i][j], self.list_dim_channels1[i - 1][j], 1,
                                            bias=True))
        # Interpolation
        self.interpolatinglist_crossconnection = torch.nn.ModuleList()
        for i in range(1, self.nb_scales):
            for j in range(1, self.length_conv1_bank[i]):
                if self.cross_connection[j - 1]:
                    self.interpolatinglist_crossconnection.append(Interpolating(nb_interpolating_points_crossconnection[i - 1][j - 1], False, batch_size))
                else:
                    self.interpolatinglist_crossconnection.append(None)
    def update_decay(self, bn_decay_value):
        for i in range(len(self.bnlist1)):
            self.bnlist1[i].momentum = bn_decay_value
        for i in range(len(self.bnlist2)):
            self.bnlist2[i].momentum = bn_decay_value
    def cross_sampling(self, x_complete, indexes, indexes_downsampling, module_reduction, x_unsampled, x_sampled,
                       module_interpolation, nb_neighbours):
        # x_complete: Tensor to sample (either to downsample or upsample) of size batch*dim_x_complete*(num_subsampled_points*num_neighbours)
        # indexes: Tensor of indexes of size batch*num_subsampled_points*num_neighbours
        # indexes_downsampling: Tensor of indexes of size size batch*num_less_subsampled_points (ATTENTION REQUIRED)
        # module_reduction: Dimension reduction module waiting for the input (x_complete)
        # module_interpolation: Interpolation module waiting for the input (x_unsampled,x_sampled,None,x_complete)
        num_batch, feature_dimension, last_dim = x_complete.size()
        _, num_subsampled_points, _ = indexes.size()
        if x_unsampled is not None:
            _, num_target_points, _ = x_unsampled.size()
        else:
            _, num_target_points = indexes_downsampling.size()
        if type(num_batch) == torch.Tensor:
            num_batch = num_batch.item()
        if type(feature_dimension) == torch.Tensor:
            feature_dimension = feature_dimension.item()
        if type(last_dim) == torch.Tensor:
            last_dim = last_dim.item()
        if type(num_subsampled_points) == torch.Tensor:
            num_subsampled_points = num_subsampled_points.item()
        if type(num_target_points) == torch.Tensor:
            num_target_points = num_target_points.item()
        x_complete = x_complete.view(num_batch, feature_dimension, num_subsampled_points, last_dim // num_subsampled_points)
        # MaxPooling
        mask = indexes == -1
        mask = mask.unsqueeze(1).repeat(1, feature_dimension, 1, 1)
        x_complete[mask] = torch.min(x_complete).item()
        x_complete, _ = torch.max(x_complete, dim=3)
        # Reducing feature space dimension
        if module_reduction is not None:
            x_complete = module_reduction(x_complete)
            feature_dimension = module_reduction.out_channels
        if module_interpolation is None:
            #  Downsampling
            x_complete = torch.gather(x_complete, 2, indexes_downsampling.unsqueeze(1).repeat(1, feature_dimension, 1))
        else:
            # Upsampling
            indexes, weights = module_interpolation(x_unsampled, x_sampled)
            x_complete = module_interpolation.extract_values(None, x_complete.transpose(1, 2), indexes, weights).transpose(1, 2)
        x_complete = x_complete.unsqueeze(3).repeat(1, 1, 1, nb_neighbours).view(num_batch, feature_dimension, num_target_points * nb_neighbours)
        return x_complete
    def forward(self, x, x_complete, bn_decay_value, alpha=0):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        batch_size, num_points, _ = x.size()
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if x_complete is None:
            x_complete = x
        # List to save the intermediate outputs before the conv layers
        list_subsampled_centroids = []
        list_x_ = []
        list_x_complete_ = []
        list_indexes = []
        # First Sampling and Grouping
        for i in range(self.nb_scales):
            # Sampling
            if self.patch_radius[i] is not None:
                if i == 0:
                    indexes = self.samplinglist[i](x)
                    subsampled_centroids = self.samplinglist[i].extract_values(x, indexes)
                else:
                    indexes = self.samplinglist[i](list_subsampled_centroids[-1])
                    subsampled_centroids = self.samplinglist[i].extract_values(list_subsampled_centroids[-1], indexes)
            # Grouping
            if self.patch_radius[i] is None:
                subsampled_centroids, indexes = self.groupinglist[i](x)
                x_complete_ = self.groupinglist[i].extract_values(x, x_complete)
                x_ = x_complete_[:,:,:,:3]
            else:
                if i == 0:
                    indexes = self.groupinglist[i](x, subsampled_centroids)
                    x_, x_complete_ =  self.groupinglist[i].extract_values(x, x_complete, indexes)
                else:
                    indexes = self.groupinglist[i](list_subsampled_centroids[-1], subsampled_centroids)
                    x_, x_complete_ =  self.groupinglist[i].extract_values(list_subsampled_centroids[-1], x_complete, indexes)
                if self.use_x[i] and not self.flag_nofeatures:
                    x_complete_ = torch.cat((x_, x_complete_), dim=3)
            x_complete = torch.gather(x_complete, 1, indexes[:, :, 0].unsqueeze(2).repeat(1, 1, self.in_channel_x_complete))
            _, _, _, dim_channels_x_complete = x_complete_.size()
            if type(dim_channels_x_complete) == torch.Tensor:
                dim_channels_x_complete = dim_channels_x_complete.item()
            if self.patch_radius[i] is not None:
                x_complete_ = x_complete_.view(batch_size, self.nb_subsampled_points[i] * self.nb_neighbours[i], \
                                               dim_channels_x_complete).transpose(1, 2)
            else:
                x_complete_ = x_complete_.view(batch_size, num_points, dim_channels_x_complete).transpose(1, 2)
            # Saving the intermediate layers before the conv layers
            list_subsampled_centroids = list_subsampled_centroids + [subsampled_centroids]
            list_x_ = list_x_ + [x_]
            list_x_complete_ = list_x_complete_ + [x_complete_]
            list_indexes = list_indexes + [indexes]
        # Iterating over convolution layers (assuming the same number of conv for each scales)
        cpt_conv = 0
        for j in range(self.length_conv1_bank[0]):
            list_ = []
            for i in range(self.nb_scales):
                x_complete_ = torch.nn.functional.relu(self.bnlist1[self.cumsum_conv1_bank[i] + j](
                    self.convlist1[self.cumsum_conv1_bank[i] + j](list_x_complete_[i + self.nb_scales * j])))
                list_ = list_ + [x_complete_]
            if (j < self.length_conv1_bank[0] - 1) and (self.nb_scales > 1) and self.cross_connection[min(self.length_conv1_bank[0] - 2, j)]:
                for i in range(self.nb_scales):
                    if i == 0:
                        list_x_complete_.append(list_[i] + self.cross_sampling(list_[i + 1], list_indexes[i + 1], None,
                                                                               self.linearlist_upsampling[cpt_conv * (self.nb_scales - 1) + i],
                                                                               list_subsampled_centroids[i],list_subsampled_centroids[i + 1],
                                                                               self.interpolatinglist_crossconnection[i * (self.nb_scales - 1) + j],
                                                                               self.nb_neighbours[i]))
                    elif i == self.nb_scales - 1:
                        list_x_complete_.append(list_[i] + self.cross_sampling(list_[i - 1], list_indexes[i - 1], list_indexes[i][:, :, 0],
                                                           self.linearlist_downsampling[cpt_conv * (self.nb_scales - 1) + i - 1], None, None,
                                                           None, self.nb_neighbours[i]))
                    else:
                        list_x_complete_.append(list_[i] + self.cross_sampling(list_[i + 1], list_indexes[i + 1], None,
                                                                               self.linearlist_upsampling[cpt_conv * (self.nb_scales - 1) + i],
                                                                               list_subsampled_centroids[i],list_subsampled_centroids[i + 1],
                                                                               self.interpolatinglist_crossconnection[i * (self.nb_scales - 1) + j],
                                                                               self.nb_neighbours[i]) + \
                                                self.cross_sampling(list_[i - 1], list_indexes[i - 1], list_indexes[i][:, :, 0],
                                                                    self.linearlist_downsampling[cpt_conv * (self.nb_scales - 1) + i - 1], None,
                                                                    None, None, self.nb_neighbours[i]))
                cpt_conv += 1
            else:
                list_x_complete_ = list_x_complete_ + list_
        list_output_channel_wise = []
        for i in range(self.nb_scales):
            if self.patch_radius[i] is not None:
                x_complete_ = list_x_complete_[self.nb_scales * self.length_conv1_bank[0] + i].contiguous().view(
                    batch_size, self.intermediate_channel[i], self.nb_subsampled_points[i], self.nb_neighbours[i])
            else:
                x_complete_ = list_x_complete_[self.nb_scales * self.length_conv1_bank[0] + i].contiguous().view(
                    batch_size, self.intermediate_channel[i], 1, num_points)
            # Pooling operation
            mask = list_indexes[i].unsqueeze(1).repeat(1, self.intermediate_channel[i], 1, 1) == -1
            if self.pooling_operation[i] == 'max':
                x_complete_[mask] = -np.inf
                x_complete_, _ = torch.max(x_complete_, 3)
            elif self.pooling_operation[i] == 'weighted_avg':
                norms = torch.sqrt(torch.sum(list_x_[i] ** 2, dim=3, keepdim=True))
                norms = torch.exp(-norms * 5)
                mask = list_indexes[i].unsqueeze(3) == -1
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
                x_complete_ = torch.nn.functional.relu(self.bnlist2[self.cumsum_conv2_bank[i] + j](
                    self.convlist2[self.cumsum_conv2_bank[i] + j](x_complete_)))
            x_complete_ = x_complete_.transpose(1, 2)
            #print('Branch ' + str(i) + ': ' + str(x_complete_.size(1)))
            if i > 0:
                indexes, weights = self.interpolatinglist[i](list_subsampled_centroids[0], list_subsampled_centroids[i])
                x_complete_ = self.interpolatinglist[i].extract_values(None, x_complete_, indexes, weights)
            list_output_channel_wise.append(x_complete_)
        x_complete = list_output_channel_wise[0]
        for i in range(1, self.nb_scales):
            x_complete = x_complete + self.convlist_sumres[i - 1](list_output_channel_wise[i].transpose(1, 2)).transpose(1, 2)
        return list_subsampled_centroids[0], x_complete

class mRes(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                 in_channel_x_complete, in_channel, list_dim_channels1, use_x, cross_connection, pooling_operation,
                 list_dim_channels2, intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled,
                 list_dim_channels, num_classes, num_parts, dropout_rate_cross, nb_interpolating_points_encoding):
        super(mRes, self).__init__()
        self.pointnet_downsampling1 = PointNetSetAbstraction(batch_size, nb_subsampled_points[0], nb_neighbours[0], sampling_method[0],
                                                             patch_radius[0], in_channel_x_complete, in_channel, list_dim_channels1[0],
                                                             pooling_operation[0], use_x[0], cross_connection[0], list_dim_channels2[0],
                                                             dropout_rate_cross, nb_interpolating_points_encoding[0])
        in_channel_x_complete1 = self.pointnet_downsampling1.last_channel
        self.pointnet_downsampling2 = PointNetSetAbstraction(batch_size, nb_subsampled_points[1], nb_neighbours[1], sampling_method[1],
                                                             patch_radius[1], in_channel_x_complete1, in_channel, list_dim_channels1[1],
                                                             pooling_operation[1], use_x[1], cross_connection[1], list_dim_channels2[1],
                                                             dropout_rate_cross, nb_interpolating_points_encoding[1])
        in_channel_x_complete2 = self.pointnet_downsampling2.last_channel
        self.pointnet_downsampling3 = PointNetSetAbstraction(batch_size, nb_subsampled_points[2], nb_neighbours[2], sampling_method[2],
                                                             patch_radius[2], in_channel_x_complete2, in_channel, list_dim_channels1[2],
                                                             pooling_operation[2], use_x[2], cross_connection[2], list_dim_channels2[2],
                                                             dropout_rate_cross, nb_interpolating_points_encoding[2])
        in_channel_x_complete3 = self.pointnet_downsampling3.last_channel
        self.pointnet_upsampling1 = PointNetFeaturePropagation(nb_interpolating_points[0], use_x_complete_unsampled[0],
                                                               in_channel_x_complete3, in_channel_x_complete2, list_dim_channels[0], batch_size)

        in_channel_x_complete4 = self.pointnet_upsampling1.last_channel
        self.pointnet_upsampling2 = PointNetFeaturePropagation(nb_interpolating_points[1], use_x_complete_unsampled[1],
                                                               in_channel_x_complete4, in_channel_x_complete1, list_dim_channels[1], batch_size)
        in_channel_x_complete5 = self.pointnet_upsampling2.last_channel
        if in_channel_x_complete is None:
            in_channel_x_complete = 0
        if num_classes is None:
            num_classes = 0
        self.pointnet_upsampling3 = PointNetFeaturePropagation(nb_interpolating_points[2], use_x_complete_unsampled[2],
                                                               in_channel_x_complete5, in_channel + in_channel_x_complete + num_classes,
                                                               list_dim_channels[2], batch_size)
        in_channel_x_complete6 = self.pointnet_upsampling3.last_channel
        self.conv1d1 = torch.nn.Conv1d(in_channel_x_complete6, intermediate_size_fc[0], 1)
        self.bn1 = torch.nn.BatchNorm1d(intermediate_size_fc[0])
        self.dropout = torch.nn.Dropout(p=dropout_rate[0])
        self.conv1d2 = torch.nn.Conv1d(intermediate_size_fc[0], num_parts, 1)
        self.one_hot = torch.nn.Parameter(torch.FloatTensor(1, num_classes).zero_(), requires_grad=False)
    def add_cross_connection(self, batch_size, nb_interpolating_points_crossconnection):
        self.pointnet_downsampling1.add_cross_connection(batch_size, nb_interpolating_points_crossconnection[0])
        self.pointnet_downsampling2.add_cross_connection(batch_size, nb_interpolating_points_crossconnection[1])
    def update_decay(self, bn_decay_value):
        self.bn1.momentum = bn_decay_value
    def forward(self, x, x_complete, labels_cat, bn_decay_value=0.99, alpha=0):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        num_batch, num_points, _ = x.size()
        if type(num_batch) == torch.Tensor:
            num_batch = num_batch.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        # Abstraction layers
        #print('----Encoding 1----')
        subsampled_centroids1, x1 = self.pointnet_downsampling1(x, x_complete, bn_decay_value, alpha)
        #print('----Encoding 2----')
        subsampled_centroids2, x2 = self.pointnet_downsampling2(subsampled_centroids1, x1, bn_decay_value, alpha)
        #print('----Encoding 3----')
        subsampled_centroids3, x3 = self.pointnet_downsampling3(subsampled_centroids2, x2, bn_decay_value, alpha)
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
        x6 = self.pointnet_upsampling3(x[:, :, :3], subsampled_centroids1, x0, x5, bn_decay_value)
        # Fully connected layers
        x = x6
        x = x.transpose(1, 2)
        x = torch.nn.functional.relu(self.bn1(self.conv1d1(x)))
        x = self.dropout(x)
        x = self.conv1d2(x)
        x = x.transpose(1, 2)
        return x

if __name__=='__main__':
    print('Testing the whole network for ShapeNet-Part')
    batch_size = 8
    nb_subsampled_points = [[512, 256, 128], [128, 96, 64], [128]]
    nb_neighbours = [[32, 32, 32], [64, 64, 64], [None]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball'], ['query_ball', 'query_ball', 'query_ball'], [None]]
    patch_radius = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], [None]]
    in_channel_x_complete = 3
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                          [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    cross_connection = [[True, True], [True, True], [None, None]]
    pooling_operation = [['max', 'max', 'max'], ['max', 'max', 'max'], ['max']]
    list_dim_channels2 = [[[]] * 3, [[]] * 3, [[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = 16
    num_parts = 50
    dropout_rate_cross = 0
    nb_interpolating_points_encoding = [[8, 8], [8, 8], [None]]
    nb_interpolating_points_crossconnection = [[[8, 8], [8, 8]], [[8, 8], [8, 8]], [None]]
    model = mRes(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                 in_channel_x_complete, in_channel, list_dim_channels1, use_x, cross_connection, pooling_operation,
                 list_dim_channels2, intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled,
                 list_dim_channels, num_classes, num_parts, dropout_rate_cross, nb_interpolating_points_encoding)
    model.add_cross_connection(batch_size, nb_interpolating_points_crossconnection)

    batch_size = 4
    num_points_training = 2048
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = torch.randn(batch_size, num_points_training, in_channel_x_complete)
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on mRes for ShapeNet-Part: Success')

if __name__=='__main__':
    print('Testing the whole network for ScanNet')
    batch_size = 8
    nb_subsampled_points =[[512, 256, 128], [128, 96, 64], [128]]
    nb_neighbours = [[32, 32, 32, 32], [64, 64, 64, 64], [None]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                       ['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                       [None]]
    patch_radius = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], [None]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                          [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    cross_connection = [[True, True], [True, True], [None, None]]
    pooling_operation = [['max', 'max', 'max'], ['max', 'max', 'max'], ['max']]
    list_dim_channels2 = [[[]] * 3, [[]] * 3, [[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = None
    num_parts = 21
    dropout_rate_cross = 0
    nb_interpolating_points_encoding = [[8, 8], [8, 8], [None]]
    nb_interpolating_points_crossconnection = [[[8, 8], [8, 8]], [[8, 8], [8, 8]], [None]]
    model = mRes(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                 in_channel_x_complete, in_channel, list_dim_channels1, use_x, cross_connection, pooling_operation,
                 list_dim_channels2, intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled,
                 list_dim_channels, num_classes, num_parts, dropout_rate_cross, nb_interpolating_points_encoding)
    model.add_cross_connection(batch_size, nb_interpolating_points_crossconnection)

    batch_size = 4
    num_points_training = 8192
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = None
    output = model(x, x_complete, labels_cat)
    print('Test on mRes for ScanNet: Success')

if __name__=='__main__':
    print('Testing the whole network for PartNet')
    batch_size = 8
    nb_subsampled_points = [[512, 256, 128], [128, 96, 64], [128]]
    nb_neighbours = [[32, 32, 32], [64, 64, 64], [None]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball'], ['query_ball', 'query_ball', 'query_ball'], [None]]
    patch_radius = [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], [None]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels1 = [[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                          [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                          [[256, 512, 1024]]]
    use_x = [[True, True, True], [True, True, True], [True]]
    cross_connection = [[True, True], [True, True], [None, None]]
    pooling_operation = [['max', 'max', 'max'], ['max', 'max', 'max'], ['max']]
    list_dim_channels2 = [[[]] * 3, [[]] * 3, [[]]]
    intermediate_size_fc = [512, 256]
    dropout_rate = [0.7, 0.7]
    nb_interpolating_points = [3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    list_dim_channels = [[256, 256], [256, 128], [128, 128]]
    num_classes = 17
    num_parts = 251
    dropout_rate_cross = 0
    nb_interpolating_points_encoding = [[8, 8], [8, 8], [None]]
    nb_interpolating_points_crossconnection = [[[8, 8], [8, 8]], [[8, 8], [8, 8]], [None]]
    model = mRes(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                 in_channel_x_complete, in_channel, list_dim_channels1, use_x, cross_connection, pooling_operation,
                 list_dim_channels2, intermediate_size_fc, dropout_rate, nb_interpolating_points, use_x_complete_unsampled,
                 list_dim_channels, num_classes, num_parts, dropout_rate_cross, nb_interpolating_points_encoding)
    model.add_cross_connection(batch_size, nb_interpolating_points_crossconnection)

    batch_size = 4
    num_points_training = 8192
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on mRes for PartNet: Success')

