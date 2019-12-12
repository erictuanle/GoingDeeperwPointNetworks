# Importation of libraries
import math
import torch
import numpy as np
from Models.model_utils import *

import pdb

##############################################
##             SLP POOLING BLOCK            ##
##############################################
class SLP_Pooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_complete, indexes, weight1, weight2, bias, use_x, pooling_operation, zerotensor, batchtensor, type_tensor='float', max_value=10**10):
        with torch.no_grad():
            batch_size, num_points, dim_features = x.size()
            _, _, dim_features_complete = x_complete.size()
            _, nb_subsampled_points, num_neighbors = indexes.size()
            output_dim_features, _ = weight1.size()
            if type(batch_size) == torch.Tensor:
                batch_size = batch_size.item()
            if type(num_points) == torch.Tensor:
                num_points = num_points.item()
            if type(dim_features) == torch.Tensor:
                dim_features = dim_features.item()
            if type(dim_features_complete) == torch.Tensor:
                dim_features_complete = dim_features_complete.item()
            if type(nb_subsampled_points) == torch.Tensor:
                nb_subsampled_points = nb_subsampled_points.item()
            if type(num_neighbors) == torch.Tensor:
                num_neighbors = num_neighbors.item()
            if type(output_dim_features) == torch.Tensor:
                output_dim_features = output_dim_features.item()
            x_complete_ = x_complete.contiguous().view(batch_size * num_points, dim_features_complete)
            x_complete_ = torch.mm(x_complete_, weight1.transpose(0, 1)) + bias
            x_complete_ = x_complete_.view(batch_size, num_points, output_dim_features)
            if use_x:
                x_ = x.view(batch_size * num_points, dim_features)
                x_ = torch.mm(x_, weight2.transpose(0, 1))
                x_ = x_.view(batch_size, num_points, output_dim_features)
                x_complete_ = x_complete_ + x_
            x_complete_ = torch.cat((zerotensor.expand(batch_size, 1, output_dim_features), x_complete_), dim=1)
            indexes_ = indexes.contiguous().view(batch_size, nb_subsampled_points * num_neighbors,1).expand(batch_size, nb_subsampled_points * num_neighbors, output_dim_features) + 1
            x_complete_ = torch.gather(x_complete_, 1, indexes_).view(batch_size, nb_subsampled_points, num_neighbors, output_dim_features)
            if use_x and (num_points == nb_subsampled_points):
                x_complete_ = x_complete_ - x_.unsqueeze(2)
            elif use_x:
                x_ = torch.cat((zerotensor.expand(batch_size, 1, output_dim_features), x_), dim=1)
                indexes__ = indexes[:,:,0].contiguous().view(batch_size, nb_subsampled_points, 1).expand(batch_size, nb_subsampled_points, output_dim_features) + 1
                x_ = torch.gather(x_, 1, indexes__).view(batch_size, nb_subsampled_points, 1, output_dim_features)
                x_complete_ = x_complete_ - x_
            if pooling_operation == 'max':
                mask = indexes_.view(batch_size, nb_subsampled_points, num_neighbors, output_dim_features) == 0
                if type_tensor == 'float':
                    mask = mask.float()
                elif type_tensor == 'double':
                    mask = mask.double()
                x_complete_ = x_complete_ - max_value*mask
                x_complete_, indexes_max = torch.max(x_complete_, dim=2)
            elif pooling_operation == 'sum':
                mask = indexes_.view(batch_size, nb_subsampled_points, num_neighbors, output_dim_features) != 0
                if type_tensor == 'float':
                    mask = mask.float()
                elif type_tensor == 'double':
                    mask = mask.double()
                x_complete_ = x_complete_ * mask
                x_complete_ = torch.sum(x_complete_, dim=2)
                indexes_max = None
            ctx.save_for_backward(x, x_complete, indexes, weight1, weight2, indexes_max, zerotensor)
            ctx.use_x = use_x
            ctx.pooling_operation = pooling_operation
            ctx.type_tensor = type_tensor
        return x_complete_
    @staticmethod
    def backward(ctx, grad_output, retain_variables=False):
        with torch.no_grad():
            x, x_complete, indexes, weight1, weight2, indexes_max, zerotensor = ctx.saved_tensors
            use_x = ctx.use_x
            pooling_operation = ctx.pooling_operation
            # Dimension
            batch_size, num_points, dim_features = x.size()
            _, _, dim_features_complete = x_complete.size()
            _, nb_subsampled_points, num_neighbors = indexes.size()
            output_dim_features, _ = weight1.size()
            if type(batch_size) == torch.Tensor:
                batch_size = batch_size.item()
            if type(num_points) == torch.Tensor:
                num_points = num_points.item()
            if type(dim_features) == torch.Tensor:
                dim_features = dim_features.item()
            if type(dim_features_complete) == torch.Tensor:
                dim_features_complete = dim_features_complete.item()
            if type(nb_subsampled_points) == torch.Tensor:
                nb_subsampled_points = nb_subsampled_points.item()
            if type(num_neighbors) == torch.Tensor:
                num_neighbors = num_neighbors.item()
            if type(output_dim_features) == torch.Tensor:
                output_dim_features = output_dim_features.item()
            # Backward in the maxpooling
            if pooling_operation == 'max':
                tensor = zerotensor.view(1, 1, 1, 1).expand(batch_size, nb_subsampled_points, num_neighbors, output_dim_features).clone()
                grad_output = tensor.scatter_(2, indexes_max.unsqueeze(2), grad_output.unsqueeze(2))
            elif pooling_operation == 'sum':
                grad_output = grad_output.unsqueeze(2).expand(batch_size, nb_subsampled_points, num_neighbors, output_dim_features).clone()
                mask = indexes.contiguous().view(batch_size, nb_subsampled_points, num_neighbors, 1).expand(batch_size, nb_subsampled_points, num_neighbors, output_dim_features) == -1
                grad_output[mask] = 0
            grad_output = grad_output.contiguous().view(batch_size, nb_subsampled_points * num_neighbors, output_dim_features)
            indexes = indexes.contiguous().view(batch_size, nb_subsampled_points * num_neighbors, 1).expand(batch_size, nb_subsampled_points * num_neighbors, output_dim_features) + 1
            # Bring the gradient down to the good shape
            value_grad_x_complete = zerotensor.view(1, 1, 1).expand(batch_size, num_points + 1, output_dim_features).clone()
            value_grad_x_complete.scatter_add_(1, indexes, grad_output)
            value_grad_x_complete = value_grad_x_complete[:,1:,:]
            if use_x:
                grad_output = torch.sum(grad_output.view(batch_size, nb_subsampled_points, num_neighbors, output_dim_features), dim=2, keepdim=False)
                if num_points == nb_subsampled_points:
                    value_grad_x = value_grad_x_complete - grad_output
                else:
                    value_grad_x = zerotensor.view(1, 1, 1).expand(batch_size, num_points + 1, output_dim_features).clone()
                    value_grad_x.scatter_(1, indexes.view(batch_size, nb_subsampled_points, num_neighbors, output_dim_features)[:,:,0,:], grad_output)
                    value_grad_x = value_grad_x[:,1:,:]
                    value_grad_x = value_grad_x_complete - value_grad_x
            # Gradient with respect to weight1
            grad_weight1 = torch.bmm(value_grad_x_complete.transpose(1,2), x_complete)
            grad_weight1 = torch.sum(grad_weight1, dim=0)
            # Gradient with respect to weight2
            if use_x:
                grad_weight2 = torch.bmm(value_grad_x.transpose(1, 2), x)
                grad_weight2 = torch.sum(grad_weight2, dim=0)
            else:
                grad_weight2 = None
            # Gradient with respect to the bias
            grad_bias = torch.sum(value_grad_x_complete, dim=[0, 1])
            # Gradient with respect to x
            if use_x:
                value_grad_x = value_grad_x.view(batch_size * num_points, output_dim_features)
                grad_x = torch.mm(value_grad_x,weight2).view(batch_size, num_points, dim_features)
            else:
                grad_x = zerotensor.view(1, 1, 1).repeat(batch_size, num_points, dim_features)
            # Gradient with respect to x_complete
            value_grad_x_complete = value_grad_x_complete.contiguous().view(batch_size * num_points, output_dim_features)
            grad_x_complete = torch.mm(value_grad_x_complete, weight1).view(batch_size, num_points, dim_features_complete)
        return grad_x, grad_x_complete, None, grad_weight1, grad_weight2, grad_bias, None, None, None, None, None
if __name__ == '__main__':
    print('Testing the gradients of the SLP_Pooling layer')
    # Building the input tensors
    device = torch.device("cpu")
    num_batch = 4
    num_points = 32
    dim_features = 3
    dim_features_complete = 6
    output_dim_features = 8
    x = torch.autograd.Variable(torch.randn(num_batch, num_points, dim_features)).double().to(device)
    x.requires_grad = True
    x_complete = torch.autograd.Variable(torch.randn(num_batch, num_points, dim_features_complete)).double().to(device)
    x_complete.requires_grad = True
    weight1 = torch.nn.Parameter(torch.randn(output_dim_features, dim_features_complete), requires_grad=True).double().to(device)
    weight2 = torch.nn.Parameter(torch.randn(output_dim_features, dim_features), requires_grad=True).double().to(device)
    bias = torch.nn.Parameter(torch.randn(output_dim_features), requires_grad=True).double().to(device)
    use_x = True
    pooling_operation = 'sum'
    zerotensor = torch.zeros([1]).double().to(device)
    batchtensor = torch.arange(num_batch).to(device)
    type_tensor = 'double'
    # Sampling
    model = Sampling(num_points // 2)
    indexes_downsampling = model(x.float())
    x_ = model.extract_values(x.float(), indexes_downsampling)
    # Grouping
    nb_neighbours = 2
    sampling_method = 'query_ball'
    patch_radius = 0.2
    num_points_training = 10000
    model = Grouping(nb_neighbours, sampling_method, use_x, num_batch, patch_radius=patch_radius)
    indexes = model(x.float(), x_)
    # Test of forward/backward pass
    try:
        slp_pooling = SLP_Pooling.apply
        torch.autograd.gradcheck(slp_pooling, (x, x_complete, indexes, weight1, weight2, bias, use_x, pooling_operation, zerotensor, batchtensor, type_tensor), raise_exception=True)
        print('Test on SLP_Pooling: Success')
    except:
        print('Test on SLP_Pooling: Failure')
        raise

class SLP_Pooling_Module(torch.nn.Module):
    def __init__(self):
        super(SLP_Pooling_Module, self).__init__()
        self.type_tensor = 'float' 
    def forward(self, x, x_complete, indexes, weight1, weight2, bias, use_x, pooling_operation, zerotensor, batchtensor, max_value=10**10):
        x_complete_ = SLP_Pooling.apply(x, x_complete, indexes, weight1, weight2, bias, use_x, pooling_operation, zerotensor, batchtensor, self.type_tensor)
        return x_complete_

##############################################
##              TRANSITION BLOCK            ##
##############################################
class TransitionBlock(torch.nn.Module):
    def __init__(self, nb_scales, input_channels, list_dim_slp):
        super(TransitionBlock, self).__init__()
        self.nb_scales = nb_scales
        self.list_dim_slp = list_dim_slp
        self.input_channels = input_channels
        self.list_slp = torch.nn.ModuleList()
        self.list_bn = torch.nn.ModuleList()
        for i in range(self.nb_scales):
            last_channel = input_channels[i]
            for out_channel in self.list_dim_slp[i]:
                self.list_slp.append(torch.nn.Conv1d(last_channel, out_channel, 1, bias=True).float())
                self.list_bn.append(torch.nn.BatchNorm1d(out_channel).float())
                last_channel = out_channel
    def update_decay(self, bn_decay_value):
        for i in range(len(self.list_bn)):
            self.list_bn[i].momentum = bn_decay_value
    def forward(self, list_x_complete, bn_decay_value=None):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        cpt = 0
        list_x_complete_ = []
        for i in range(self.nb_scales):
            x_complete = list_x_complete[i].transpose(1, 2)
            for j in range(len(self.list_dim_slp[i])):
                x_complete = self.list_slp[cpt](x_complete)
                x_complete = self.list_bn[cpt](x_complete)
                x_complete = torch.nn.functional.relu(x_complete)
                cpt += 1
            x_complete = x_complete.transpose(1, 2)
            list_x_complete_.append(x_complete)
        return list_x_complete_

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, nb_scales, batch_size, spatial_channels, input_channels, patch_radius, nb_neighbours, sampling_method, pooling_operation,
                       require_grouping, use_x, list_dim_slp, use_crosslinks, use_reslinks):
        super(ConvolutionBlock, self).__init__()
        self.nb_scales = nb_scales
        self.batch_size = batch_size
        self.spatial_channels = spatial_channels
        self.input_channels = input_channels
        self.patch_radius = patch_radius
        self.nb_neighbours = nb_neighbours
        self.sampling_method = sampling_method
        self.pooling_operation = pooling_operation
        self.require_grouping = require_grouping
        self.use_x = use_x
        self.list_dim_slp = list_dim_slp
        self.use_crosslinks = use_crosslinks
        self.use_reslinks = use_reslinks
        # Grouping if needed
        self.groupinglist = torch.nn.ModuleList()
        for i in range(self.nb_scales):
            self.groupinglist.append(Grouping(self.nb_neighbours[i], self.sampling_method[i], self.use_x[i], self.batch_size, patch_radius=self.patch_radius[i]))
        # Single Layer Perceptron
        self.slppool = torch.nn.ModuleList()
        self.slppool_weight1 = torch.nn.ParameterList()
        self.slppool_weight2 = torch.nn.ParameterList()
        self.slppool_bias = torch.nn.ParameterList()
        self.bnlist = torch.nn.ModuleList()
        for i in range(self.nb_scales):
            self.slppool.append(SLP_Pooling_Module())
            weight1 = torch.nn.Parameter(torch.zeros(self.list_dim_slp[i], self.input_channels[i]).float(), requires_grad=True)
            self.slppool_weight1.append(weight1)
            if self.use_x[i]:
                weight2 = torch.nn.Parameter(torch.zeros(self.list_dim_slp[i], self.spatial_channels).float(), requires_grad=True)
                self.slppool_weight2.append(weight2)
            else:
                self.slppool_weight2.append(None)
            bias = torch.nn.Parameter(torch.zeros(self.list_dim_slp[i]).float(), requires_grad=True)
            self.slppool_bias.append(bias)
            self.bnlist.append(torch.nn.BatchNorm1d(self.list_dim_slp[i]).float())
        # Cross-Connection
        if self.use_crosslinks:
            self.slp_crossconnection_downsampling = torch.nn.ModuleList()
            self.bn_crossconnection_downsampling = torch.nn.ModuleList()
            self.slp_crossconnection_upsampling = torch.nn.ModuleList()
            self.bn_crossconnection_upsampling = torch.nn.ModuleList()
            for i in range(self.nb_scales):
                if i < self.nb_scales - 1:
                    self.slp_crossconnection_downsampling.append(torch.nn.Conv1d(self.list_dim_slp[i], self.list_dim_slp[i + 1], 1, bias=True).float())
                    self.bn_crossconnection_downsampling.append(torch.nn.BatchNorm1d(self.list_dim_slp[i + 1]).float())
                if i>0:
                    self.slp_crossconnection_upsampling.append(torch.nn.Conv1d(self.list_dim_slp[i], self.list_dim_slp[i - 1], 1, bias=True).float())
                    self.bn_crossconnection_upsampling.append(torch.nn.BatchNorm1d(self.list_dim_slp[i - 1]).float())
        # Utils
        self.list_indexes_grouping = None
        self.zerotensor = torch.nn.Parameter(torch.zeros([1]).float(), requires_grad=False)
        self.batchtensor = torch.nn.Parameter(torch.arange(self.batch_size), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        for i in range(self.nb_scales):
            # Initializing weight1
            torch.nn.init.kaiming_uniform_(self.slppool_weight1[i], a=math.sqrt(5))
            if self.use_x[i]:
                # Initializing weight2
                torch.nn.init.kaiming_uniform_(self.slppool_weight2[i], a=math.sqrt(5))
            # Initializing the bias
            bound = 1 / math.sqrt(self.slppool_bias[i].size(0))
            torch.nn.init.uniform_(self.slppool_bias[i], -bound, bound)
    def update_decay(self, bn_decay_value):
        if self.use_crosslinks:
            for i in range(self.nb_scales - 1):
                self.bn_crossconnection_downsampling[i].momentum = bn_decay_value
                self.bn_crossconnection_upsampling[i].momentum = bn_decay_value
    def forward(self, list_x, list_x_complete, list_indexes_downsampling, list_indexes_upsampling, list_weights_upsampling,
                      list_indexes_grouping=None, bn_decay_value=None):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        batch_size, _, _ = list_x[0].size()
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        batchtensor = self.batchtensor[:batch_size]
        if self.require_grouping:
            list_indexes_grouping = []
        list_x_complete_ = []
        for i in range(self.nb_scales):
            # Grouping
            if self.require_grouping:
                indexes = self.groupinglist[i](list_x[i], list_x[i])
                # Saving the list of indexes
                list_indexes_grouping.append(indexes)
            else:
                indexes = list_indexes_grouping[i]
            # Convolution: SLP+Pooling
            x_complete_ = self.slppool[i](list_x[i], list_x_complete[i], indexes, self.slppool_weight1[i], self.slppool_weight2[i],
                                          self.slppool_bias[i], self.use_x[i], self.pooling_operation[i], self.zerotensor, batchtensor)
            x_complete_ = x_complete_.transpose(1, 2)
            x_complete_ = self.bnlist[i](x_complete_).transpose(1, 2)
            x_complete_ = torch.nn.functional.relu(x_complete_)
            list_x_complete_.append(x_complete_)
        self.list_indexes_grouping = list_indexes_grouping
        # Cross-Connection
        offset = 0
        offset_sampling = 0
        if len(list_indexes_downsampling) > self.nb_scales:
            offset_sampling += 1
        if self.use_crosslinks:
            for i in range(self.nb_scales):
                x_complete = list_x_complete_[i]
                if i < self.nb_scales-1:
                    x_complete_ = self.slp_crossconnection_upsampling[i](list_x_complete_[i + 1].transpose(1, 2))
                    x_complete_ = self.bn_crossconnection_upsampling[i](x_complete_).transpose(1, 2)
                    x_complete_ = upsampling(x_complete_, list_indexes_upsampling[i], list_weights_upsampling[i])
                    x_complete = x_complete + torch.nn.functional.relu(x_complete_)
                if i > 0:
                    x_complete_ = downsampling(list_x_complete_[i - 1], list_indexes_downsampling[i - 1 + offset_sampling])
                    self.list_x_complete_xlinks_downsampled_backward.append(x_complete_)
                    x_complete_ = self.slp_crossconnection_downsampling[i - 1](x_complete_.transpose(1, 2))
                    x_complete_ = self.bn_crossconnection_downsampling[i - 1](x_complete_).transpose(1, 2)
                    x_complete = x_complete + torch.nn.functional.relu(x_complete_)
                list_x_complete_.append(x_complete)
            offset += self.nb_scales
        # Residual connection
        if self.use_reslinks:
            for i in range(self.nb_scales):
                if list_x_complete[i].size(1) > list_x_complete_[i+offset].size(1):
                    list_x_complete_.append(list_x_complete_[i + offset] + downsampling(list_x_complete[i], list_indexes_downsampling[i]))
                else:
                    list_x_complete_.append(list_x_complete_[i + offset] + list_x_complete[i])
            offset += self.nb_scales
        return list_x_complete_[offset:]

##############################################
##         LOCAL ENCODING BLOCK             ##
##############################################
class PointNetSetAbstraction(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius, in_channel_x_complete,
                 in_channel_x, list_dim_channels, pooling_operation, use_x, nb_interpolating_points, use_crosslinks,
                 use_reslinks, sequence, blockout_rate, test=False):
        # Rq: nb_subsampled_points is now a list
        super(PointNetSetAbstraction, self).__init__()
        self.batch_size = batch_size
        self.nb_subsampled_points = nb_subsampled_points
        self.nb_scales = len(self.nb_subsampled_points)
        self.nb_neighbours = nb_neighbours
        self.sampling_method = sampling_method
        self.patch_radius = patch_radius
        self.in_channel_x_complete = in_channel_x_complete
        self.in_channel_x = in_channel_x
        self.list_dim_channels = list_dim_channels
        self.pooling_operation = pooling_operation
        self.use_x = use_x
        self.length_conv1_bank = []
        self.length_conv2_bank = []
        self.last_channel = 0
        self.nb_interpolating_points = nb_interpolating_points
        self.sequence = sequence
        self.blockout_rate = blockout_rate
        self.test = test
        flag = self.in_channel_x_complete is None
        if flag:
            self.in_channel_x_complete = self.in_channel_x
        self.samplinglist = torch.nn.ModuleList()
        if 'CS' in self.sequence:
            self.groupinglist = torch.nn.ModuleList()
        self.interpolatinglist = torch.nn.ModuleList()
        for i in range(self.nb_scales):
            self.samplinglist.append(Sampling(self.nb_subsampled_points[i]))
            if 'CS' in self.sequence:
                self.groupinglist.append(Grouping(self.nb_neighbours[i], self.sampling_method[i], self.use_x[i], self.batch_size, patch_radius=self.patch_radius[i]))
            self.interpolatinglist.append(Interpolating(self.nb_interpolating_points[i], False, self.batch_size))
        self.transitionBlocks = torch.nn.ModuleList()
        self.convolutionBlocks = torch.nn.ModuleList()
        self.in_channels = [[self.in_channel_x_complete] + self.list_dim_channels[i][:-1] for i in range(0, self.nb_scales)]
        # Stacking blocks together
        offset = 0
        cpt = 0
        require_grouping = True
        for j in range(len(self.list_dim_channels[0])):
            if self.sequence[cpt] == 'S':
                offset += 1
                cpt += 1
                require_grouping = True
            if np.any([self.list_dim_channels[k][j] != self.in_channels[k][j] for k in range(self.nb_scales)]):
                # We need to add a transition block
                self.transitionBlocks.append(
                    TransitionBlock(self.nb_scales, [self.in_channels[k][j] for k in range(self.nb_scales)],
                                    [[self.list_dim_channels[k][j]] for k in range(self.nb_scales)]))
                self.sequence = self.sequence[:cpt] + ['T'] + self.sequence[cpt:]
                cpt += 1
            if self.sequence[cpt] == 'CS':
                require_grouping = False
                self.convolutionBlocks.append(ConvolutionBlock(self.nb_scales, self.batch_size, self.in_channel_x,
                                                               [self.list_dim_channels[k][j] for k in range(self.nb_scales)],
                                                               self.patch_radius[offset:self.nb_scales + offset],
                                                               self.nb_neighbours[offset:self.nb_scales + offset],
                                                               self.sampling_method[offset:self.nb_scales + offset],
                                                               self.pooling_operation[offset:self.nb_scales + offset],
                                                               require_grouping, self.use_x[offset:self.nb_scales + offset],
                                                               [self.list_dim_channels[k][j] for k in range(self.nb_scales)],
                                                               use_crosslinks, use_reslinks))
                offset += 1
                require_grouping = True
            elif self.sequence[cpt]=='C':
                self.convolutionBlocks.append(ConvolutionBlock(self.nb_scales, self.batch_size, self.in_channel_x,
                                                               [self.list_dim_channels[k][j] for k in range(self.nb_scales)],
                                                               self.patch_radius[offset:self.nb_scales + offset],
                                                               self.nb_neighbours[offset:self.nb_scales + offset],
                                                               self.sampling_method[offset:self.nb_scales + offset],
                                                               self.pooling_operation[offset:self.nb_scales + offset],
                                                               require_grouping, self.use_x[offset:self.nb_scales + offset],
                                                               [self.list_dim_channels[k][j] for k in range(self.nb_scales)],
                                                               use_crosslinks, use_reslinks))
                require_grouping = False
            cpt += 1
        self.last_channel = [self.list_dim_channels[k][len(self.list_dim_channels[0]) - 1] for k in range(self.nb_scales)]
    def train_custom(self):
        self.test = False
    def eval_custom(self):
        self.test = True
    def forward(self, x, x_complete, bn_decay_value=None):
        list_subsampled_centroids = [x]
        list_indexes_downsampling = []
        if x_complete is not None:
            list_x_complete = [x_complete]
        else:
            list_x_complete = [x]
        # Loop for sampling
        for i in range(self.nb_scales):
            indexes_subsampled_centroids = self.samplinglist[i](list_subsampled_centroids[i])
            subsampled_centroids = self.samplinglist[i].extract_values(list_subsampled_centroids[i], indexes_subsampled_centroids)
            # Saving the intermediate layers before the conv layers
            list_indexes_downsampling.append(indexes_subsampled_centroids)
            list_subsampled_centroids.append(subsampled_centroids)
            if i < self.nb_scales - 1:
                x_complete = self.samplinglist[i].extract_values(list_x_complete[i], indexes_subsampled_centroids)
                list_x_complete.append(x_complete)
        # Loop for interpolating
        list_indexes_upsampling = []
        list_weights_upsampling = []
        for i in range(self.nb_scales):
            indexes, weights = self.interpolatinglist[i](list_subsampled_centroids[i], list_subsampled_centroids[i + 1])
            list_indexes_upsampling.append(indexes)
            list_weights_upsampling.append(weights.detach())
        # Stacking the blocks
        cpt_T = 0
        cpt_C = 0
        offset = 0
        offset_list = 0
        list_indexes_grouping = None
        require_grouping = True
        for seq in self.sequence:
            if seq == 'T':
                #print('->Transition Block')
                list_x_complete.extend(self.transitionBlocks[cpt_T]([list_x_complete[offset_list + i] for i in range(self.nb_scales)], bn_decay_value=bn_decay_value))
                offset_list += self.nb_scales
                #print(self.transitionBlocks[cpt_T].list_dim_slp)
                cpt_T += 1
            elif seq == 'C':
                #print('->Convolution Block')
                pass_through = np.random.rand() >= self.blockout_rate[cpt_C]
                if (pass_through or self.test) and (self.blockout_rate[cpt_C]<1):
                    factor = 1
                    if self.test:
                        factor = 1/(1-self.blockout_rate[cpt_C])
                    if require_grouping:
                        self.convolutionBlocks[cpt_C].require_grouping = True
                    list_x_complete.extend(self.convolutionBlocks[cpt_C](
                        list_subsampled_centroids[offset:self.nb_scales + offset],[list_x_complete[offset_list + i] for i in range(self.nb_scales)],
                        list_indexes_downsampling[offset:self.nb_scales - 1 + offset],
                        list_indexes_upsampling[offset:self.nb_scales - 1 + offset],
                        list_weights_upsampling[offset:self.nb_scales - 1 + offset],
                        list_indexes_grouping, bn_decay_value=bn_decay_value))
                    list_indexes_grouping = self.convolutionBlocks[cpt_C].list_indexes_grouping
                    #print(self.convolutionBlocks[cpt_C].list_dim_slp)
                    offset_list += self.nb_scales
                    require_grouping = False
                #else:
                #    print('--> Block dropout')
                cpt_C += 1
            elif seq=='CS':
                #print('->Convolution Block')
                list_indexes_grouping = []
                for i in range(self.nb_scales):
                    list_indexes_grouping.append(self.groupinglist[i](list_subsampled_centroids[i + offset], list_subsampled_centroids[i + 1 + offset]))
                # Grouping Step
                list_x_complete.extend(self.convolutionBlocks[cpt_C](list_subsampled_centroids[offset:self.nb_scales + offset],[list_x_complete[offset_list+i] for i in range(self.nb_scales)],
                                                                     list_indexes_downsampling[offset:self.nb_scales - 1 + offset + 1],
                                                                     list_indexes_upsampling[offset + 1:self.nb_scales - 1 + offset + 1],
                                                                     list_weights_upsampling[offset + 1:self.nb_scales - 1 + offset + 1],
                                                                     list_indexes_grouping, bn_decay_value=bn_decay_value))
                list_indexes_grouping = None
                #print(self.convolutionBlocks[cpt_C].list_dim_slp)
                offset_list += self.nb_scales
                offset += 1
                cpt_C += 1
                require_grouping = True
            elif seq == 'S':
                for i in range(self.nb_scales):
                    list_x_complete.append(downsampling(list_x_complete[offset_list + i], list_indexes_downsampling[i]))
                offset_list += self.nb_scales
                #print('->Sampling')
                list_indexes_grouping = None
                offset += 1
                require_grouping = True
        # Interpolating back the points
        for i in range(1, self.nb_scales):
            x_complete = list_x_complete[offset_list + i]
            for j in range(i, 0, -1):
                x_complete = upsampling(x_complete, list_indexes_upsampling[j], list_weights_upsampling[j])
            list_x_complete.append(x_complete)
        x_complete = torch.cat([list_x_complete[offset_list + self.nb_scales + i] for i in range(self.nb_scales - 1)], dim=2)
        x_complete = torch.cat((list_x_complete[offset_list], x_complete), dim=2)
        return list_subsampled_centroids[1], x_complete

##############################################
##         GLOBAL ENCODING BLOCK            ##
##############################################
class PointNetSetCollapse(torch.nn.Module):
    def __init__(self, num_points, use_x,in_channels, in_channels_complete, list_dim_slp, residuallinks_input,
                 residuallinks_output, pooling_operation):
        super(PointNetSetCollapse, self).__init__()
        self.num_points = num_points
        self.use_x = use_x
        self.residuallinks_input = residuallinks_input
        self.residuallinks_output = residuallinks_output
        self.pooling_operation = pooling_operation
        self.grouping = Grouping_all(self.num_points, self.use_x)
        self.list_dim_slp = list_dim_slp
        self.list_slp = torch.nn.ModuleList()
        self.list_bn = torch.nn.ModuleList()
        last_channel = in_channels_complete
        if self.use_x:
            last_channel += in_channels
        for output in self.list_dim_slp:
            self.list_slp.append(torch.nn.Conv1d(last_channel, output, 1, bias=True))
            self.list_bn.append(torch.nn.BatchNorm1d(output))
            last_channel = output
        self.last_channel = last_channel
    def update_decay(self, bn_decay_value):
        for i in range(len(self.list_bn)):
            self.list_bn[i].momentum = bn_decay_value
    def forward(self, x, x_complete, bn_decay_value):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        subsampled_centroids, indexes = self.grouping(x)
        x_complete = self.grouping.extract_values(x, x_complete)
        x_complete = x_complete.squeeze(1).transpose(1, 2)
        list_x_complete = [x_complete]
        offset = 0
        for i in range(len(self.list_dim_slp)):
            x_complete = self.list_slp[i](list_x_complete[i + offset])
            x_complete = self.list_bn[i](x_complete)
            list_x_complete.append(torch.nn.functional.relu(x_complete))
            index = np.where([elt == i for elt in self.residuallinks_output])[0]
            if len(index) == 1:
                list_x_complete.append(list_x_complete[i + 1 + offset] + list_x_complete[self.residuallinks_input[index[0]] + offset])
                offset += 1
        x_complete = list_x_complete[len(self.list_dim_slp) + offset].transpose(1, 2)
        if self.pooling_operation == 'max':
            x_complete, indexes_max = torch.max(x_complete, dim=1)
        elif self.pooling_operation == 'sum':
            x_complete = torch.sum(x_complete, dim=1)
            indexes_max = None
        self.indexes_max_backward = indexes_max
        x_complete = x_complete.unsqueeze(1)
        return subsampled_centroids, x_complete

##############################################
##               MAIN INTERFACE             ##
##############################################
class convPN(torch.nn.Module):
    def __init__(self, batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                 in_channel_x_complete, in_channel, list_dim_channels_encoding, use_x, use_crosslinks, use_reslinks,
                 sequence, pooling_operation, residuallinks_input, residuallinks_output, intermediate_size_fc, 
                 dropout_rate, nb_interpolating_points, use_x_complete_unsampled, list_dim_channels_decoding, num_classes,
                 num_parts, blockout_rate, test=False):
        super(convPN, self).__init__()
        self.pointnet_downsampling1 = PointNetSetAbstraction(batch_size, nb_subsampled_points[0], nb_neighbours[0],
                                                             sampling_method[0], patch_radius[0], in_channel_x_complete, in_channel,
                                                             list_dim_channels_encoding[0], pooling_operation[0], use_x[0],
                                                             nb_interpolating_points[0], use_crosslinks[0], use_reslinks[0],
                                                             sequence[0], blockout_rate[0], test=test)
        in_channel_x_complete1 = sum(self.pointnet_downsampling1.last_channel)
        self.pointnet_downsampling2 = PointNetSetAbstraction(batch_size, nb_subsampled_points[1], nb_neighbours[1],
                                                             sampling_method[1], patch_radius[1], in_channel_x_complete1, in_channel,
                                                             list_dim_channels_encoding[1], pooling_operation[1], use_x[1],
                                                             nb_interpolating_points[1], use_crosslinks[1], use_reslinks[1],
                                                             sequence[1], blockout_rate[1], test=test)
        in_channel_x_complete2 = sum(self.pointnet_downsampling2.last_channel)
        self.pointnet_downsampling3 = PointNetSetCollapse(nb_subsampled_points[1][0], use_x[2], in_channel, in_channel_x_complete2,
                                                          list_dim_channels_encoding[2], residuallinks_input, residuallinks_output,
                                                          pooling_operation[2])
        in_channel_x_complete3 = self.pointnet_downsampling3.last_channel

        self.pointnet_upsampling1 = PointNetFeaturePropagation(nb_interpolating_points[3], use_x_complete_unsampled[0],
                                                               in_channel_x_complete3, in_channel_x_complete2,
                                                               list_dim_channels_decoding[0], batch_size)
        in_channel_x_complete4 = self.pointnet_upsampling1.last_channel
        self.pointnet_upsampling2 = PointNetFeaturePropagation(nb_interpolating_points[4], use_x_complete_unsampled[1],
                                                               in_channel_x_complete4, in_channel_x_complete1,
                                                               list_dim_channels_decoding[1], batch_size)
        in_channel_x_complete5 = self.pointnet_upsampling2.last_channel
        if in_channel_x_complete is None:
            in_channel_x_complete = 0
        if num_classes is None:
            num_classes = 0
        self.pointnet_upsampling3 = PointNetFeaturePropagation(nb_interpolating_points[5], use_x_complete_unsampled[2],
                                                               in_channel_x_complete5, in_channel + in_channel_x_complete + num_classes,
                                                               list_dim_channels_decoding[2], batch_size)
        in_channel_x_complete6 = self.pointnet_upsampling3.last_channel
        self.conv1d1 = torch.nn.Conv1d(in_channel_x_complete6, intermediate_size_fc[0], 1, bias=True).float()
        self.bn1 = torch.nn.BatchNorm1d(intermediate_size_fc[0]).float()
        self.dropout1 = torch.nn.Dropout(p=dropout_rate[0]).float()
        self.conv1d2 = torch.nn.Conv1d(intermediate_size_fc[0],num_parts,1,bias=True).float()
        if num_classes > 0:
            self.one_hot = torch.nn.Parameter(torch.FloatTensor(1,num_classes).zero_(), requires_grad=False)
    def train_custom(self):
        self.pointnet_downsampling1.train_custom()
        self.pointnet_downsampling2.train_custom()
    def eval_custom(self):
        self.pointnet_downsampling1.eval_custom()
        self.pointnet_downsampling2.eval_custom()
    def update_decay(self,bn_decay_value):
        self.bn1.momentum = bn_decay_value
    def forward(self, x, x_complete, labels_cat, bn_decay_value=0.99):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        num_batch, num_points, _ = x.size()
        if type(num_batch) == torch.Tensor:
            num_batch = num_batch.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        # Abstraction layers
        #print('----Encoding 1----')
        subsampled_centroids1, x1 = self.pointnet_downsampling1(x, x_complete, bn_decay_value=bn_decay_value)
        #print('----Encoding 2----')
        subsampled_centroids2, x2 = self.pointnet_downsampling2(subsampled_centroids1, x1, bn_decay_value=bn_decay_value)
        #print('----Encoding 3----')
        subsampled_centroids3, x3 = self.pointnet_downsampling3(subsampled_centroids2, x2, bn_decay_value=bn_decay_value)
        # Feature Propagation layers
        x4 = self.pointnet_upsampling1(subsampled_centroids2, subsampled_centroids3, x2, x3, bn_decay_value)
        x5 = self.pointnet_upsampling2(subsampled_centroids1, subsampled_centroids2, x1, x4, bn_decay_value)
        if labels_cat is not None:
            target = self.one_hot.repeat(num_batch, 1)
            labels_cat = labels_cat.view(num_batch, 1)
            target = target.scatter_(1, labels_cat.data, 1).unsqueeze(1).repeat(1, num_points, 1)
            x0 = torch.cat([target, x], dim=2)
        else:
            x0 = x
        if x_complete is not None:
            x0 = torch.cat([x0, x_complete], dim=2)
        x6 = self.pointnet_upsampling3(x, subsampled_centroids1, x0, x5, bn_decay_value)
        # Fully connected layers
        x = x6.transpose(1, 2)
        x = self.conv1d1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.conv1d2(x)
        x = x.transpose(1, 2)
        return x

if __name__=='__main__':
    print('Testing the whole network for ShapeNet-Part')
    batch_size = 8
    nb_subsampled_points = [[512, 256, 128], [128, 96, 64]]
    nb_neighbours = [[32, 32, 32, 32], [64, 64, 64, 64]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                       ['query_ball', 'query_ball', 'query_ball', 'query_ball']]
    patch_radius = [[0.1, 0.2, 0.4, 0.4], [0.2, 0.4, 0.8, 0.8]]
    in_channel_x_complete = 3
    in_channel = 3
    list_dim_channels_encoding = [[[32, 32, 32, 32, 64, 64], [64, 64, 64, 64, 128, 128], [64, 64, 96, 96, 128, 128]],
                                  [[64, 64, 64, 64, 128, 128], [128, 128, 128, 128, 256, 256], [128, 128, 128, 128, 256, 256]],
                                  [256, 128, 256, 512, 128, 512, 1024, 128, 1024]]
    use_x = [[True, True, True, True], [True, True, True, True], True]
    use_crosslinks = [False, False]
    use_reslinks = [True, True]
    sequence = [['C', 'C', 'C', 'S', 'C', 'C', 'C'],
                ['C', 'C', 'C', 'S', 'C', 'C', 'C']]
    pooling_operation = [['max', 'max', 'max', 'max'], ['max', 'max', 'max', 'max'], 'max']
    residuallinks_input = [1, 4, 7]
    residuallinks_output = [2, 5, 8]
    nb_interpolating_points = [[8, 8, 8], [8, 8, 8], None, 3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    intermediate_size_fc = [512,256]
    blockout_rate = [[0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
    dropout_rate = [0.7,0.7]
    list_dim_channels_decoding = [[256,256],[256,128],[128,128]]
    num_classes = 16
    num_parts = 50
    num_points_training = 2048
    test = False
    model = convPN(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                   in_channel_x_complete, in_channel, list_dim_channels_encoding, use_x, use_crosslinks, use_reslinks,
                   sequence, pooling_operation, residuallinks_input, residuallinks_output, intermediate_size_fc, dropout_rate,
                   nb_interpolating_points, use_x_complete_unsampled, list_dim_channels_decoding, num_classes, num_parts,
                   blockout_rate, test=False)    
    batch_size = 4
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = torch.randn(batch_size, num_points_training, in_channel_x_complete)
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on convPN for ShapeNet-Part: Success')

if __name__=='__main__':
    print('Testing the whole network for ScanNet')
    batch_size = 8
    nb_subsampled_points = [[512, 256, 128], [128, 96, 64]]
    nb_neighbours = [[32, 32, 32, 32], [64, 64, 64, 64]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                       ['query_ball', 'query_ball', 'query_ball', 'query_ball']]
    patch_radius = [[0.1, 0.2, 0.4, 0.4], [0.2, 0.4, 0.8, 0.8]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels_encoding = [[[32, 32, 32, 32, 64, 64], [64, 64, 64, 64, 128, 128], [64, 64, 96, 96, 128, 128]],
                                   [[64, 64, 64, 64, 128, 128], [128, 128, 128, 128, 256, 256], [128, 128, 128, 128, 256, 256]],
                                   [256, 128, 256, 512, 128, 512, 1024, 128, 1024]]
    use_x = [[True, True, True, True], [True, True, True, True], True]
    use_crosslinks = [False, False]
    use_reslinks = [True, True]
    sequence = [['CS', 'C', 'C', 'C', 'C', 'C'],
                ['C', 'C', 'C', 'S', 'C', 'C', 'C']]
    pooling_operation = [['max', 'max', 'max', 'max'], ['max', 'max', 'max', 'max'], 'max']
    residuallinks_input = [1, 4, 7]
    residuallinks_output = [2, 5, 8]
    nb_interpolating_points = [[8, 8, 8], [8, 8, 8], None, 3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    intermediate_size_fc = [512, 256]
    blockout_rate = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    dropout_rate = [0.7, 0.7]
    list_dim_channels_decoding = [[256, 256], [256, 128], [128, 128]]
    num_classes = None
    num_parts = 21
    num_points_training = 8192
    test = False
    model = convPN(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                   in_channel_x_complete, in_channel, list_dim_channels_encoding, use_x, use_crosslinks, use_reslinks,
                   sequence, pooling_operation, residuallinks_input, residuallinks_output, intermediate_size_fc,
                   dropout_rate, nb_interpolating_points, use_x_complete_unsampled, list_dim_channels_decoding, num_classes,
                   num_parts, blockout_rate, test=False)
    batch_size = 4
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = None
    output = model(x, x_complete, labels_cat)
    print('Test on convPN for ScanNet: Success')

if __name__=='__main__':
    print('Testing the whole network for PartNet')
    batch_size = 8
    nb_subsampled_points = [[512, 256, 128], [128, 79, 56]]
    nb_neighbours = [[32, 32, 32, 32], [64, 64, 64, 64]]
    sampling_method = [['query_ball', 'query_ball', 'query_ball', 'query_ball'],
                       ['query_ball', 'query_ball', 'query_ball', 'query_ball']]
    patch_radius = [[0.1, 0.2, 0.4, 0.4], [0.2, 0.4, 0.8, 0.8]]
    in_channel_x_complete = None
    in_channel = 3
    list_dim_channels_encoding = [[[32, 32, 32, 32, 64, 64], [64, 64, 64, 64, 128, 128], [64, 64, 96, 96, 128, 128]],
                                   [[64, 64, 64, 64, 128, 128], [128, 128, 128, 128, 256, 256], [128, 128, 128, 128, 256, 256]],
                                   [256, 128, 256, 512, 128, 512, 1024, 128, 1024]]
    use_x = [[True, True, True, True], [True, True, True, True], True]
    use_crosslinks = [False, False]
    use_reslinks = [True, True]
    sequence = [['CS', 'C', 'C', 'C', 'C', 'C'],
                ['C', 'C', 'C', 'S', 'C', 'C', 'C']]
    pooling_operation = [['max', 'max', 'max', 'max'], ['max', 'max', 'max', 'max'], 'max']
    residuallinks_input = [1, 4, 7]
    residuallinks_output = [2, 5, 8]
    nb_interpolating_points = [[8, 8, 8], [8, 8, 8], None, 3, 3, 3]
    use_x_complete_unsampled = [True, True, True]
    intermediate_size_fc = [512, 256]
    blockout_rate = [[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]
    dropout_rate = [0.7, 0.7]
    list_dim_channels_decoding = [[256, 256], [256, 128], [128, 128]] 
    num_classes = 17
    num_parts = 251
    num_points_training = 10000
    test = False
    model = convPN(batch_size, nb_subsampled_points, nb_neighbours, sampling_method, patch_radius,
                   in_channel_x_complete, in_channel, list_dim_channels_encoding, use_x, use_crosslinks, use_reslinks,
                   sequence, pooling_operation, residuallinks_input, residuallinks_output, intermediate_size_fc,
                   dropout_rate, nb_interpolating_points, use_x_complete_unsampled, list_dim_channels_decoding, num_classes,
                   num_parts, blockout_rate, test=False)    
    batch_size = 4
    x = torch.randn(batch_size, num_points_training, in_channel)
    x_complete = None
    labels_cat = torch.randint(0, num_classes, [batch_size, 1])
    output = model(x, x_complete, labels_cat)
    print('Test on convPN for PartNet: Success')