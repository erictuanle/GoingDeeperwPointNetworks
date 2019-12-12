# Importation of libraries
import torch
import numpy as np

import pdb

def square_distance(matrix1, matrix2):
    '''
    ||x-y||**2=||x||**2+||y||**2-2x.y
    '''
    _, num_points1, _ = matrix1.size()
    _, num_points2, _ = matrix2.size()
    if type(num_points1) == torch.Tensor:
        num_points1 = num_points1.item()
    if type(num_points2) == torch.Tensor:
        num_points2 = num_points2.item()
    term1 = torch.sum(matrix1**2, dim=2, keepdim=True).repeat(1, 1, num_points2)
    term2 = torch.sum(matrix2**2,dim=2,keepdim=True).transpose(1, 2).repeat(1, num_points1, 1)
    term3 = torch.matmul(matrix1, matrix2.transpose(1, 2))
    distance = term1 + term2 - 2 * term3
    distance = torch.clamp(distance, min=0)
    distance = torch.sqrt(distance)
    return distance

def downsampling(x, indexes):
    _, _, feature_dimension = x.size()
    if type(feature_dimension) == torch.Tensor:
        feature_dimension = feature_dimension.item()
    x = torch.gather(x, 1, indexes.unsqueeze(2).repeat(1, 1, feature_dimension))
    return x

def upsampling(x, indexes, weights):
    batch_size, _, feature_dimension = x.size()
    _, targeted_num_points, nb_interpolating_points = indexes.size()
    if type(batch_size) == torch.Tensor:
        batch_size = batch_size.item()
    if type(feature_dimension) == torch.Tensor:
        feature_dimension = feature_dimension.item()
    if type(targeted_num_points) == torch.Tensor:
        targeted_num_points = targeted_num_points.item()
    if type(nb_interpolating_points) == torch.Tensor:
        nb_interpolating_points = nb_interpolating_points.item()
    indexes = indexes.contiguous().view(batch_size, targeted_num_points * nb_interpolating_points, 1)
    x = torch.gather(x, 1, indexes.repeat(1, 1, feature_dimension)).view(batch_size, targeted_num_points, nb_interpolating_points, feature_dimension)
    x = x * weights
    x = torch.sum(x, dim=2)
    return x

##############################################
##               SAMPLING BLOCK             ##
##############################################
class Sampling(torch.nn.Module):
    def __init__(self, nb_subsampled_points):
        super(Sampling, self).__init__()
        self.nb_subsampled_points = nb_subsampled_points
        self.min_distances = torch.nn.Parameter(np.inf * torch.ones(1,1), requires_grad=False)
        self.first_index = torch.nn.Parameter(torch.LongTensor(1), requires_grad=False)
    def extract_values(self, x, indexes):
        _, num_points, dim_features = x.size()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(dim_features) == torch.Tensor:
            dim_features = dim_features.item()
        indexes = indexes.unsqueeze(2).repeat(1, 1, dim_features)
        x = torch.gather(x, 1, indexes)
        return x
    def forward(self, x):
        batch_size, num_points, dim_features = x.size()
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(dim_features) == torch.Tensor:
            dim_features = dim_features.item()
        index_subsampled_centroids = []
        min_distances = self.min_distances.repeat(batch_size, num_points)
        index = self.first_index.repeat(batch_size).random_(0, num_points)
        for i in range(self.nb_subsampled_points):
            index_subsampled_centroids.append(index.unsqueeze(1))
            index = index.view(batch_size, 1, 1).repeat(1, 1, dim_features)
            centroid = torch.gather(x, 1, index)
            additional_distances = torch.sqrt(torch.sum((x - centroid)**2, dim=2))
            min_distances = torch.min(min_distances, additional_distances)
            _, index = torch.max(min_distances, dim=1)
        index_subsampled_centroids = torch.cat(index_subsampled_centroids, dim=1)
        return index_subsampled_centroids

##############################################
##               GROUPING BLOCK             ##
##############################################
class Grouping(torch.nn.Module):
    def __init__(self, nb_neighbours, sampling_method, use_x, batch_size, patch_radius=None):
        super(Grouping, self).__init__()
        self.nb_neighbours = nb_neighbours
        self.zerotensor = torch.nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.patch_radius = patch_radius
        self.sampling_method = sampling_method
        self.use_x = use_x
        self.batch_tensor = torch.nn.Parameter(torch.arange(batch_size), requires_grad=False)
    def k_nn(self, x, subsampled_centroids):
        distances = square_distance(subsampled_centroids, x)
        distances, indexes = torch.sort(distances, dim=2)
        indexes = indexes[:,:,0:self.nb_neighbours]
        distances = distances[:,:,0:self.nb_neighbours]
        return indexes, distances
    def query_ball(self, x, subsampled_centroids):
        indexes, distances = self.k_nn(x, subsampled_centroids)
        mask = distances > self.patch_radius
        indexes[mask] = -1
        return indexes
    def extract_values(self, x, x_complete, indexes):
        batch_size, _, dim_features = x.size()
        if x_complete is not None:
            _, _, dim_features_complete = x_complete.size()
            if type(dim_features_complete) == torch.Tensor:
                dim_features_complete = dim_features_complete.item()
        _, nb_subsampled_points, num_neighbors = indexes.size()
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        if type(dim_features) == torch.Tensor:
            dim_features = dim_features.item()
        if type(nb_subsampled_points) == torch.Tensor:
            nb_subsampled_points = nb_subsampled_points.item()
        if type(num_neighbors) == torch.Tensor:
            num_neighbors = num_neighbors.item()
        if x_complete is None:
            x_complete = x
            dim_features_complete = dim_features
        x_complete = torch.cat((self.zerotensor.repeat(batch_size, 1, dim_features_complete), x_complete), dim=1)
        indexes = indexes.contiguous().view(batch_size, nb_subsampled_points * num_neighbors, 1) + 1
        indexes_ = indexes.repeat(1, 1, dim_features_complete)
        x_complete = torch.gather(x_complete, 1 , indexes_).view(batch_size, nb_subsampled_points, num_neighbors, dim_features_complete)
        if self.use_x:
            x = torch.cat((self.zerotensor.repeat(batch_size, 1, dim_features), x), dim=1)
            indexes_ = indexes.repeat(1, 1, dim_features)
            x = torch.gather(x, 1, indexes_).view(batch_size, nb_subsampled_points, num_neighbors, dim_features)
            x = x - x[:,:,0,:].unsqueeze(2)
        return x, x_complete
    def forward(self, x, subsampled_centroids):
        if self.sampling_method == 'knn':
            indexes,_ = self.k_nn(x, subsampled_centroids)
        elif self.sampling_method == 'query_ball':
            indexes = self.query_ball(x, subsampled_centroids)
        return indexes

##############################################
##             GROUPING ALL BLOCK           ##
##############################################
class Grouping_all(torch.nn.Module):
    def __init__(self, num_points, use_x):
        super(Grouping_all, self).__init__()
        self.zeros = torch.nn.Parameter(torch.zeros([1, 1, 1]).type(torch.FloatTensor), requires_grad=False)
        self.numpoints_tensor = torch.nn.Parameter(torch.arange(num_points), requires_grad=False)
        self.use_x = use_x
    def extract_values(self, x, x_complete):
        if x_complete is not None:
            if self.use_x:
                x_complete = torch.cat([x, x_complete], dim=2)
            x_complete = x_complete.unsqueeze(1)
        else:
            x_complete = x.unsqueeze(1)
        return x_complete
    def forward(self, x):
        num_batch, num_points, in_channel = x.size()
        if type(num_batch) == torch.Tensor:
            num_batch = num_batch.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(in_channel) == torch.Tensor:
            in_channel = in_channel.item()
        subsampled_centroids = self.zeros.repeat(num_batch, 1, in_channel)
        indexes = self.numpoints_tensor.view(1, 1, num_points).repeat(num_batch, 1, 1)
        return subsampled_centroids, indexes

##############################################
##            INTERPOLATING BLOCK           ##
##############################################
class Interpolating(torch.nn.Module):
    def __init__(self, nb_interpolating_points, use_x_complete_unsampled, batch_size, epsilon=10**(-10)):
        super(Interpolating, self).__init__()
        self.nb_interpolating_points = nb_interpolating_points
        self.use_x_complete_unsampled = use_x_complete_unsampled
        self.epsilon = epsilon
        self.batch_tensor = torch.nn.Parameter(torch.arange(batch_size), requires_grad=False)
    def extract_values(self, x_complete_unsampled, x_complete_sampled, indexes, weights):
        batch_size, num_points, dim_channels_complete_sampled = x_complete_sampled.size()
        if type(indexes) == torch.Tensor:
            _, targeted_num_points, _ = indexes.size()
            if type(targeted_num_points) == torch.Tensor:
                targeted_num_points = targeted_num_points.item()
        else:
            targeted_num_points = indexes
        if type(batch_size) == torch.Tensor:
            batch_size = batch_size.item()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(dim_channels_complete_sampled) == torch.Tensor:
            dim_channels_complete_sampled = dim_channels_complete_sampled.item()
        if num_points == 1:
            x_complete_sampled_temp = x_complete_sampled.repeat(1, targeted_num_points, 1)
        else:
            indexes = indexes.contiguous().view(batch_size, targeted_num_points * self.nb_interpolating_points, 1)
            x_complete_sampled_temp = torch.gather(x_complete_sampled, 1, indexes.repeat(1, 1, dim_channels_complete_sampled)).view(batch_size, targeted_num_points, self.nb_interpolating_points, dim_channels_complete_sampled)
            x_complete_sampled_temp = x_complete_sampled_temp * weights
            x_complete_sampled_temp = torch.sum(x_complete_sampled_temp, dim=2)
        if self.use_x_complete_unsampled and x_complete_unsampled is not None:
            x_complete_sampled = torch.cat((x_complete_sampled_temp, x_complete_unsampled), dim=2)
        else:
            x_complete_sampled = x_complete_sampled_temp
        return x_complete_sampled
    def forward(self, x_unsampled, x_sampled):
        _, num_points, _ = x_sampled.size()
        _, targeted_num_points, _ = x_unsampled.size()
        if type(num_points) == torch.Tensor:
            num_points = num_points.item()
        if type(targeted_num_points) == torch.Tensor:
            targeted_num_points = targeted_num_points.item()
        if num_points < self.nb_interpolating_points:
            self.nb_interpolating_points = num_points
        if num_points == 1:
            indexes = targeted_num_points
            weights = None
        else:
            distances = square_distance(x_unsampled, x_sampled)
            distances, indexes = torch.sort(distances, dim=2)
            distances = distances[:,:,:self.nb_interpolating_points]
            indexes = indexes[:,:,:self.nb_interpolating_points]
            mask = distances < self.epsilon
            distances[mask] = self.epsilon
            weights = 1.0 / distances
            normalization = torch.sum(weights, dim=2, keepdim=True)
            weights = weights / normalization
            weights = weights.unsqueeze(3)
        return indexes, weights
    
##############################################
##               DECODING BLOCK             ##
##############################################
class PointNetFeaturePropagation(torch.nn.Module):
    def __init__(self, nb_interpolating_points, use_x_complete_unsampled, in_channel_complete_sampled, in_channel_complete_unsampled,
                       list_dim_channels, batch_size):
        super(PointNetFeaturePropagation, self).__init__()
        self.nb_interpolating_points = nb_interpolating_points
        self.convlist = torch.nn.ModuleList()
        self.bnlist = torch.nn.ModuleList()
        last_channel = in_channel_complete_sampled
        if use_x_complete_unsampled:
            last_channel += in_channel_complete_unsampled
        self.interpolating = Interpolating(nb_interpolating_points, use_x_complete_unsampled, batch_size)
        for out_channel in list_dim_channels:
            self.convlist.append(torch.nn.Conv1d(last_channel, out_channel, 1, bias=True).float())
            self.bnlist.append(torch.nn.BatchNorm1d(out_channel).float())
            last_channel = out_channel
        self.last_channel = last_channel
    def update_decay(self,bn_decay_value):
        for i in range(len(self.bnlist)):
            self.bnlist[i].momentum = bn_decay_value
    def forward(self, x_unsampled, x_sampled, x_complete_unsampled, x_complete_sampled, bn_decay_value):
        if bn_decay_value is not None:
            self.update_decay(bn_decay_value)
        indexes, weights = self.interpolating(x_unsampled, x_sampled)
        x = self.interpolating.extract_values(x_complete_unsampled, x_complete_sampled, indexes, weights)
        x = x.transpose(1, 2)
        for i in range(len(self.convlist)):
            x = self.convlist[i](x)
            x = self.bnlist[i](x)
            x = torch.nn.functional.relu(x)
        x = x.transpose(1, 2)
        return x