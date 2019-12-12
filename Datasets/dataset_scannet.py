# Importation of libraries
import os
import pickle
import numpy as np
import torch.utils.data as data
from Datasets.scene_util import virtual_scan
from Datasets.point_clouds import PointClouds

import pdb

#######################################################################################################################
# This script is based on the PointNet++ code repository
# https://github.com/charlesq34/pointnet2
#######################################################################################################################
# Loading the dataset
class ScanNetDataset(data.Dataset):
    def __init__(self, root, seed=None, num_points=8192, mode='training'):
        self.root = root
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32 - 1, 1)[0]
        np.random.seed(self.seed)
        self.num_points = num_points
        self.mode = mode 
        if self.mode == 'training':
            self.data_filename = 'scannet_train.pickle'
        elif (self.mode == 'validation') or (self.mode == 'evaluation'):
            self.data_filename = 'scannet_test.pickle'
        with open(os.path.join(self.root, self.data_filename), 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='latin1')
            self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if self.mode == 'training':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        else:
            labelweights = np.ones(21)
            self.labelweights = labelweights / np.sum(labelweights)
    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        smpmin = np.maximum(coordmax - [1.5, 1.5, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax - smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2] - coordmin[2]
        isvalid = False
        for _ in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0],:]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask,:] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:,0] * 31.0 * 62.0 + vidx[:,1] * 62.0 + vidx[:,2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.num_points, replace=True)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        if self.mode=='training':
            point_set = PointClouds(point_set)
            point_set.rotate_point_cloud_z(factor=1)
            point_set = point_set.pts

            dropout_ratio = np.random.random() * 0.875
            drop_idx = np.where(np.random.random((point_set.shape[0])) <= dropout_ratio)[0]
            point_set[drop_idx,:] = point_set[0,:]
            semantic_seg[drop_idx] = semantic_seg[0]
            sample_weight[drop_idx] *= 0
        return point_set, semantic_seg, sample_weight
    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetWholeScene(data.Dataset):
    def __init__(self, root, seed=None, num_points=8192, mode='training'):
        self.root = root
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32 - 1, 1)[0]
        np.random.seed(self.seed)
        self.num_points = num_points
        self.mode = mode
        if self.mode == 'training':
            self.data_filename = 'scannet_train.pickle'
        elif (self.mode == 'validation') or (self.mode == 'evaluation'):
            self.data_filename = 'scannet_test.pickle'
        with open(os.path.join(self.root, self.data_filename), 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='latin1')
            self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if self.mode == 'training':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif (self.mode == 'validation') or (self.mode == 'evaluation'):
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmax = coordmin + [(i + 1) * 1.5, (j +1 ) * 1.5, coordmax[2] - coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.num_points, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set, 0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        if self.mode == 'training':
            point_sets = PointClouds(point_sets)
            point_sets.rotate_point_cloud_z(factor=1)
            point_sets = point_sets.pts
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetVirtualScan(data.Dataset):
    def __init__(self, root, seed=None, num_points=8192, mode='training'):
        self.root = root
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32 - 1, 1)[0]
        np.random.seed(self.seed)
        self.num_points = num_points
        self.mode = mode
        if self.mode == 'training':
            self.data_filename = 'scannet_train.pickle'
        elif (self.mode == 'validation') or (self.mode == 'evaluation'):
            self.data_filename = 'scannet_test.pickle'
        with open(os.path.join(self.root, self.data_filename), 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='latin1')
            self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if self.mode == 'training':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif (self.mode == 'validation') or (self.mode == 'evaluation'):
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(8):
            smpidx = virtual_scan(point_set_ini, mode=i)
            if len(smpidx) < 300:
                continue
            point_set = point_set_ini[smpidx,:]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.num_points, replace=True)
            point_set = point_set[choice,:] # Nx3
            semantic_seg = semantic_seg[choice] # N
            sample_weight = sample_weight[choice] # N
            point_sets.append(np.expand_dims(point_set, 0)) # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg, 0)) # 1xN
            sample_weights.append(np.expand_dims(sample_weight, 0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        if self.mode == 'training':
            point_sets = PointClouds(point_sets)
            point_sets.rotate_point_cloud_z(factor=1)
            point_sets = point_sets.pts
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

if __name__=='__main__':
    import torch
    from dataset_samplers import *
    root = '/mnt/Data/scannet/'
    maxbatchSize = 4
    workers = 2
    dataset = ScanNetDataset(root)
    datasampler = Sampler(data_source=dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=datasampler,batch_size=maxbatchSize,num_workers=int(workers))
    enum = enumerate(dataloader, 0)
    for batchind, data in enum:
        print(len(data))
        print(data[0].shape)
    
    maxbatchSize = 1
    dataset = ScannetDatasetWholeScene(root)
    datasampler = Sampler(data_source=dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=datasampler,batch_size=maxbatchSize,num_workers=int(workers))
    enum = enumerate(dataloader, 0)
    for batchind, data in enum:
        print(len(data))
        print(data[0].shape)