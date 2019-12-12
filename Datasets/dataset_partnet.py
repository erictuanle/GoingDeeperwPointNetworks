import os
import h5py
import torch
import numpy as np
import torch.utils.data as data
from Datasets.point_clouds import PointClouds

import pdb

# Loading the dataset
class PartNetDataset(data.Dataset):
    def __init__(self, root, level='3', seed=None, num_points=10000, center_points=True, use_pca=False, mode='training'):
        self.root = root
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32 - 1, 1)[0]
        np.random.seed(self.seed)
        self.num_points = num_points
        self.center_points = center_points
        self.use_pca = use_pca
        self.mode = mode
        if self.mode == 'training':
            self.prefix = 'train'
        elif self.mode == 'validation':
            self.prefix = 'val'
        elif self.mode == 'evaluation':
            self.prefix = 'test'
        self.categories = sorted([category for category in os.listdir(self.root) if (level in category) and os.path.isdir(os.path.join(self.root, category))])
        list_pts = []
        list_labels_cat = []
        list_labels_seg = []
        self.part_categories = []
        for category in self.categories:
            files = [file_ for file_ in os.listdir(os.path.join(self.root, category)) if (self.prefix in file_) and ('mod' in file_) and ('.h5' in file_[-3:])]
            part_category = np.array([])
            for file_ in files:
                data = h5py.File(os.path.join(self.root, category, file_), 'r')
                pts = data['pts'][...]
                labels_cat = data['labels_cat'][...]
                labels_seg = data['labels_seg'][...]
                list_pts.append(pts)
                list_labels_cat.append(labels_cat)
                list_labels_seg.append(labels_seg)
                part_category = np.unique(np.append(part_category, np.unique(labels_seg)))
            self.part_categories.append(part_category)
        self.pts = np.concatenate(list_pts, axis=0)
        self.labels_cat = np.concatenate(list_labels_cat, axis=0)
        self.labels_seg = np.concatenate(list_labels_seg, axis=0)
    def __getitem__(self, index):
        pointcloud = self.load_pointcloud_by_index(index)
        vertices = torch.from_numpy(pointcloud.pts.astype(float))
        labels_cat = torch.LongTensor([pointcloud.labels_cat])
        labels_seg = torch.from_numpy(pointcloud.labels_seg.astype(int))
        weigths = torch.from_numpy(pointcloud.weigths.astype(float))
        if self.center_points == True:
            vertices = (vertices - torch.mean(vertices, dim=0, keepdim=True))
            normalization, _ = torch.max(torch.sqrt(torch.sum(vertices**2, dim=1)), dim=0)
            vertices = vertices / normalization
        if self.use_pca:
            # compute pca of points in the patch:
            mean = vertices.mean(0)
            vertices = vertices - mean
            trans, _, _ = torch.svd(torch.t(vertices))
            vertices = torch.mm(vertices, trans)
            cp_new = -mean
            cp_new = torch.matmul(cp_new, trans)
            vertices = vertices - cp_new
        return (vertices, labels_cat, labels_seg, weigths)
    def __len__(self):
        return len(self.pts)
    # Loading a pointcloud
    def load_pointcloud_by_index(self, pointcloud_ind):
        pts = self.pts[pointcloud_ind]
        num_points, _ = pts.shape
        labels_cat = self.labels_cat[pointcloud_ind]
        labels_seg = self.labels_seg[pointcloud_ind]
        if (self.mode != 'evaluation') and (pts.shape[0]!=num_points):
            if pts.shape[0] < num_points:
                new_indexes = np.random.choice(pts.shape[0], num_points, replace=True)
            elif pts.shape[0] > num_points:
                new_indexes = np.random.choice(pts.shape[0], num_points, replace=False)
            pts = pts[new_indexes,:]
            labels_seg = labels_seg[new_indexes,:]
        weigths = (labels_seg != -1).astype(float)
        if sum(weigths) > 0:
            weigths = weigths / sum(weigths)
        labels_seg[labels_seg == -1] = 0
        pointcloud = PointClouds(pts=pts, labels_cat=labels_cat, labels_seg=labels_seg, weigths=weigths)
        if self.mode == 'training':
            pointcloud.shuffle_points()
        return pointcloud

if __name__=='__main__':
    from dataset_samplers import *
    root = '/mnt/Data/PartNet2'
    maxbatchSize = 4
    workers = 2
    dataset = PointCloudDataset(root)
    datasampler = Sampler(data_source=dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=datasampler,batch_size=maxbatchSize,num_workers=int(workers))
    enum = enumerate(dataloader, 0)
    for batchind, data in enum:
        print(len(data))
        print(data[0].shape)
