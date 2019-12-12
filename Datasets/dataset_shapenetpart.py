import os
import json
import torch
import itertools
import numpy as np
import torch.utils.data as data
from Datasets.point_clouds import PointClouds

import pdb

# Loading the dataset
class ShapeNetPartDataset(data.Dataset):
    def __init__(self, root, seed=None, num_points=2048, center_points=True, use_pca=False, mode='training'):
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
            file_split_train = os.path.join(self.root, 'train_test_split/shuffled_train_file_list.json')
            files_train = json.load(open(file_split_train))
            file_split_val = os.path.join(self.root, 'train_test_split/shuffled_val_file_list.json')
            files_val = json.load(open(file_split_val))
            files = files_train + files_val
        else:
            file_split_test = os.path.join(self.root, 'train_test_split/shuffled_test_file_list.json')
            files = json.load(open(file_split_test))
        # Getting the shape names
        self.shape_names = np.sort(np.unique([file_.split('/')[-2] for file_ in files]))
        self.nb_shapes = len(self.shape_names)
        self.shape_length = []
        self.pointcloud_names = []
        self.files_output = []
        for shape in self.shape_names:
            additional_pointclouds = np.unique([file_.split('/')[-1].replace('.txt', '') for file_ in files if shape in file_])
            self.files_output += [os.path.join(self.root, shape + '/' + file_ + '_prediction.npy') for file_ in additional_pointclouds]
            self.shape_length += [len(additional_pointclouds)]
            self.pointcloud_names += additional_pointclouds.tolist()
        textfile = open(os.path.join(self.root,'synsetoffset2category.txt'),'r')
        self.dictionary_shapes = {}
        for line in textfile.readlines():
            name_shape = line.split('\t')[0]
            code_shape = line.split('\t')[1].replace('\n','')
            self.dictionary_shapes[code_shape] = name_shape
        textfile = open(os.path.join(self.root,'category_per_class.txt'),'r')
        self.dictionary_categories = {}
        for line in textfile.readlines():
            directory = line.split(': ')[0]
            categories = list(map(int,line.split(': ')[1].split(' ')))
            self.dictionary_categories[directory] = categories
        self.num_parts = len(sum(list(self.dictionary_categories.values()), []))
    def __getitem__(self, index):
        # find shape that contains the point with given global index
        pointcloud = self.load_pointcloud_by_index(index)
        vertices = torch.from_numpy(pointcloud.pts)
        normals = torch.from_numpy(pointcloud.normals)
        labels_cat = torch.LongTensor([pointcloud.labels_cat])
        labels_seg = torch.from_numpy(pointcloud.labels_seg)
        parts_tensor = torch.zeros(self.num_parts)
        parts_tensor[self.dictionary_categories[sorted(self.dictionary_categories.keys())[pointcloud.labels_cat]]] = 1
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
        features = torch.cat((vertices, normals), dim=1)
        return (features, labels_cat, labels_seg, parts_tensor)
    def __len__(self):
        return len(self.pointcloud_names)
    # Loading a pointcloud
    def get_sample_info(self, pointcloud_ind):
        cumsum = np.cumsum(self.shape_length)
        index_shape = np.where(cumsum > pointcloud_ind)[0][0]
        return index_shape, self.shape_names[index_shape], self.pointcloud_names[pointcloud_ind]
    def load_pointcloud_by_index(self, pointcloud_ind):
        labels_cat, shape_name, pointcloud_name = self.get_sample_info(pointcloud_ind)
        filename = os.path.join(self.root, shape_name + '/' + pointcloud_name)
        data = np.loadtxt(filename + '.txt', delimiter=' ')
        if self.mode != 'evaluation':
            if data.shape[0]<self.num_points:
                new_indexes = np.random.choice(data.shape[0], self.num_points, replace=True)
            else:
                new_indexes = np.random.choice(data.shape[0], self.num_points, replace=False)
            data = data[new_indexes,:]
        pts = data[:,0:3]
        normals = data[:,3:6]
        labels_seg = data[:,6]
        labels_seg = labels_seg.astype(int)
        pointcloud = PointClouds(pts, normals, labels_cat, labels_seg, weigths=None)
        if self.mode != 'evaluation':
            pointcloud.shuffle_points()
        return pointcloud

if __name__=='__main__':
    from dataset_samplers import *
    root = '/mnt/Data/ShapenetPart/'
    maxbatchSize = 4
    workers = 2
    dataset = ShapeNetPartDataset(root)
    datasampler = Sampler(data_source=dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=datasampler,batch_size=maxbatchSize,num_workers=int(workers))
    enum = enumerate(dataloader, 0)
    for batchind, data in enum:
        print(len(data))
        print(data[0].shape)