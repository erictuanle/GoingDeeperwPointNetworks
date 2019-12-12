from __future__ import print_function
import os
import h5py
import numpy as np
import torch
import torch.utils.data as data


#######################################################################################################################
# This script is based on the DeepGCN code repository
# https://github.com/lightaime/deep_gcns
#######################################################################################################################
# Loading the dataset
class PointCloudDataset_training(data.Dataset):
    def __init__(self, root='/mnt/Data/', test_area_idx=None, num_points=4096, mode='training'):
        self.root = root
        self.all_files = [line.rstrip() for line in open(os.path.join(root, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))]
        self.room_filelist = [line.rstrip() for line in open(os.path.join(root, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]
        self.num_points = num_points
        self.mode = mode
        data_batch_list = []
        label_batch_list = []
        for h5_filename in self.all_files:
            f = h5py.File(os.path.join(root,h5_filename), 'r', swmr=True)
            data_batch = f['data'][:]
            label_batch = f['label'][:]
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)
        test_area = 'Area_' + str(test_area_idx)
        idxs = []
        for i, room_name in enumerate(self.room_filelist):
            if (self.mode=='training') and not (test_area in room_name):
                idxs.append(i)
            elif (self.mode=='evaluation') and (test_area in room_name):
                idxs.append(i)
        self.idxs = idxs
        self.data_batches = data_batches[idxs]
        self.label_batches = label_batches[idxs]
    def __getitem__(self,index):
        data_batch = self.data_batches[index][:,:6]
        label_batch = self.label_batches[index]
        index = self.idxs[index]
        indexes = np.load(os.path.join(self.root, 'indoor3d_sem_seg_hdf5_data', 'Indexes', str(index) + '.npy'))
        # Centering
        vertices = data_batch[:,:3]
        vertices = (vertices - np.mean(vertices, axis=0, keepdims=True))
        normalization = np.max(np.sqrt(np.sum(vertices**2, axis=1)), axis=0)
        data_batch[:,:3] = vertices / normalization
        data_batch = torch.from_numpy(data_batch[:self.num_points])
        label_batch = torch.from_numpy(label_batch[:self.num_points])
        indexes = torch.from_numpy(indexes[:self.num_points])
        return (data_batch, label_batch, indexes)
    def __len__(self):
        return len(self.data_batches)

class PointCloudDataset_evaluation(data.Dataset):
    def __init__(self, root='/mnt/Data/', test_area_idx=None, num_point=4096, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1,
                 vizualization=False):
        self.num_points = num_point
        self.vizualization = vizualization
        self.room_path_list = [os.path.join(root, line.rstrip().replace('data/','')) for line in open(os.path.join(root, 'Stanford3dDataset_v1.2_Aligned_Version/meta/area' + str(test_area_idx) + '_data_label.txt'))]
        new_data_batch_list = []
        label_batch_list = []
        if self.vizualization:
            size_batch_list = []
            max_values_list = []
        for room_path in self.room_path_list:
            if room_path[-3:] == 'txt':
                data_label = np.loadtxt(room_path)
            elif room_path[-3:] == 'npy':
                data_label = np.load(room_path)
            else:
                print('Unknown file type! exiting.')
            data = data_label[:,0:6]
            data[:,3:6] /= 255.0
            label = data_label[:,-1].astype(np.uint8)
            max_room_x = max(data[:,0])
            max_room_y = max(data[:,1])
            max_room_z = max(data[:,2])
            data_batch, label_batch = self.room2blocks(data, label, num_point, block_size, stride, random_sample, sample_num, sample_aug)
            new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
            for b in range(data_batch.shape[0]):
                new_data_batch[b,:,6] = data_batch[b,:,0] / max_room_x
                new_data_batch[b,:,7] = data_batch[b,:,1] / max_room_y
                new_data_batch[b,:,8] = data_batch[b,:,2] / max_room_z
                minx = min(data_batch[b,:,0])
                miny = min(data_batch[b,:,1])
                data_batch[b,:,0] -= (minx + block_size / 2)
                data_batch[b,:,1] -= (miny + block_size / 2)
            new_data_batch[:,:,0:6] = data_batch
            new_data_batch_list.append(new_data_batch)
            label_batch_list.append(label_batch)
            if self.vizualization:
                max_values_list.append(np.array([[max_room_x, max_room_y, max_room_z]]))
                size_batch_list.append(new_data_batch.shape[0])
        self.data_batches = np.concatenate(new_data_batch_list, axis=0)
        self.label_batches = np.concatenate(label_batch_list, axis=0)
        if self.vizualization:
            self.max_values = np.concatenate(max_values_list, axis=0)
            self.size_batch = np.array(size_batch_list)
    def sample_data_label(self, data, label, num_sample):
        N = data.shape[0]
        if (N == num_sample):
            new_data = data
            sample_indices = list(range(N))
        elif (N > num_sample):
            sample = np.random.choice(N, num_sample)
            new_data = data[sample, ...]
            sample_indices = sample
        else:
            sample = np.random.choice(N, num_sample - N)
            dup_data = data[sample, ...]
            new_data = np.concatenate([data, dup_data], 0)
            sample_indices = list(range(N)) + list(sample)
        new_label = label[sample_indices]
        return new_data, new_label
    def room2blocks(self, data, label, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):
        assert(stride <= block_size)
        limit = np.amax(data, 0)[0:3]
        # Get the corner location for our sampling blocks
        xbeg_list = []
        ybeg_list = []
        if not random_sample:
            num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
            num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
            for i in range(num_block_x):
                for j in range(num_block_y):
                    xbeg_list.append(i * stride)
                    ybeg_list.append(j * stride)
        else:
            num_block_x = int(np.ceil(limit[0] / block_size))
            num_block_y = int(np.ceil(limit[1] / block_size))
            if sample_num is None:
                sample_num = num_block_x * num_block_y * sample_aug
            for _ in range(sample_num):
                xbeg = np.random.uniform(-block_size, limit[0])
                ybeg = np.random.uniform(-block_size, limit[1])
                xbeg_list.append(xbeg)
                ybeg_list.append(ybeg)
        # Collect blocks
        block_data_list = []
        block_label_list = []
        for idx in range(len(xbeg_list)):
            xbeg = xbeg_list[idx]
            ybeg = ybeg_list[idx]
            xcond = (data[:,0] <= xbeg + block_size) & (data[:,0] >= xbeg)
            ycond = (data[:,1] <= ybeg + block_size) & (data[:,1] >= ybeg)
            cond = xcond & ycond
            if np.sum(cond) < 100: # discard block if there are less than 100 pts.
                continue
            block_data = data[cond,:]
            block_label = label[cond]
            # randomly subsample data
            block_data_sampled, block_label_sampled = self.sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
        return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)
    def __getitem__(self,index):
        data_batch = self.data_batches[index]
        label_batch = self.label_batches[index]
        if not self.vizualization:
            data_batch = data_batch[:,:6]
        vertices = data_batch[:,:3]
        vertices = (vertices - np.mean(vertices, axis=0, keepdims=True))
        normalization = np.max(np.sqrt(np.sum(vertices**2, axis=1)), axis=0)
        data_batch[:,:3] = vertices / normalization
        data_batch = torch.from_numpy(data_batch[:self.num_points])
        label_batch = torch.from_numpy(label_batch[:self.num_points])
        return (data_batch, label_batch)
    def __len__(self):
        return len(self.data_batches)

if __name__=='__main__':
    from dataset_samplers import *
    root = '/mnt/Data/ShapenetPart/'
    maxbatchSize = 4
    workers = 2
    dataset = PointCloudDataset_training(root='/mnt/Data/', test_area_idx=None, num_points=4096, mode='training'
    datasampler = Sampler(data_source=dataset)
    dataloader = torch.utils.data.DataLoader(dataset,sampler=datasampler,batch_size=maxbatchSize,num_workers=int(workers))
    enum = enumerate(dataloader, 0)
    for batchind, data in enum:
        print(len(data))
        print(data[0].shape)