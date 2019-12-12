import os
import json
import itertools
import numpy as np
import torch
import torch.utils.data as data

import pdb

class PointClouds():
    def __init__(self, pts, normals=None, labels_cat=None, labels_seg=None, weigths=None):
        self.pts = pts
        self.normals = normals
        self.labels_cat = labels_cat
        self.labels_seg = labels_seg
        self.weigths = weigths
    def shuffle_points(self):
        num_pts = len(self.pts)
        new_indexes = np.arange(num_pts)
        np.random.shuffle(new_indexes)
        # Vertices
        self.pts = self.pts[new_indexes,:]
        # Normals
        if self.normals is not None:
            self.normals = self.normals[new_indexes,:]
        # Labels
        self.labels_seg = self.labels_seg[new_indexes]
        # Weigths
        if self.weigths is not None:
            self.weigths = self.weigths[new_indexes]
    def rotate_point_cloud_x(self, factor=1):
        rotation_angle = np.random.uniform() * 2 * np.pi * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0], [0, cosval, sinval], [0, -sinval, cosval]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_point_cloud_y(self, factor=1):
        rotation_angle = np.random.uniform() * 2 * np.pi * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_point_cloud_z(self, factor=1):
        rotation_angle = np.random.uniform() * 2 * np.pi * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],[-sinval, cosval, 0],[0, 0, 1]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_point_cloud_by_angle_x(self, rotation_angle, factor=1):
        rotation_angle = rotation_angle * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],[0, cosval, sinval],[0, -sinval, cosval]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_point_cloud_by_angle_y(self, rotation_angle, factor=1):
        rotation_angle = rotation_angle * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],[0, 1, 0],[-sinval, 0, cosval]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_point_cloud_by_angle_z(self, rotation_angle, factor=1):
        rotation_angle = rotation_angle * factor
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],[-sinval, cosval, 0],[0, 0, 1]])
        self.pts = np.dot(self.pts, rotation_matrix)
        if self.normals is not None:
            self.normals = np.dot(self.normals, rotation_matrix)
    def rotate_perturbation_point_cloud(self, angle_sigma=0.06, angle_clip=0.18, factor=1):
        angles = np.clip(angle_sigma * np.random.randn(3) * factor, -angle_clip * factor, angle_clip * factor)
        Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), np.sin(angles[0])], [0, -np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), np.sin(angles[2]), 0], [-np.sin(angles[2]), np.cos(angles[2]), 0],[0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        self.pts = np.dot(self.pts, R)
        if self.normals is not None:
            self.normals = np.dot(self.normals, R)
    def random_scale_point_cloud(self, scale_low=0.8, scale_high=1.25, factor=1):
        scales = np.random.uniform(scale_low, scale_high)
        scales = 1 + factor * (scales - 1)
        self.pts *= scales
    def shift_point_cloud(self, shift_range=0.1, factor=1):
        shifts = np.random.uniform(-shift_range * factor, shift_range * factor, 3)
        shifts = shifts * factor
        self.pts += shifts
    def jitter_point_cloud(self, sigma=0.01, clip=0.05, factor=1):
        self.pts = self.pts+np.clip(sigma * np.random.randn(pts.shape[0], pts.shape[1]) * factor, -1 * clip * factor, clip * factor)
    def augment_data(self, factor=1):
        pts, normals = self.rotate_point_cloud_y(self.pts, normals=self.normals, factor=factor)
        pts, normals = self.rotate_perturbation_point_cloud(pts, normals=normals, factor=factor)
        pts = self.jitter_point_cloud(pts, factor=factor)
        self.pts = pts
        self.normals = normals
    def point_cloud_label_to_surface_voxel_label_fast(self, res=0.0484):
        coordmax = np.max(self.pts, axis=0)
        coordmin = np.min(self.pts, axis=0)
        nvox = np.ceil((coordmax - coordmin) / res)
        vidx = np.ceil((self.pts - coordmin) / res)
        vidx = vidx[:,0] + vidx[:,1] * nvox[0] + vidx[:,2] * nvox[0] * nvox[1]
        uvidx, vpidx = np.unique(vidx, return_index=True)
        if self.labels_seg.ndim==1:
            uvlabel = self.labels_seg[vpidx]
        else:
            assert(self.labels_seg.ndim==2)
            uvlabel = self.labels_seg[vpidx,:]
        return uvidx, uvlabel, nvox