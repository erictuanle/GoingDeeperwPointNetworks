import os
import sys
import torch
import shutil
import random
import numpy as np
from tensorboardX import SummaryWriter

from Datasets.dataset_shapenetpart import ShapeNetPartDataset
from Datasets.dataset_samplers import RandomSampler, Sampler

from Models.model_PN2 import PointNet2
from Models.model_mRes import mRes
from Models.model_convPN import convPN

from Configs.shapenetpart_options import ShapeNetPartOptions

from Utils.evaluation_metrics import compute_performance_metrics

import pdb

def compute_loss(pred, target):
    num_batch, num_points, num_classes = pred.size()
    pred = pred.contiguous().view(num_batch * num_points, num_classes)
    target = target.view(num_batch * num_points)
    loss = torch.nn.functional.cross_entropy(pred, target)
    return loss

def test_shapenetpart(opt, trainopt, model_filename):
    # Creating the device
    if opt.use_GPU:
        device = torch.device("cuda:" + str(opt.device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print('Network loaded on the device: ', device)
    # Colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'
    # Set up folder directories
    resdir = os.path.join(opt.resdir, trainopt.name)
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    # Create train and test dataset loaders
    test_dataset = ShapeNetPartDataset(root=opt.indir, seed=opt.seed, num_points=opt.num_points_training, center_points=opt.center_points,
                                        use_pca=opt.use_pca, mode='evaluation')
    test_datasampler = Sampler(data_source=test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_datasampler, batch_size=1, num_workers=int(opt.workers))
    print('test set: %d pointclouds (in %d minibatches)' % (len(test_datasampler), len(test_dataloader)))
    list_num_parts = list(test_dataset.dictionary_categories.values())
    # Creating the network
    if trainopt.network == 'PointNet++':
        # PN++
        network = PointNet2(trainopt.batch_size, trainopt.nb_subsampled_points, trainopt.nb_neighbours, trainopt.sampling_method, trainopt.patch_radius, trainopt.in_channel_x_complete,
                            trainopt.in_channel, trainopt.list_dim_channels_encoding1, trainopt.use_x, trainopt.pooling_operation, trainopt.list_dim_channels_encoding2,
                            trainopt.intermediate_size_fc, trainopt.dropout_rate, trainopt.nb_interpolating_points, trainopt.use_x_complete_unsampled, trainopt.list_dim_channels_decoding,
                            trainopt.num_classes, trainopt.num_parts).to(device)
    elif (trainopt.network == 'mRes') or (trainopt.network == 'mResX'):
        # mRes
        network = mRes(trainopt.batch_size, trainopt.nb_subsampled_points, trainopt.nb_neighbours, trainopt.sampling_method, trainopt.patch_radius, trainopt.in_channel_x_complete, trainopt.in_channel,
                       trainopt.list_dim_channels_encoding1, trainopt.use_x, trainopt.cross_connection, trainopt.pooling_operation, trainopt.list_dim_channels_encoding2, trainopt.intermediate_size_fc,
                       trainopt.dropout_rate, trainopt.nb_interpolating_points, trainopt.use_x_complete_unsampled, trainopt.list_dim_channels_decoding, trainopt.num_classes, trainopt.num_parts,
                       trainopt.dropout_rate_cross, trainopt.nb_interpolating_points_encoding).to(device)
        network.add_cross_connection(trainopt.batch_size, trainopt.nb_interpolating_points_crossconnection)
        network = network.to(device)
    elif (trainopt.network == 'convPN') or (trainopt.network == 'deepConvPN'):
        # convPN
        network = convPN(trainopt.batch_size, trainopt.nb_subsampled_points, trainopt.nb_neighbours, trainopt.sampling_method, trainopt.patch_radius, trainopt.in_channel_x_complete, trainopt.in_channel,
                         trainopt.list_dim_channels_encoding, trainopt.use_x, trainopt.use_crosslinks, trainopt.use_reslinks, trainopt.sequence, trainopt.pooling_operation, trainopt.residuallinks_input,
                         trainopt.residuallinks_output, trainopt.intermediate_size_fc, trainopt.dropout_rate, trainopt.nb_interpolating_points, trainopt.use_x_complete_unsampled,
                         trainopt.list_dim_channels_decoding, trainopt.num_classes, trainopt.num_parts, trainopt.blockout_rate, test=False).to(device)
    if os.path.isfile(model_filename):
        network.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
    else:
        print('Model not found at: '+model_filename)
    num_parameters = np.sum([np.prod(parameter.shape) for parameter in network.parameters()])
    print('Number of parameters for ' + trainopt.network + ': ' + str(num_parameters))

    network.eval()
    if (trainopt.network == 'convPN') or (trainopt.network == 'deepConvPN'):
        network.eval_custom()
    # Initializing the metrics variable
    loss_evaluation = np.zeros([opt.num_classes])
    accuracy_evaluation = np.zeros([opt.num_classes])
    iou_evaluation = np.zeros([opt.num_classes])
    intersection_evaluation = np.zeros([opt.num_parts])
    union_evaluation = np.zeros([opt.num_parts])
    cpt_samples = np.zeros([opt.num_classes])
    # Iterating over the batches
    for i, data in enumerate(test_dataloader, 0):
        if i % opt.nb_rolling_iterations == 0:
            print(str(i) + '/' + str(len(test_dataloader)))
        input_tensor = data[0].type(torch.FloatTensor).to(device)
        vertices = input_tensor[:,:,:3].repeat(opt.batch_size, 1, 1)
        normals = input_tensor[:,:,3:].repeat(opt.batch_size, 1, 1)
        labels_cat = data[1].type(torch.LongTensor).to(device).repeat(opt.batch_size, 1)
        labels_seg = data[2].type(torch.LongTensor).to(device)
        parts_tensor = data[3].type(torch.FloatTensor).to(device)
        zero_tensor = torch.zeros([1]).to(device)
        one_tensor = torch.ones([1]).to(device)
        with torch.no_grad():
            print(vertices.shape)
            pred = network(vertices, normals, labels_cat, bn_decay_value=None)
            pred = torch.sum(pred, dim=0, keepdim=True).detach()
            labels_cat = labels_cat[0].unsqueeze(0)
            loss = compute_loss(pred=pred, target=labels_seg)

            cpt_samples[labels_cat.item()] += 1
            loss_evaluation[labels_cat.item()] += loss.detach().cpu().item()
            batch_accuracy, batch_iou, batch_intersection, batch_union = compute_performance_metrics(labels_cat, labels_seg, pred, None, parts_tensor, zero_tensor, one_tensor)
            accuracy_evaluation[labels_cat.item()] += batch_accuracy
            iou_evaluation[labels_cat.item()] += batch_iou
            intersection_evaluation = intersection_evaluation + batch_intersection.detach().cpu().numpy()
            union_evaluation = union_evaluation + batch_union.detach().cpu().numpy()
    # Exporting the performance on the test set
    with open(os.path.join(resdir, 'Results_ShapeNet-Part.txt'), 'w') as f:
        f.write('-------------------Loss (Softmax Cross-Entropy)-------------------\n')
        f.write('Global loss: ' + str(np.sum(loss_evaluation) / np.sum(cpt_samples)) + '\n')
        for i, category in enumerate(test_dataset.shape_names):
            if cpt_samples[i] > 0:
                f.write(test_dataset.dictionary_shapes[category] + ': ' + str(iou_evaluation[i] / cpt_samples[i]) + '\n')
            else:
                f.write(test_dataset.dictionary_shapes[category] + ': None\n')
        f.write('-------------------Accuracy-------------------\n')
        f.write('Global accuracy: ' + str(np.sum(accuracy_evaluation) / np.sum(cpt_samples)) + '\n')
        for i, category in enumerate(test_dataset.shape_names):
            if cpt_samples[i] > 0:
                f.write(test_dataset.dictionary_shapes[category] + ': ' + str(accuracy_evaluation[i] / cpt_samples[i]) + '\n')
            else:
                f.write(test_dataset.dictionary_shapes[category] + ': None\n')
        f.write('-------------------Intersection Over Union-------------------\n')
        f.write('Global IoU: ' + str(np.sum(iou_evaluation) / np.sum(cpt_samples)) + '\n')
        for i, category in enumerate(test_dataset.shape_names):
            if cpt_samples[i] > 0:
                f.write(test_dataset.dictionary_shapes[category] + ': ' + str(iou_evaluation[i] / cpt_samples[i]) + '\n')
            else:
                f.write(test_dataset.dictionary_shapes[category] + ': None\n')
        f.write('-------------------Part Intersection Over Union-------------------\n')
        indexes_mask = np.where(union_evaluation!=0)[0]
        intersection_global = intersection_evaluation[indexes_mask]
        union_global = union_evaluation[indexes_mask]
        f.write('Global: ' + str(np.mean(intersection_global/union_global))+'\n')
        for i, category in enumerate(test_dataset.shape_names):
            parts = test_dataset.dictionary_categories[category]
            union_cat = union_global[parts]
            intersection_cat = intersection_global[parts]
            if len(union_cat)>0:
                f.write(test_dataset.dictionary_shapes[category] + ': '+str(np.mean(intersection_cat / union_cat)) + '\n')
            else:
                f.write(test_dataset.dictionary_shapes[category] + ': None\n')

if __name__ == '__main__':
    configs = ShapeNetPartOptions()
    print('The parameters used to build the network will be the same of for the training')
    opt = configs.parse()
    model_filename = os.path.join(opt.outdir, '%s_model_200.pth' % (opt.name))
    trainparam_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    trainopt = torch.load(trainparam_filename)
    test_shapenetpart(opt, trainopt, model_filename)