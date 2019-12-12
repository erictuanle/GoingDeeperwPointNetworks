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

np.seterr(divide='ignore', invalid='ignore')

import pdb

def compute_loss(pred, target):
    num_batch, num_points, num_classes = pred.size()
    pred = pred.contiguous().view(num_batch * num_points, num_classes)
    target = target.view(num_batch * num_points)
    loss = torch.nn.functional.cross_entropy(pred, target)
    return loss

def train_shapenetpart(opt):
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
    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))
    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()
    if not os.path.isdir(opt.outdir):
        os.makedirs(opt.outdir)
    # Set up the seed
    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # Create train and test dataset loaders
    train_dataset = ShapeNetPartDataset(root=opt.indir, seed=opt.seed, num_points=opt.num_points_training, center_points=opt.center_points,
                                        use_pca=opt.use_pca, mode='training')
    train_datasampler = RandomSampler(data_source=train_dataset, seed=opt.seed, identical_epochs=opt.identical_epochs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_datasampler, batch_size=opt.batch_size, num_workers=int(opt.workers))
    print('training set: %d pointclouds (in %d minibatches)' % (len(train_datasampler), len(train_dataloader)))
    if opt.validation_batch == True:
        test_dataset = ShapeNetPartDataset(root=opt.indir, seed=opt.seed, num_points=opt.num_points_training, center_points=opt.center_points,
                                           use_pca=opt.use_pca, mode='validation')
        test_datasampler = Sampler(data_source=test_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_datasampler, batch_size=opt.batch_size, num_workers=int(opt.workers))
        print('test set: %d pointclouds (in %d minibatches)' % (len(test_datasampler), len(test_dataloader)))
    list_num_parts = list(train_dataset.dictionary_categories.values())
    # Creating the network
    if opt.network == 'PointNet++':
        # PN++
        network = PointNet2(opt.batch_size, opt.nb_subsampled_points, opt.nb_neighbours, opt.sampling_method, opt.patch_radius, opt.in_channel_x_complete,
                            opt.in_channel, opt.list_dim_channels_encoding1, opt.use_x, opt.pooling_operation, opt.list_dim_channels_encoding2,
                            opt.intermediate_size_fc, opt.dropout_rate, opt.nb_interpolating_points, opt.use_x_complete_unsampled,opt.list_dim_channels_decoding,
                            opt.num_classes, opt.num_parts).to(device)
    elif (opt.network == 'mRes') or (opt.network == 'mResX'):
        # mRes
        network = mRes(opt.batch_size, opt.nb_subsampled_points, opt.nb_neighbours, opt.sampling_method, opt.patch_radius, opt.in_channel_x_complete, opt.in_channel,
                       opt.list_dim_channels_encoding1, opt.use_x, opt.cross_connection, opt.pooling_operation, opt.list_dim_channels_encoding2, opt.intermediate_size_fc,
                       opt.dropout_rate, opt.nb_interpolating_points, opt.use_x_complete_unsampled, opt.list_dim_channels_decoding, opt.num_classes, opt.num_parts,
                       opt.dropout_rate_cross, opt.nb_interpolating_points_encoding).to(device)
        network.add_cross_connection(opt.batch_size, opt.nb_interpolating_points_crossconnection)
        network = network.to(device)
    elif (opt.network == 'convPN') or (opt.network == 'deepConvPN'):
        # convPN
        network = convPN(opt.batch_size, opt.nb_subsampled_points, opt.nb_neighbours, opt.sampling_method, opt.patch_radius, opt.in_channel_x_complete, opt.in_channel,
                         opt.list_dim_channels_encoding, opt.use_x, opt.use_crosslinks, opt.use_reslinks, opt.sequence, opt.pooling_operation, opt.residuallinks_input,
                         opt.residuallinks_output, opt.intermediate_size_fc, opt.dropout_rate, opt.nb_interpolating_points, opt.use_x_complete_unsampled,
                         opt.list_dim_channels_decoding, opt.num_classes, opt.num_parts, opt.blockout_rate, test=False).to(device)
    if opt.refine != '':
        network.load_state_dict(torch.load(opt.refine, map_location=lambda storage, loc: storage))
    num_parameters = np.sum([np.prod(parameter.shape) for parameter in network.parameters()])
    print('Number of parameters for ' + opt.network + ': ' + str(num_parameters))
    # Creating the tensorboardX writers
    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    if opt.validation_batch == True:
        test_writer = SummaryWriter(os.path.join(log_dirname, 'validation'))
    # Creating the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=opt.weight_decay, amsgrad=True)
    # Saving parameters
    torch.save(opt, params_filename)
    # Saving description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)
    # Starting the training
    print('Starting the training')
    for epoch in range(opt.nepoch):
        # Updating the learning rate and the batch norm decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(opt.lr * opt.decay_rate**(epoch // opt.milestone_step), opt.lr_clip)
        print('Learning rate: ' + str(optimizer.param_groups[0]['lr']))
        bn_decay = min(1 - opt.bn_init_decay * opt.bn_decay_decay_rate**(epoch // opt.bn_decay_decay_step), opt.bn_decay_clip)
        input_decay = 1 - bn_decay
        print('Batchnorm decay: ' + str(bn_decay))
        # Setting to training mode
        network.train()
        if (opt.network == 'convPN') or (opt.network == 'deepConvPN'):
            network.train_custom()
        # Initializing the metrics variable
        loss_training = 0
        cpt_rolling_average = 0
        accuracy_training = 0
        iou_training = 0
        intersection_training = np.zeros([opt.num_parts])
        union_training = np.zeros([opt.num_parts])
        # Iterating over the batches
        for train_batchind, data in enumerate(train_dataloader, 0):
            # Rebooting the optimizer gradients
            optimizer.zero_grad()
            # Getting the input tensors from the data list
            input_tensor = data[0].type(torch.FloatTensor).to(device)
            vertices = input_tensor[:,:,:3]
            normals = input_tensor[:,:,3:]
            labels_cat = data[1].type(torch.LongTensor).to(device)
            labels_seg = data[2].type(torch.LongTensor).to(device)
            parts_tensor = data[3].type(torch.FloatTensor).to(device)
            zero_tensor = torch.zeros([1]).to(device)
            one_tensor = torch.ones([1]).to(device)
            # Forward pass
            pred = network(vertices, normals, labels_cat, bn_decay_value=input_decay)
            loss = compute_loss(pred=pred, target=labels_seg)
            loss_training = loss_training + loss.detach().cpu().item()            
            batch_accuracy, batch_iou, batch_intersection, batch_union = compute_performance_metrics(labels_cat, labels_seg, pred, None, parts_tensor, zero_tensor, one_tensor)
            cpt_rolling_average += 1
            accuracy_training += batch_accuracy
            iou_training += batch_iou
            intersection_training = intersection_training + batch_intersection.detach().cpu().numpy()
            union_training = union_training + batch_union.detach().cpu().numpy()
            # Backward pass
            loss.backward()
            optimizer.step()
            if (train_batchind % opt.nb_rolling_iterations == 0) or (train_batchind == len(train_dataloader) - 1):
                loss_training /= cpt_rolling_average
                accuracy_training /= cpt_rolling_average
                iou_training /= cpt_rolling_average
                partiou_training = np.nanmean(intersection_training / union_training)
                print('[%s %d - %d / %d] %s Loss: %f' % (opt.name, epoch, train_batchind+1, len(train_dataloader), green('training'), loss_training))
                print('[%s %d - %d / %d] %s Accuracy: %f' % (opt.name, epoch, train_batchind+1, len(train_dataloader), green('training'), accuracy_training))
                print('[%s %d - %d / %d] %s IoU: %f' % (opt.name, epoch, train_batchind+1, len(train_dataloader), green('training'), iou_training))
                print('[%s %d - %d / %d] %s PartIoU: %f' % (opt.name, epoch, train_batchind+1, len(train_dataloader), green('training'), partiou_training))
                train_writer.add_scalar('Loss', loss_training, len(train_dataloader)*epoch + (train_batchind+1))
                train_writer.add_scalar('Accuracy', accuracy_training, len(train_dataloader)*epoch + (train_batchind+1))
                train_writer.add_scalar('IoU', iou_training, len(train_dataloader)*epoch + (train_batchind+1))
                train_writer.add_scalar('PartIoU', partiou_training, len(train_dataloader)*epoch + (train_batchind+1))
                # Rebooting the rolling variables
                cpt_rolling_average = 0
                loss_training = 0
                accuracy_training = 0
                iou_training = 0
                intersection_training = 0
                union_training = 0
        if (opt.validation_batch == True) and ((epoch % opt.nb_rolling_iterations == 0) or (epoch == opt.nepoch - 1)):
            loss_validation, accuracy_validation, iou_validation, partiou_validation = validation_epoch(opt, network, test_dataloader, list_num_parts, device)
            print('[%s %d - %d / %d] %s Loss: %f' % (opt.name, epoch, len(train_dataloader), len(train_dataloader), blue('validation'), loss_validation))
            print('[%s %d - %d / %d] %s Accuracy: %f' % (opt.name, epoch, len(train_dataloader), len(train_dataloader), blue('validation'), accuracy_validation))
            print('[%s %d - %d / %d] %s IoU: %f' % (opt.name, epoch, len(train_dataloader), len(train_dataloader), blue('validation'), iou_validation))
            print('[%s %d - %d / %d] %s PartIoU: %f' % (opt.name, epoch, len(train_dataloader), len(train_dataloader), blue('validation'), partiou_validation))
            test_writer.add_scalar('Loss', loss_validation, len(train_dataloader) * (epoch + 1))
            test_writer.add_scalar('Accuracy', accuracy_validation, len(train_dataloader) * (epoch + 1))
            test_writer.add_scalar('IoU', iou_validation, len(train_dataloader) * (epoch + 1))
            test_writer.add_scalar('PartIoU', partiou_validation, len(train_dataloader) * (epoch + 1))
        if (epoch % opt.nb_rolling_iterations == 0) or (epoch == opt.nepoch - 1):
            torch.save(network.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))

def validation_epoch(opt, network, test_dataloader, list_num_parts, device):
    # Setting to evalution mode
    network.eval()
    if (opt.network == 'convPN') or (opt.network == 'deepConvPN'):
        network.eval_custom()
    # Initializing the metrics variable
    loss_validation = 0
    accuracy_validation = 0
    iou_validation = 0
    intersection_validation = np.zeros([opt.num_parts])
    union_validation = np.zeros([opt.num_parts])
    # Iterating over the batches
    for _, data in enumerate(test_dataloader, 0):
        input_tensor = data[0].type(torch.FloatTensor).to(device)
        vertices = input_tensor[:,:,:3]
        normals = input_tensor[:,:,3:]
        labels_cat = data[1].type(torch.LongTensor).to(device)
        labels_seg = data[2].type(torch.LongTensor).to(device)
        parts_tensor = data[3].type(torch.FloatTensor).to(device)
        zero_tensor = torch.zeros([1]).to(device)
        one_tensor = torch.ones([1]).to(device)
        with torch.no_grad():
            pred = network(vertices, normals, labels_cat, bn_decay_value=None)
            loss = compute_loss(pred=pred, target=labels_seg)
            loss_validation = loss_validation + loss.detach().cpu().item()
            batch_accuracy, batch_iou, batch_intersection, batch_union = compute_performance_metrics(labels_cat, labels_seg, pred, None, parts_tensor, zero_tensor, one_tensor)
            accuracy_validation += batch_accuracy
            iou_validation += batch_iou
            intersection_validation = intersection_validation + batch_intersection.detach().cpu().numpy()
            union_validation = union_validation + batch_union.detach().cpu().numpy()
    loss_validation /= len(test_dataloader)
    accuracy_validation /= len(test_dataloader)
    iou_validation /= len(test_dataloader)
    partiou_validation = np.nanmean(intersection_validation / union_validation)
    return loss_validation, accuracy_validation, iou_validation, partiou_validation

if __name__ == '__main__':
    configs = ShapeNetPartOptions()
    opt = configs.parse()
    train_shapenetpart(opt)