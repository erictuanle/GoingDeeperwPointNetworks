import os
import torch
import argparse

#######################################################################################################################
# This script is based on the CycleGan & Pix2Pix code repository
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#######################################################################################################################

class BaseOptions():
    """This class defines general options.
    It implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False
    
    def initialize(self, parser):
        self.initialized = True
        parser.add_argument('--network', type=str, required=True, help='network to use')
        parser.add_argument('--use_GPU', type=lambda x: str(x).lower()=='true', default=True, help='network to use')
        parser.add_argument('--device_idx', type=int, default=0, help='network to use')
        parser.add_argument('--workers', type=int, default=4, help='number of data loading workers - 0 means same thread as main execution')
        parser.add_argument('--logdir', type=str, default='logs', help='training log folder')
        parser.add_argument('--outdir', type=str, default='models', help='output folder (trained models)')
        parser.add_argument('--resdir', type=str, default='results', help='results folder')
        parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
        parser.add_argument('--center_points', type=lambda x: str(x).lower()=='true', default=True, help='center point cloud at mean or not')
        parser.add_argument('--use_pca', type=lambda x: str(x).lower()=='true', default=False, help='whether to give inputs in local PCA coordinate frame')
        parser.add_argument('--identical_epochs', type=lambda x: str(x).lower()=='true', default=False, help='use same order for each epoch, mainly for debugging')
        parser.add_argument('--validation_batch', type=lambda x: str(x).lower()=='true', default=True, help='testing the result on a validation set at each few epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--decay_rate', type=float, default=0.7, help='decay rate')
        parser.add_argument('--milestone_step', type=float, default=20, help='miletone step for learning rate decay')
        parser.add_argument('--lr_clip', type=float, default=10**(-5), help='minimum learning rate')
        parser.add_argument('--bn_init_decay', type=float, default=0.5, help='batch norm initial decay')
        parser.add_argument('--bn_decay_decay_step', type=float, default=20, help='batch norm decay decay rate')
        parser.add_argument('--bn_decay_decay_rate', type=float, default=0.5, help='batch norm decay decay rate')
        parser.add_argument('--bn_decay_clip', type=float, default=0.99, help='minimum batch norm decay')
        parser.add_argument('--nepoch', type=int, default=201, help='number of epochs to train for')
        parser.add_argument('--nb_rolling_iterations', type=int, default=5, help='number of training epochs to wait for to run an evaluation epoch')
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once)"""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        if not self.initialized:  # check if it has been initialized
            parser = self.initialize(parser)
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt