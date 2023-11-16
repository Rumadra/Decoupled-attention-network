# coding:utf-8
import torch
import torch.optim as optim
import os
from dataset_hw import *
from DAN import *

global_cfgs = {
    'state': 'Train',
    'epoch': 100,
    'show_interval': 50,
    'test_interval': 500
}

dataset_cfgs = {
    'dataset_train': IAMSynthesisDataset,
    'dataset_train_args': {
        'img_list': 'data/IAM/train_list.txt',
        'img_height': 192,
        'img_width': 2048,
        'augment': True, # with the data augmentation toolkit
    },
    'dataloader_train': {
        'batch_size': 24,
        'shuffle': True,
        'num_workers': 2,
    },

    'dataset_test': IAMDataset,
    'dataset_test_args': {
        'img_list': 'data/IAM/eval_list.txt',
        'img_height': 192,
        'img_width': 2048,
    },
    'dataloader_test': {
        'batch_size': 12,
        'shuffle': False,
        'num_workers': 2,
    },

    'case_sensitive': False,
    'dict_dir': 'dict/dic_techno.txt'
}

net_cfgs = {
    'FE': Feature_Extractor,
    'FE_args': {
        'strides': [(2,2), (2,2), (2,1), (2,2), (2,2), (2,1)],
        # 'strides': [(2,2), (2,2), (2,1), (1,2), (1,1), (1,1)],
        'compress_layer' : True, 
        'input_shape': [1, 192, 2048], # C x H x W
    },
    'CAM': CAM_transposed,
    'CAM_args': {
        # 'maxT': 150,
        'maxT': 30, 
        'depth': 14, 
        'num_channels': 128,
        # 'num_channels': 64,
    },
    'DTD': DTD,
    'DTD_args': {
        'nclass': 1429, # extra 2 classes for Unknown and End-token
        'nchannel': 256,
        'dropout': 0.7,
    },

    # 'init_state_dict_fe': 'models/hw/exp1_E99_I2000-2295_M0.pth',
    # 'init_state_dict_cam': 'models/hw/exp1_E99_I2000-2295_M1.pth',
    # 'init_state_dict_dtd': 'models/hw/exp1_E99_I2000-2295_M2.pth',

    'init_state_dict_fe': None,
    'init_state_dict_cam': None,
    'init_state_dict_dtd': None,
}

optimizer_cfgs = {
    # optim for FE
    'optimizer_0': optim.SGD,
    'optimizer_0_args':{
        'lr': 0.1,
        'momentum': 0.9,
    },

    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [20, 40, 60, 80],
        'gamma': 0.3162,
    },

    # optim for CAM
    'optimizer_1': optim.SGD,
    'optimizer_1_args':{
        'lr': 0.1,
        'momentum': 0.9,
    },
    'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_1_scheduler_args': {
        'milestones': [20, 40, 60, 80],
        'gamma': 0.3162,
    },

    # optim for DTD
    'optimizer_2': optim.SGD,
    'optimizer_2_args':{
        'lr': 0.1,
        'momentum': 0.9,
    },
    'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_2_scheduler_args': {
        'milestones': [20, 40, 60, 80],
        'gamma': 0.3162,
    },
}

saving_cfgs = {
    'saving_iter_interval': 2000,
    'saving_epoch_interval': 3,

    'saving_path': 'models/hw/exp1_',
}

def mkdir(path_):
    if not os.path.exists(path_):
        os.makedirs(path_)

def showcfgs(s):
    for key in s.keys():
        print(key, s[key])
    print('')

mkdir(saving_cfgs['saving_path'])