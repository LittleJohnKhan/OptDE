import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset


from arguments import Arguments

from loss import *

# from optde import OptDE 
from multiclass_optde import OptDE 

from model.network import Generator, Discriminator
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

import random

import numpy as np

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

time_stamp = time.strftime('Log_%Y-%m-%d_%H-%M-%S/', time.gmtime())

class Trainer(object):

    def __init__(self, args):
        '''
        load virtual/real train/test data
        '''
        self.args = args
        p2c_batch_size = 10
        test_batch_size = 1
        
        save_inversion_dirname = args.save_inversion_path.split('/')
        log_pathname = './'+args.log_dir+'/'+ save_inversion_dirname[-3] + '/' + save_inversion_dirname[-2] + '/log.txt'
        args.log_pathname = log_pathname

        visual_class_list=args.visual_class_choices.split(',')
        real_class_list=args.real_class_choices.split(',')
        self.virtual_train_dataloader_list = []
        self.virtual_test_dataloader_list = []
        self.real_train_dataloader_list = []
        self.real_test_dataloader_list = []

        # Define the dataset-class mapping
        dataset_classes = {
            'MatterPort': PlyDataset,
            'ScanNet': PlyDataset,
            'KITTI': PlyDataset,
            'PartNet': PlyDataset,
            'ModelNet': GeneratedDataset,
            '3D_FUTURE': GeneratedDataset,
            'CRN': CRNShapeNet
        }

        ###* create model
        self.model = OptDE(self.args)
        
        ###*Load Virtual Train Data
        self.virtual_data_name = self.args.virtualdataset
        self.args.dataset = self.virtual_data_name
        if self.virtual_data_name in ['ScanNet', 'MatterPort']:
            self.args.split = 'trainval'
        elif self.virtual_data_name in ['ModelNet', '3D_FUTURE', 'KITTI', 'CRN']:
            self.args.split = 'train'
        # Loop through the dataset-class mapping and construct the datasets and dataloaders
        for dataset_name, dataset_class in dataset_classes.items():
            if self.virtual_data_name == dataset_name:
                for cla in visual_class_list:
                    self.args.class_choice = cla
                    virtual_train_dataset = dataset_class(self.args)
                    self.virtual_train_dataloader_list.append(DataLoader(
                        virtual_train_dataset,
                        batch_size=p2c_batch_size,
                        shuffle=False,
                        pin_memory=True))
        
        ###*Load Virtual Test Data
        self.args.split = 'test'
        for dataset_name, dataset_class in dataset_classes.items():
            if self.virtual_data_name == dataset_name:
                for cla in visual_class_list:
                    self.args.class_choice = cla
                    virtual_test_dataset = dataset_class(self.args)
                    self.virtual_test_dataloader_list.append(DataLoader(
                        virtual_test_dataset,
                        batch_size=test_batch_size,
                        shuffle=False,
                        pin_memory=True))

        ###*Load Real Train Data
        self.real_data_name = self.args.realdataset
        self.args.dataset = self.real_data_name
        if self.real_data_name in ['ScanNet', 'MatterPort']:
            self.args.split = 'trainval'
        elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'KITTI', 'CRN']:
            self.args.split = 'train'
        for dataset_name, dataset_class in dataset_classes.items():
            if self.real_data_name == dataset_name:
                for cla in real_class_list:
                    self.args.class_choice = cla
                    real_train_dataset = dataset_class(self.args)
                    self.real_train_dataloader_list.append(DataLoader(
                        real_train_dataset,
                        batch_size=p2c_batch_size,
                        shuffle=False,
                        pin_memory=True))    
        ###*Load Real Test Data
        self.args.split = 'test'
        for dataset_name, dataset_class in dataset_classes.items():
            if self.real_data_name == dataset_name:
                for cla in real_class_list:
                    self.args.class_choice = cla
                    real_test_dataset = dataset_class(self.args)
                    self.real_test_dataloader_list.append(DataLoader(
                        real_test_dataset,
                        batch_size=test_batch_size,
                        shuffle=False,
                        pin_memory=True))

        
    def train(self):
        load_path_name = self.args.finetune_ckpt_load
        print(load_path_name)

        for class_idx in range(self.args.num_class):
            test_real_ucd_loss_list = []
            test_real_uhd_loss_list = []
            test_real_cd_loss_list = []
            
            # finetune each class
            tic = time.time()
            for i, data in enumerate(self.real_test_dataloader_list[class_idx]):
                # with gt
                if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                    partial, index = data
                elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    gt, partial, index = data
                    gt = gt.squeeze(0).float().cuda()
                partial = partial.squeeze(0).float().cuda()

                # reset G for each new input
                #self.model.reset_G(pcd_id=index.item())
                self.model.reset_G_tmp()
                self.model.pcd_id = index[0].item()

                # set target and complete shape 
                # for ['reconstruction', 'jittering', 'morphing'], GT is used for reconstruction
                # else, GT is not involved for training
                if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                    self.model.set_target(partial=partial)
                elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    self.model.set_target(gt=gt, partial=partial)
                
                # inversion
                self.model.reset_whole_network(load_path_name)
                if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                    test_real_ucd_loss, test_real_uhd_loss = self.model.finetune(class_idx=class_idx)
                elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    test_real_ucd_loss, test_real_uhd_loss, test_real_cd_loss = self.model.finetune(class_idx=class_idx, bool_gt=True)
                    test_real_cd_loss_list.append(test_real_cd_loss)
                test_real_ucd_loss_list.append(test_real_ucd_loss)
                test_real_uhd_loss_list.append(test_real_uhd_loss)
            toc = time.time()
            
            if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                np.save(self.args.save_inversion_path+'/class_%d_cd_list.npy'%class_idx, np.array(test_real_cd_loss_list))
                test_real_cd_loss_mean = np.mean(np.array(test_real_cd_loss_list))
            test_real_ucd_loss_mean = np.mean(np.array(test_real_ucd_loss_list))
            test_real_uhd_loss_mean = np.mean(np.array(test_real_uhd_loss_list))
            
            print('==================================================================================')
            print('class_idx ', class_idx , ' optimize done in ', int(toc-tic), 's, ', '#samples = ', len(self.real_test_dataloader_list[class_idx]))
            print('----------------------------------------------------------------------------------')
            if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                print("Mean CD on Real Test Set:", test_real_cd_loss_mean)
            print("Mean UCD on Real Test Set:", test_real_ucd_loss_mean)
            print("Mean UHD on Real Test Set:", test_real_uhd_loss_mean)
            with open(self.args.log_pathname, "a") as file_object:
                msg = ">>> " + "class " + str(class_idx) + " optimize done in " + str(int(toc-tic)) + "s, " + "#samples = " + str(len(self.real_test_dataloader_list[class_idx])) + " <<<"
                file_object.write(msg+'\n')
                if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    msg =  "Mean CD on Real Test Set:" + "%.8f"%test_real_cd_loss_mean
                    file_object.write(msg+'\n')
                msg =  "Mean UCD on Real Test Set:" + "%.8f"%test_real_ucd_loss_mean
                file_object.write(msg+'\n')
                msg =  "Mean UHD on Real Test Set:" + "%.8f"%test_real_uhd_loss_mean
                file_object.write(msg+'\n')


if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    
    if not os.path.isdir('./'+args.log_dir+'/'):
        os.mkdir('./'+args.log_dir+'/')
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1]):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1])
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1]):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1])
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/best_results'):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1]+'/best_results')
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code'):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code')
        script_name = 'multiclass_optimize_' + args.virtualdataset + '_' + args.realdataset + '.sh'
        os.system('cp %s %s'% ('scripts/' + script_name, './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('multiclass_optimizer.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('multiclass_optde.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('model/network.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/ply_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/real_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
    

    args.save_inversion_path += '/' + time_stamp[:-1]
    args.ckpt_path_name = args.save_inversion_path + '/' + 'multiclass.pt'
    args.save_inversion_path += '/' + 'saved_results'
    trainer = Trainer(args)
    trainer.train()
