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
from realtime_render import *

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
        best_mean_ucd = 1e4
        best_uhd = 1e4
        best_mean_cd = 1e4
        curr_step = 0

        # mean loss for each class, updated on each batch
        test_real_cd_loss_mean_array = np.ones(self.args.num_class) * 1e4
        test_real_ucd_loss_mean_array = np.ones(self.args.num_class) * 1e4
        test_real_uhd_loss_mean_array = np.ones(self.args.num_class) * 1e4
        
        virtual_train_dataloader_iter_list = [iter(it) for it in self.virtual_train_dataloader_list]
        real_train_dataloader_iter_list = [iter(it) for it in self.real_train_dataloader_list]
        virtual_train_domain_dataloader_iter_list = [iter(it) for it in self.virtual_train_dataloader_list]
        real_train_domain_dataloader_iter_list = [iter(it) for it in self.real_train_dataloader_list]

        for epoch in range(self.args.epochs * self.args.num_class):
            class_idx = epoch % self.args.num_class
            print("##########EPOCH {:0>4d}##########".format(epoch))
            with open(self.args.log_pathname, "a") as file_object:
                msg =  "##########EPOCH {:0>4d}##########".format(epoch)
                file_object.write(msg+'\n')
            bool_virtual_train = True
            bool_real_train = True
            bool_domain_train = True
            if epoch < 200 * self.args.num_class and epoch >= 120 * self.args.num_class:
                bool_cons_train = True
            else:
                bool_cons_train = False
            bool_virtual_test = True
            bool_real_test = True
            train_cd_loss_list = []
            train_ucd_loss_list = []
            train_di_loss_list = []
            train_ds_loss_list = []
            train_vp_loss_list = []
            train_cs_loss_list = []
            tic = time.time()
            for tmp_i in range(144):    #145 iterations
                ###Train on virtual scans
                if bool_virtual_train:
                    virtual_train_iter_times = 2
                    for i in range(virtual_train_iter_times):
                        curr_step += 1
                        try:
                            data = next(virtual_train_dataloader_iter_list[class_idx])
                        except StopIteration:
                            virtual_train_dataloader_iter_list[class_idx] = iter(self.virtual_train_dataloader_list[class_idx])
                            data = next(virtual_train_dataloader_iter_list[class_idx])
                        # with gt
                        gt, partial, index = data
                        gt = gt.squeeze(0).cuda()
                        partial = partial.squeeze(0).cuda()
                        partial, _, _, azel_batch = partial_render_batch(gt, partial)

                        self.model.reset_G_tmp()
                        self.model.pcd_id = index[0].item()

                        # set gt and partial shape 
                        self.model.set_target(gt=gt, partial=partial)
                        
                        # train one batch
                        train_cd_loss = self.model.train_virtual_one_batch(class_idx, curr_step)
                        train_cd_loss_list.append(train_cd_loss)

                #Train for domain invariant feature and domain specific feature
                if bool_domain_train:
                    domain_train_iter_times = 1
                    for i in range(domain_train_iter_times):
                        try:
                            virtual_data = next(virtual_train_domain_dataloader_iter_list[class_idx])
                        except StopIteration:
                            virtual_train_domain_dataloader_iter_list[class_idx] = iter(self.virtual_train_dataloader_list[class_idx])
                            virtual_data = next(virtual_train_domain_dataloader_iter_list[class_idx])
                        try:
                            real_data = next(real_train_domain_dataloader_iter_list[class_idx]) 
                        except StopIteration:
                            real_train_domain_dataloader_iter_list[class_idx] = iter(self.real_train_dataloader_list[class_idx])
                            real_data = next(real_train_domain_dataloader_iter_list[class_idx])
                        virtual_gt, virtual_partial, virtual_index = virtual_data
                        random_idx = np.random.choice(virtual_partial.shape[0],10)
                        if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                            real_gt, real_partial, real_index = real_data
                        else:
                            real_partial, real_index = real_data
                        random_idx = np.random.choice(real_partial.shape[0],10)
                        real_partial = real_partial[random_idx]
                        virtual_gt = virtual_gt.cuda()
                        virtual_partial = virtual_partial.cuda()
                        virtual_partial, rotmat_az_batch, rotmat_el_batch, azel_batch = partial_render_batch(virtual_gt, virtual_partial)
                        rotmat_batch = np.matmul(rotmat_az_batch, rotmat_el_batch)
                        rotmat_batch = rotmat_batch.transpose((0, 2, 1))
                        rotmat_batch = torch.Tensor(rotmat_batch).float().cuda()
                        azel_batch = torch.Tensor(azel_batch).float().cuda()
                        real_partial = real_partial.cuda()
                        self.model.reset_G_tmp()
                        self.model.pcd_id = index[0].item()
                        self.model.set_virtual_real(virtual_partial=virtual_partial, real_partial=real_partial, rotmat=rotmat_batch, azel=azel_batch)
                        p = epoch / float(self.args.epochs)
                        alpha = 2./(1.+np.exp(-10.*p))-1.
                        alpha = min(alpha, 0.462)
                        # train domain one batch
                        di_loss, ds_loss, vp_loss, cons_feature = self.model.train_domain_one_batch(class_idx, curr_step, alpha)
                        if bool_cons_train:
                            cs_loss = self.model.train_consistency_one_batch(class_idx, curr_step, cons_feature)
                        else:
                            cs_loss = 0.
                        train_di_loss_list.append(di_loss)
                        train_ds_loss_list.append(ds_loss)
                        train_vp_loss_list.append(vp_loss)
                        train_cs_loss_list.append(cs_loss)
                
                ###Train on real scans
                if bool_real_train:
                    real_train_iter_times = 1
                    for i in range(real_train_iter_times):
                        curr_step += 1
                        try:
                            data = next(real_train_dataloader_iter_list[class_idx])
                        except StopIteration:
                            real_train_dataloader_iter_list[class_idx] = iter(self.real_train_dataloader_list[class_idx])
                            data = next(real_train_dataloader_iter_list[class_idx])
                        # with gt
                        if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                            partial, index = data
                        elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                            gt, partial, index = data
                        partial = partial.squeeze(0).float().cuda()

                        self.model.reset_G_tmp()
                        self.model.pcd_id = index[0].item()

                        # set partial shape 
                        self.model.set_target(partial=partial)
                        
                        # train real one batch
                        train_ucd_loss = self.model.train_real_one_batch(class_idx, curr_step, epoch)
                        train_ucd_loss_list.append(train_ucd_loss)
            ###
            toc = time.time()

            print('>>>class', class_idx,'train done in ',int(toc-tic),'s')
            if bool_virtual_train:
                train_cd_loss_mean = np.mean(np.array(train_cd_loss_list))
                print("Mean Chamfer Distance on Training Set:", train_cd_loss_mean)
            if bool_real_train:
                train_ucd_loss_mean = np.mean(np.array(train_ucd_loss_list))
                print("Mean UCD on Real Training Set:", train_ucd_loss_mean)
            if bool_domain_train:
                train_di_loss_mean = np.mean(np.array(train_di_loss_list))
                train_ds_loss_mean = np.mean(np.array(train_ds_loss_list))
                train_vp_loss_mean = np.mean(np.array(train_vp_loss_list))
                train_cs_loss_mean = np.mean(np.array(train_cs_loss_list))
                print("Mean Loss on Real Training Set DI:", train_di_loss_mean, "DS:", train_ds_loss_mean, "VP:", train_vp_loss_mean, "CS:", train_cs_loss_mean)
            with open(self.args.log_pathname, "a") as file_object:
                msg = '>>>'+'class' + str(class_idx) + 'train done in ' + str(int(toc-tic)) +'s'
                file_object.write(msg+'\n')

                if bool_virtual_train:
                    msg =  "Mean Chamfer Distance on Training Set:" + "%.8f"%train_cd_loss_mean
                    file_object.write(msg+'\n')
                if bool_real_train:
                    msg =  "Mean UCD on Real Training Set:" + "%.8f"%train_ucd_loss_mean
                    file_object.write(msg+'\n')
                if bool_domain_train:
                    msg =  "Mean Loss on Training Set DI:" + "%.8f"%train_di_loss_mean + "DS:" + "%.8f"%train_ds_loss_mean + "VP:" + "%.8f"%train_vp_loss_mean + "CS:" + "%.8f"%train_cs_loss_mean
                    file_object.write(msg+'\n')

            ###Test on virtual test set
            if bool_virtual_test:
                test_cd_loss_list = []
                tic = time.time()
                for i, data in enumerate(self.virtual_test_dataloader_list[class_idx]):
                    # with gt
                    gt, partial, index = data
                    gt = gt.squeeze(0).cuda()
                    partial = partial.squeeze(0).cuda()

                    self.model.reset_G_tmp()
                    self.model.pcd_id = index[0].item()

                    # set gt and partial shape 
                    self.model.set_target(gt=gt, partial=partial)
                    
                    # test one batch 
                    test_cd_loss = self.model.test_virtual_one_batch(class_idx=class_idx)
                    test_cd_loss_list.append(test_cd_loss)
                toc = time.time()

                print('>>>class', class_idx,'virtual test done in ',int(toc-tic),'s')
                test_cd_loss_mean = np.mean(np.array(test_cd_loss_list))
                print("Mean Chamfer Distance on Test Set:", test_cd_loss_mean)
                with open(self.args.log_pathname, "a") as file_object:
                    msg = '>>>'+'class' + str(class_idx) + 'virtual test done in ' + str(int(toc-tic)) +'s'
                    file_object.write(msg+'\n')
                    msg =  "Mean Chamfer Distance on Test Set:" + "%.8f"%test_cd_loss_mean
                    file_object.write(msg+'\n')
            ### test on real test set
            if bool_real_test:
                test_real_ucd_loss_list = []
                test_real_uhd_loss_list = []
                test_real_cd_loss_list = []
                tic = time.time()
                for i, data in enumerate(self.real_test_dataloader_list[class_idx]):
                    # with gt
                    if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                        partial, index = data
                    elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                        gt, partial, index = data
                        gt = gt.squeeze(0).float().cuda()
                    partial = partial.squeeze(0).float().cuda()

                    self.model.reset_G_tmp()
                    self.model.pcd_id = index[0].item()

                    # set partial or gt and partial shape 
                    if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                        self.model.set_target(partial=partial)
                    elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                        self.model.set_target(gt=gt, partial=partial)
                    
                    # test real one batch
                    if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                        test_real_ucd_loss, test_real_uhd_loss = self.model.test_real_one_batch(class_idx=class_idx)
                    elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                        test_real_ucd_loss, test_real_uhd_loss, test_real_cd_loss = self.model.test_real_one_batch(class_idx=class_idx, bool_gt=True)
                        test_real_cd_loss_list.append(test_real_cd_loss)
                    test_real_ucd_loss_list.append(test_real_ucd_loss)
                    test_real_uhd_loss_list.append(test_real_uhd_loss)

                toc = time.time()
                print('>>>class', class_idx, 'real test done in ',int(toc-tic),'s')
                
                if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    test_real_cd_loss_mean = np.mean(np.array(test_real_cd_loss_list))
                    test_real_cd_loss_mean_array[epoch % self.args.num_class] = test_real_cd_loss_mean

                test_real_ucd_loss_mean = np.mean(np.array(test_real_ucd_loss_list))
                test_real_ucd_loss_mean_array[epoch % self.args.num_class] = test_real_ucd_loss_mean
                test_real_uhd_loss_mean = np.mean(np.array(test_real_uhd_loss_list))
                
                # choose model that has the lowest mean ucd/cd loss on class average
                if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                    if test_real_ucd_loss_mean_array.mean() < best_mean_ucd:
                        best_mean_ucd = test_real_ucd_loss_mean_array.mean()
                        print('class average best_mean_ucd = ', best_mean_ucd) 
                        print('test_real_ucd_loss_mean_array = ', test_real_ucd_loss_mean_array)
                        os.system("mv " + './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/saved_results/*' + ' ./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/best_results/')
                        #-------- Save checkpoint --------#
                        self.model.save_checkpoint(self.args.ckpt_path_name)
                elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    if test_real_cd_loss_mean_array.mean() < best_mean_cd:
                        best_mean_cd = test_real_cd_loss_mean_array.mean()
                        print('class average best_mean_cd = ', best_mean_cd)
                        print('test_real_cd_loss_mean_array = ', test_real_cd_loss_mean_array)
                        os.system("mv " + './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/saved_results/*' + ' ./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/best_results/')
                        #-------- Save checkpoint --------#
                        self.model.save_checkpoint(self.args.ckpt_path_name)

                if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                    print("Mean CD on Real Test Set:", test_real_cd_loss_mean)
                print("Mean UCD on Real Test Set:", test_real_ucd_loss_mean)
                print("Mean UHD on Real Test Set:", test_real_uhd_loss_mean)
                with open(self.args.log_pathname, "a") as file_object:
                    msg = '>>>'+'class' + str(class_idx) + 'real test done in ' + str(int(toc-tic)) +'s'
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
        script_name = 'multiclass_train_' + args.virtualdataset + '_' + args.realdataset + '.sh'
        os.system('cp %s %s'% ('scripts/' + script_name, './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('multiclass_trainer.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('multiclass_optde.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('model/network.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/ply_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/real_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
    

    args.save_inversion_path += '/' + time_stamp[:-1]
    args.ckpt_path_name = args.save_inversion_path + '/' + 'multiclass.pt'
    args.save_inversion_path += '/' + 'saved_results'
    trainer = Trainer(args)
    trainer.train()
