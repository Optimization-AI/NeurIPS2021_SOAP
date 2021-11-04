#! /usr/bin/env python
# -*-coding=utf-8 -*-
__author__ = 'Qi'
# Created by on 7/3/21.

from torchvision.models import resnet
from preprocess import *
from utils import *
from config_melanoma import conf
from train_eval_melanoma import *
import os


if 'amax' in os.uname()[1]:
    data_path = '/data/qiuzh/melanoma/jpeg/'
elif 'optimus' in os.uname()[1]:
    data_path = '/optimus_data/backed_up/qqi7/melanoma/data/melanoma/'



train_dataset = MyImageFolder(data_path + 'mytrain', get_transform(input_size=conf['input_size'], augment=True))
val_dataset = ImageFolder(data_path + 'myval', get_transform(input_size=conf['input_size'], augment=False))
test_dataset = ImageFolder( data_path + 'mytest', get_transform(input_size=conf['input_size'], augment=False))

model = resnet34()
model_name = 'resnet34'

for conf['batch_size'] in [64]:
    for i in range(1,3):
        for loss_type in ['auprc_lang', 'wce']: #  'ldam', 'wce','focal', 'auroc2', # ['focal', 'smoothAP', 'fastAP']

            print(i)
            conf['ft_mode'] = 'fc_random'

            if 'amax' in os.uname()[1]:
                conf['pre_train'] = './cepretrainmodels/YZ_melanoma_ce_pretrain_' + model_name + '.pth'
            elif 'optimus' in os.uname()[1]:
                conf['pre_train'] = './cepretrainmodels/YZ_melanoma_ce_pretrain_' + model_name + '.pth'  # last.ckpt

            conf['lr'] = 1e-4
            conf['epochs'] = 100
            tau = 1
            conf['posNum'] = 1
            conf['th'] = 5

            if 'amax' in os.uname()[1]:
                out_path = '/data/qiuzh/qiqi_res/{}/melanoma/YZAdam_results_{}_bth_{}_epoch_{}_lr_{}_ft_mode_{}_th_{}_tau_{}_posNum_{}'.format(
                    model_name, loss_type, conf['batch_size'], conf['epochs'], conf['lr'], conf['ft_mode'], conf['th'], tau,
                    conf['posNum'])
            elif 'optimus' in os.uname()[1]:
                out_path = './Released_results/{}/melanoma/YZAdam_results_{}_bth_{}_epoch_{}_lr_{}_ft_mode_{}_th_{}_tau_{}_posNum_{}'.format(
                    model_name, loss_type, conf['batch_size'], conf['epochs'], conf['lr'], conf['ft_mode'], conf['th'], tau,
                    conf['posNum'])


            if not os.path.exists(out_path):
                os.makedirs(out_path)
            conf['loss_type'] = loss_type
            conf['loss_param'] = {'threshold': conf['th'], 'm':5, 'gamma':1000}

            print(conf)
            print("posNum: ", conf['posNum'], ' option: ', 1)
            mv_gamma = 0.999
            bins = 5
            run_classification(i, train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path,
                           bins = bins, tau = tau, posNum = conf['posNum'], dataset = 'melanoma', mv_gamma = mv_gamma)
            print(mv_gamma, conf)
