import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
import numpy as np
from torchvision.models import resnet
from preprocess import *
from utils import *
from config_cifar import conf
from train_eval import *
from imbalanced_cifar import *
from torchvision import datasets
# train_dataset = ('/mnt/dive/shared/lyz/auprc_img/datasets/train', get_transform(input_size=conf['input_size'], augment=True))
# val_dataset = ImageFolder('/mnt/dive/shared/lyz/auprc_img/datasets/valid', get_transform(input_size=conf['input_size'], augment=False))
# test_dataset = ImageFolder('/mnt/dive/shared/lyz/auprc_img/datasets/test', get_transform(input_size=conf['input_size'], augment=False))


imb_ratio = 0.02
train_dataset = IMBALANCECIFAR100(root='./data', download=True, transform = transform_train, imb_factor=imb_ratio )
val_dataset = IMBALANCECIFAR100(root='./data', download=True, transform=transform_train, imb_factor= imb_ratio, val = True)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)



model = resnet18()
model_name = 'resnet18'


if imb_ratio == 0.02:
    apdix = '002'
elif imb_ratio == 0.2:
    apdix = '020'


for conf['batch_size'] in [64]:
     for i in range(3):
         for loss_type in ['SOAP']:
             conf['epochs'] = 64
             conf['ft_mode'] = 'fc_random'
             conf['lr'] = 1e-5
             conf['pre_train'] = './cepretrainmodels/cifar100_' + model_name + '_' + apdix + '.ckpt' # imb_factor 0.02
             conf['surr_loss'] = 'sqh'
             tau = 1
             conf['posNum'] = 2
             th = 0.6
             gamma1 = 0.999
             gamma2 = 0.999

             out_path = './Released_results/{}/cifar100/SGD_results_{}_bth_{}_epoch_{}_lr_{}_ft_mode_{}_tau_{}_posNum_{}_threshold_{}_repeats_{}_imb_{}_surrloss_{}_gamma_{}'.format(model_name, loss_type,  conf['batch_size'], conf['epochs'], conf['lr'], conf['ft_mode'],tau, conf['posNum'], th, i, imb_ratio, conf['surr_loss'], str(gamma1))
             if not os.path.exists(out_path):
                 os.makedirs(out_path)
             conf['loss_type'] = loss_type
             conf['loss_param'] = {'threshold': th, 'm':5, 'gamma':1000}
             print(conf)
             print(i)
             bins = 2
             mv_gamma =  0.999
             run_classification(i, train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path,
                            bins = bins, tau = tau, posNum = conf['posNum'], surr_loss= conf['surr_loss'], dataset = 'cifar100', mv_gamma = mv_gamma)
             print(mv_gamma, conf)
             print(i)
