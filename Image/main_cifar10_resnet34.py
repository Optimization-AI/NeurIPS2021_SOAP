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
train_dataset = IMBALANCECIFAR10(root='./data', download=True, transform = transform_train, imb_factor=imb_ratio )
val_dataset = IMBALANCECIFAR10(root='./data', download=True, transform=transform_train, imb_factor= imb_ratio, val = True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)



model = resnet34()
model_name = 'resnet34'


if imb_ratio == 0.02:
    apdix = '002'
elif imb_ratio == 0.2:
    apdix = '020'


np.random.seed(0)
for conf['batch_size'] in [64]:
     for i in range(3):
         for loss_type in ['SOAP']: #, 'auroc2', 'auprc_lang', 'wce', 'ldam', 'focal', 'smoothAP', 'fastAP'
             conf['epochs'] = 64
             conf['ft_mode'] = 'fc_random'
             conf['lr'] = 1e-2
             conf['pre_train'] = './cepretrainmodels/cifar10_' + model_name + '_' + apdix + '.ckpt'  # imb_factor 0.02
             conf['surr_loss'] = 'sqh'
             tau = 1
             posNum = 3
             th = 0.5
             out_path = './Released_results/{}/cifar10/SGD_results_{}_bth_{}_epoch_{}_lr_{}_ft_mode_{}_tau_{}_posNum_{}_threshold_{}_repeats_{}_imb_{}_surrloss_{}'.format(model_name, loss_type,  conf['batch_size'], conf['epochs'], conf['lr'], conf['ft_mode'],tau, posNum, th, i, imb_ratio, conf['surr_loss'])
             if not os.path.exists(out_path):
                 os.makedirs(out_path)
             conf['loss_type'] = loss_type
             conf['loss_param'] = {'threshold': th, 'm':5, 'gamma':1000}
             print(conf)
             print(i, posNum)
             bins = 2
             mv_gamma = 0.99
             run_classification(i, train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path,
                            bins = bins, tau = tau, posNum = posNum, surr_loss= conf['surr_loss'], dataset = 'cifar10', mv_gamma = mv_gamma)

# gamma = 1 1-gamma = 0
# gamma = 0.1 1- gamma = 0.9, 1- 0.05 = 0.95
# gamma = 0.01 1- gamma = 0.99
