

### Configuration
Dependencies: \
python>=3.6.8 \
torch>=1.7.0 


### Data

##### [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
For each dataset we manually take the first half of classes as positive class and last half of classes as negative class. \
**imb_ratio** = the number samples in positive classes / the number of samples in negative classes \
The imb_ratio is 0.02 for both datasets. To construct highly imbalanced data, we remove 98% of the positive images from the training data and keep the test data unchanged.\
The indexes can be found in **imbalaced_cifar.py** 
##### [Melanoma](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)
The Melanoma dataset is from a medical image Kaggle competition, which serves as a natural real imbalanced image dataset. It contains 33,126 labeled medical images, \
among which 584 images are related to malignant melanoma and labelled as positive samples. \
We manually split the training data into train/validation/test set at 80%/10%/10% ratio and report the achieved AUPRC on test results. 
The split csv are provided in: **./data/melanoma/{train, test, valid}_split.csv**




### Model and Optimizer
Arch: ResNet18, ResNet34 for all three datasets. \
Optimizer: SGD with Momentum 0.9



### Algorithm and Pretrained Models
Proposed loss and the SOAP algorithm is implemented in SOAP.py \
SOAP algorithm with **squared hinge (sqh)** surrogate loss are trained from ce_pretrained models. \
The **pretrained models** for **CIFAR10, CIFAR100, Melanoma** data are provided in https://drive.google.com/drive/folders/13Bxt0eLeOKNEPbwbq1oEeOLNo9AhnQvr?usp=sharing \
Unzip the downloaded models to: \
**cepretainmodels/**:
-  cifar10_resnet18_002.ckpt
-  cifar10_resnet34_002.ckpt
-  cifar100_resnet18_002.ckpt
-  cifar100_resnet34_002.ckpt
-  melanoma_ce_pretrain_resnet18.pth
-  melanoma_ce_pretrain_resnet34.pth


**config_cifar.py**: The hyperparameter configurations for CIFAR10, CIFAR100 \
**config_melanoma.py**: The hyperparameter configuration for Melanoma \



### The hyperparameters for SOAPLOSS:
  --**threshold**: the margin in squared hinge loss | **conf['loss_param']['threshold'] = 0.6** \
  --**batch_size**: batch size | **conf['batch_size'] = 64** \
  --**data_length**: length of the dataset \
  --**loss_type**: squared hinge surrogate loss for SOAP | **conf['surr_loss'] = 'sqh'** \
  --**gamma**:  gamma parameter in the paper | mv_gamma = {0.9, 0.99}

**conf['ft_mode']** = 'fc_random': Reinitializing the Fully-Connected layer for the pretrained model when starting training SOAP.
**conf['pre_train']** = { None : training from scratch,
                      'path_of_pretrained_model': training from a pretrained model }
**conf['posNum']** : Number of positive samples per batch, \{1,2,3,4,5\}




### Results
To replicate the SOAP results for CIFAR10, CIFAR100, Melanoma
```python
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_cifar10_resnet18.py # ResNet18, CIFAR10
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_cifar10_resnet34.py # ResNet34, CIFAR10
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_cifar100_resnet18.py # ResNet18, CIFAR100
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_cifar100_resnet34.py # ResNet34, CIFAR100
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_melanoma_resnet18.py # ResNet18, Melanoma
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_melanoma_resnet34.py # ResNet34, Melanoma
```

The wrapped package can be found in https://github.com/Optimization-AI/LibAUC/
with the following installation and cases command:
```python
pip install libauc
>>> #import library
>>> from libauc.losses import APLoss_SH
>>> from libauc.optimizers import SOAP_SGD, SOAP_ADAM
...
>>> #define loss
>>> Loss = APLoss_SH()
>>> optimizer = SOAP_ADAM()
...
>>> #training
>>> model.train()
>>> for index, data, targets in trainloader:
        data, targets  = data.cuda(), targets.cuda()
        logits = model(data)
	    preds = torch.sigmoid(logits)
        loss = Loss(preds, targets, index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```








