

### Configuration
Dependencies:
python>=3.6.8
torch>=1.7.0
torch_geometric==1.6.3
Other packages:
Install necessary packages required in MoleculeKit gnn part, rdkit, pytorch geometrics, descriptastorus. Ensure the rdkit version is 2020.03.3, otherwise the feature extraction may be problematic.
Referenced literature:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html



### Data
Use tran_data.py to get GNN features from .csv data files.


### Model
GINE -- a variant of GIN model

MPNN -- message passing neural network

MLN -- GNN in MoleculeKit

GINE and MPNN are implemented in gine_mpnn.py

MLN is implemented in model.py


### Algorithm and Pretrained Models
Proposed loss and the SOAP algorithm is implemented in SOAP.py
SOAP algorithm with **squared hinge (sqh)** surrogate loss are trained from ce_pretrained models.
The **pretrained models** for **hiv** data are provided in the
**pretrained_models/hiv_pretrianed_model/**:
-  hiv_gine_ce.ckpt
-  hiv_mlpmpnn_ce.ckpt
-  hiv_mpnn_ce.ckpt


The pretrained model is trained using **ce_loss** for 100 epochs using Adam with the following default configurations in **./config/config_hiv.py**:
It has been used for training ce_pretrained model.
conf['epochs'] = 100
conf['early_stopping'] = 50
conf['lr'] = 0.0005
conf['lr_decay_factor'] = 0.5
conf['lr_decay_step_size'] = 50
conf['dropout'] = 0
conf['weight_decay'] = 0.00005
conf['weight_decay'] = 0.00005
conf['depth'] = 3
conf['hidden'] = 32
conf['batch_size'] = 64
conf['loss_type'] = 'ce_loss'
conf['loss_param'] = {'threshold':10}
conf['ft_mode'] = 'fc_random'
conf['pre_train'] = None
conf['vt_batch_size'] = 1000

The default conf['loss_type'] = 'ce_loss' is for standard cross entropy loss for training the pretrained model.



### The hyperparameters for SOAPLOSS:
  --**threshold**: the margin in squared hinge loss | **conf['loss_param']['threshold'] = 10**
  --**batch_size**: batch size | **conf['batch_size'] = 64**
  --**data_length**: length of the dataset
  --**loss_type**: squared hinge surrogate loss for SOAP | **conf['loss_param']['type'] = 'sqh'**
  --**gamma**:  gamma parameter in the paper | **conf['loss_param']['mv_gamma'] = {0.99, 0.9}**

**conf['ft_mode']** = 'fc_random': Reinitializing the Fully-Connected layer for the pretrained model when starting training SOAP.
**conf['pre_train']** = { None : training from scratch,
                      'path_of_pretrained_model': training from a pretrained model }
**conf['posNum']** : Number of positive samples per batch
We use the same conf['posNum'] = 1 both for all the baselines.



### Results
To replicate the SOAP results in Table 2, Run:
```
 CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_hiv.py
 CUDA_VISIBLE_DEVICES=0 python3 -W ignore main_hiv.py
```
 | HIV | Network |       GINE       |       MPNN      |       MLPNN      |
|-----|:-------:|:----------------:|:---------------:|:----------------:|
|     |   SOAP  |0.3462 (0.0083)  | 0.3406 (0.0053) | 0.3646 (0.0076) |

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

### Tips:
If some error happens with the file path, **please adjust the corresponding data file, pretrained model file to your own path**.







