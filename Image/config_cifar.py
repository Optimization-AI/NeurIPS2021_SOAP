__author__ = 'Qi'
# Created by on 9/7/21.
"""
Configuration file
"""



conf = {}
conf['input_size'] = 384
conf['num_tasks'] = 1


######################################################################################################################
# Settings for training
##    'epochs': maximum training epochs
##    'early_stopping': patience used to stop training
##    'lr': starting learning rate
##    'lr_decay_factor': learning rate decay factor
##    'lr_decay_step_size': step size of learning rate decay
##    'dropout': dropout rate
##    'weight_decay': l2 regularizer term
##    'depth': number of layers
##    'batch_size': training batch_size
######################################################################################################################



# cifar10
conf['epochs'] = 64
conf['early_stopping'] = 50
conf['lr'] = 0.001
conf['lr_decay_factor'] = 0.1
conf['lr_decay_step_size'] = 32
conf['dropout'] = 0
conf['weight_decay'] = 2e-4
conf['batch_size'] = 64
conf['loss_type'] = 'auprc2'
conf['loss_param'] = None
conf['ft_mode'] = None
conf['pre_train'] = None



######################################################################################################################
# Settings for val/test
##    'vt_batch_size': val/test batch_size
######################################################################################################################
conf['vt_batch_size'] = 64