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

# melanoma
conf['epochs'] = 100
conf['early_stopping'] = 50
conf['lr'] = 0.0001
conf['lr_decay_factor'] = 0.5
conf['lr_decay_step_size'] = 100
conf['dropout'] = 0
conf['weight_decay'] = 1e-5
conf['batch_size'] = 64
conf['loss_type'] = 'ce'
conf['loss_param'] = None
conf['ft_mode'] = None
conf['pre_train'] = None


######################################################################################################################
# Settings for val/test
##    'vt_batch_size': val/test batch_size
######################################################################################################################
conf['vt_batch_size'] = 64
