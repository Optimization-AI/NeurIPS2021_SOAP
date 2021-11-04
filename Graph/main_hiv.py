__author__ = 'Qi'
# Created by on 10/31/21.
from gine_mpnn import NNNet, GINENet
import os
from datasets import *
from train_eval import run_classification
from model import *
from config.config_hiv import conf


def run_main(data_name = 'tox21_t0', arch_list = ['gine', 'mpnn', 'mlpmpnn'], conf = conf):
    conf['dataset'] = data_name
    if conf['dataset'] == 'hiv':
        data_file, split_file = 'datasets/gnn_feats/hiv.pt', 'datasets/split_inds/hivscaffold.pkl'
    elif conf['dataset'] == 'muv':
        data_file, split_file = 'datasets/gnn_feats/muv_2.pt', 'datasets/split_inds/muv_2.pkl'
    elif conf['dataset'] == 'tox21_t0':
        conf['hidden'] = 128
        data_file, split_file = 'datasets/gnn_feats/tox21/tox21_0.pt', 'datasets/split_inds/tox21/tox21_0.pkl'
    elif conf['dataset'] == 'tox21_t2':
        conf['hidden'] = 128
        data_file, split_file = 'datasets/gnn_feats/tox21/tox21_2.pt', 'datasets/split_inds/tox21/tox21_2.pkl'
    elif conf['dataset'] == 'toxcast_t8':
        conf['hidden'] = 64
        data_file, split_file = 'datasets/gnn_feats/toxcast/toxcast_8.pt', 'datasets/split_inds/toxcast/toxcast_8.pkl'
    elif conf['dataset'] == 'toxcast_t12':
        conf['hidden'] = 64
        data_file, split_file = 'datasets/gnn_feats/toxcast/toxcast_12.pt', 'datasets/split_inds/toxcast/toxcast_12.pkl'

    dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(data_file, conf['graph_level_feature'])
    assert conf['num_tasks'] == dataset[0].y.shape[-1]
    train_dataset, val_dataset, test_dataset = split_data(dataset, split_file)



    conf['loss_param'] = {'K':10, 'm':5, 'gamma':1000, 'tau':3.0, 'bins':2, 'mv_gamma':0.9, 'type':'sqh', 'threshold':10}
    conf['ft_mode'] = 'fc_random'
    conf['posNum'] = 1
    model = None
    for j in range(2):
       for method in ['SOAP']: # ['wce', 'focal', 'ldam', 'auroc', 'smoothAP', 'fastAP', 'minmax', 'SOAP']: #['wce', 'focal', 'ldam', 'auroc', 'smoothAP', 'fastAP', 'minmax', 'SOAP']:
    # for j in range(1):
    #     for method in ['ce']: #['wce', 'focal', 'ldam', 'auroc', 'smoothAP', 'fastAP', 'minmax', 'SOAP']:
            conf['loss_type'] = method
            for i in range(len(arch_list)):
                if arch_list[i] == 'mpnn':
                    conf['arch'] = 'mpnn'
                    # conf['pre_train'] = None
                    conf['pre_train'] = './pretrained_models/' + str.split(conf['dataset'], '_')[0] +'_pretrained_model/' +  '_'.join([conf['dataset'], conf['arch'], 'ce.ckpt'])
                    model = NNNet(num_node_features, num_edge_features, conf['hidden'], conf['dropout'], conf['num_tasks'])
                elif  arch_list[i]  == 'mlpmpnn':
                    conf['arch'] = 'mlpmpnn'
                    # conf['pre_train'] = None
                    conf['pre_train'] = './pretrained_models/' + str.split(conf['dataset'], '_')[0] +'_pretrained_model/' +  '_'.join([conf['dataset'], conf['arch'], 'ce.ckpt'])
                    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'],
                                conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
                elif arch_list[i] == 'gine':
                    conf['arch'] = 'gine'
                    conf['epochs'] = 150
                    # conf['pre_train'] = None
                    conf['pre_train'] = './pretrained_models/' + str.split(conf['dataset'], '_')[0] +'_pretrained_model/' +  '_'.join([conf['dataset'], conf['arch'], 'ce.ckpt'])
                    model = GINENet(num_node_features, num_edge_features, conf['hidden'], conf['dropout'], conf['num_tasks'])
                print(model is  not None)
                if model is not None:
                    out_path = 'results_' + conf['loss_type'] + '/' + conf['arch'] + '/' + '_'.join(
                        ['Re_SOAP', conf['arch'], conf['dataset'], conf['loss_type'], conf['loss_param']['type'], 'lr', str(conf['lr']), 'th', str(conf['loss_param']['threshold']), 'posNum', str(conf['posNum']), 'wd', str(conf['weight_decay']), 'repeats', str(j), 'epoch', str(conf['epochs']), 'mv_gamma', str(conf['loss_param']['mv_gamma'])])
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    print("conf:", conf)
                    run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['posNum'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path)
                    print("conf:", conf)



if __name__ == '__main__':

    data_list = ['hiv'] #'tox21_t0', 'tox21_t2', , 'toxcast_t12', 'tox21_t0'
    arch_list = ['mpnn', 'gine', 'mlpmpnn']
    conf['lr']  = 5e-4
    for data in data_list:
        run_main(data_name = data, arch_list=arch_list, conf = conf)




# out_path = 'results_gine_hiv_auprc2'
# if not os.path.exists(out_path):
#     os.makedirs(out_path)
# conf['loss_type'] = 'fastAP' #'smoothAP'
# conf['loss_param'] = {'K':10, 'm':5, 'gamma':1000, 'tau':3.0, 'bins':3}
# conf['ft_mode'] = 'fc_random'
# conf['pre_train'] = 'results_gine_hiv_ce/params150.ckpt'
# model = GINENet(num_node_features, num_edge_features, conf['hidden'], conf['dropout'], conf['num_tasks'])
# run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path)




