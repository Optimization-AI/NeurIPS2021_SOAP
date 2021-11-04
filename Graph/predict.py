from datasets import *
from model import *

import os
import torch

from torch_geometric.data import DataLoader
from metric import compute_cla_metric
import numpy as np


def get_data_files(name, seed=122):
    if name in ['hiv']:
        data_file = 'datasets/gnn_feats/{}.pt'.format(name)
        split_files = ['datasets/split_inds/{}scaffold{}.pkl'.format(name, x) for x in [122, 123, 124]]
        split_file = split_files[seed-122]
    elif name in ['pcba', 'muv', 'tox21', 'toxcast']:
        data_file = 'datasets/gnn_feats/{}.pt'.format(name)
        split_files = ['datasets/split_inds/{}random{}.pkl'.format(name, x) for x in [122, 123, 124]]
        split_file = split_files[seed-122]
    return data_file, split_file


def test_classification(model, test_loader, num_tasks, device, save_pred=False, out_path=None):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)

    if torch.cuda.is_available():
        if save_pred:
            np.save(out_path+'/'+'pred.npy', preds.cpu().detach().numpy())
        prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    else:
        if save_pred:
            np.save(out_path+'/'+'pred.npy', preds)
        prc_results, roc_results = compute_cla_metric(targets, preds, num_tasks)
    
    return prc_results, roc_results


from config.config_hiv import conf
out_path = 'results_hiv'
if not os.path.exists(out_path):
    os.makedirs(out_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_file, split_file = 'datasets/gnn_feats/hiv.pt','datasets/split_inds/hivscaffold.pkl'
dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(data_file, conf['graph_level_feature'])
assert conf['num_tasks'] == dataset[0].y.shape[-1]
train_dataset, val_dataset, test_dataset = split_data(dataset, split_file)

test_loader = DataLoader(test_dataset, conf['batch_size'], shuffle=False)
model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])    
model = model.to(device)
print('======================') 
print('Loading trained medel and testing...')
model_dir = 'bce_models/ml2features_hiv'
model_dir = os.path.join(model_dir, 'params.ckpt') 
model.load_state_dict(torch.load(model_dir))
num_tasks = conf['num_tasks']

test_prc_results, test_roc_results = test_classification(model, test_loader, num_tasks, device, out_path=out_path)
print('======================')        
print('Test PRC (avg over multitasks): {:.4f}'.format(np.mean(test_prc_results)))

