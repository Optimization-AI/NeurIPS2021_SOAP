import os
from numpy.lib import ufunclike
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from metric import compute_cla_metric
import numpy as np
from SOAP import AUPRCSampler, SOAPLOSS
# from imbalanced_loss import *
# from auprc_hinge import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### This is run function for classification tasks
def run_classification(train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size,
                       lr, lr_decay_factor, lr_decay_step_size, weight_decay, posNum = 1, loss_type='ce', loss_param={},
                       ft_mode='fc_random', pre_train=None, save_dir=None, repeats = 0):
    model = model.to(device)
    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train))
    if ft_mode == 'fc_random':
        model.mlp1.reset_parameters()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    global u, a, b, m, alpha, lamda
    if loss_type == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif loss_type in ['SOAP']:
        labels = [int(data.y.item()) for data in train_dataset]
        criterion = SOAPLOSS(loss_param['threshold'], batch_size,
                             len(train_dataset) + len(val_dataset) + len(test_dataset), loss_param['type'])
    elif loss_type in ['sum']:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        labels = [int(data.y.item()) for data in train_dataset]
        u = torch.zeros([len(train_dataset) + len(val_dataset) + len(test_dataset)])
    elif loss_type in ['wce', 'focal', 'ldam']:
        labels = [int(data.y.item()) for data in train_dataset]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        cls_num_list = [n_neg, n_pos]
        if loss_type == 'wce':
            criterion = WeightedBCEWithLogitsLoss(cls_num_list=cls_num_list)
        elif loss_type == 'focal':
            criterion = FocalLoss(cls_num_list=cls_num_list)
        elif loss_type == 'ldam':
            criterion = BINARY_LDAMLoss(cls_num_list=cls_num_list)
    elif loss_type in ['auroc']:
        criterion = None
        a, b, alpha, m = float(1), float(0), float(1), loss_param['m']
        labels = [int(data.y.item()) for data in train_dataset]
        loss_param['pos_ratio'] = sum(labels) / len(labels)
    elif loss_type in ['minmax']:
        criterion = AUCPRHingeLoss()
    elif loss_type in ['smoothAP']:
        criterion = smoothAP
    elif loss_type in ['fastAP']:
        criterion = fastAP

    train_loader_for_prc = DataLoader(train_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    best_val_metric = 0
    best_test_metric = 0
    epoch_bvl = 0
    epoch_test = 0
    crs_test_metric = 0

    # save the training records
    save_file = os.path.join(save_dir, 'record' + '_' + str(repeats) + '.txt')
    labels = [int(data.y.item()) for data in train_dataset]
    for epoch in range(1, epochs + 1):
        # if loss_type in ['ce', 'wce', 'focal', 'ldam', 'auroc', 'auroc2', 'minmax', 'smoothAP', 'fastAP']:
        #     train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        # elif loss_type in ['auprc1', 'auprc2', 'SOAP', 'sum']:

        if loss_type == 'ce':
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=AUPRCSampler(labels, batch_size, posNum=posNum))

        avg_train_loss = train_classification(model, optimizer, train_loader, num_tasks, device, epoch, lr, posNum, criterion,
                                              loss_type, loss_param)
        train_prc_results, train_roc_results = test_classification(model, train_loader_for_prc, num_tasks, device)
        val_prc_results, val_roc_results = test_classification(model, val_loader, num_tasks, device)
        test_prc_results, test_roc_results = test_classification(model, test_loader, num_tasks, device)

        print('Epoch: {:03d}, Training Loss: {:.6f}, Val PRC (avg over multitasks): {:.4f}, Test PRC (avg over multitasks): {:.4f}'.format(epoch, avg_train_loss, np.mean(val_prc_results), np.mean(test_prc_results)))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if np.mean(val_prc_results) > best_val_metric:
            epoch_bvl = epoch
            best_val_metric, crs_test_metric = np.mean(val_prc_results), np.mean(test_prc_results)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_eval.ckpt'))


        if np.mean(test_prc_results) > best_test_metric:
            epoch_test = epoch
            best_test_metric = np.mean(test_prc_results)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_test.ckpt'))


        if epoch == 1:
            while os.path.exists(save_file):
                repeats += 1
                save_file = os.path.join(save_dir, 'record' + '_' + str(repeats) + '.txt')

        if save_file is not None:
            if epoch == epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, 'params{}.ckpt'.format(epoch)))
            fp = open(save_file, 'a')
            fp.write(
                'Epoch: {:03d}, Train avg loss: {:.4f}, Train PRC: {:.4f}, Val PRC: {:.4f}, Test PRC: {:.4f}\n'.format(
                    epoch, avg_train_loss, np.mean(train_prc_results), np.mean(val_prc_results),
                    np.mean(test_prc_results)))
            fp.close()


        if epoch - epoch_bvl >= 50:
            break

    fp = open(save_file, 'a')
    fp.write(
        'Best val metric is: {:.4f}, Best val metric achieves at epoch: {:03d}, Corresponding test metric: {:04f}\n'.format(best_val_metric, epoch_bvl, crs_test_metric))
    fp.write('Best test metric is: {:.4f}, Best test metric achieves at epoch: {:03d}\n'.format(best_test_metric,
                                                                                                epoch_test))
    fp.close()

    print('Best val metric is: {:.4f}, Best val metric achieves at epoch: {:03d}, Corresponding test metric: {:04f}\n'.format(best_val_metric, epoch_bvl, crs_test_metric))
    print('Best test metric is: {:.4f}, Best test metric achieves at epoch: {:03d}\n'.format(best_test_metric, epoch_test))
    return crs_test_metric, best_test_metric

def train_classification(model, optimizer, train_loader, num_tasks, device, epoch, lr, posNum, criterion=None, loss_type=None,
                         loss_param={}):
    model.train()

    global a, b, m, alpha
    if loss_type == 'auroc' and epoch % 10 == 1:
        # Periordically update w_{ref}, a_{ref}, b_{ref}
        global state, a_0, b_0
        a_0, b_0 = a, b
        state = []
        for name, param in model.named_parameters():
            state.append(param.data)

    losses = []
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)

        if loss_type == 'ce':
            if len(batch_data.y.shape) != 2:
                batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in
                                 batch_data.y.cpu()])  # Skip those without targets (in PCBA, MUV, Tox21, ToxCast)
            mask = mask.to(device)
            target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
            target = target.to(device)
            loss = criterion(out, target) * mask
            loss = loss.sum()
            loss.backward()
            optimizer.step()

        elif loss_type == 'SOAP':
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            loss = criterion(predScore[:posNum], predScore[posNum:], batch_data.idx.view(-1, 1).long(), loss_param['mv_gamma'])
            loss.backward()
            optimizer.step()
        elif loss_type == 'sum':
            target = batch_data.y
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, num_tasks))
            loss1 = criterion(out, target)
            loss1 = loss1.sum()
            predScore = torch.nn.Sigmoid()(out)
            g = pairLossAlg2(10, predScore[0], predScore[1:])
            p = calculateP(g, u, batch_data.idx[0], 1)
            loss2 = surrLoss(g, p)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
        elif loss_type in ['wce', 'focal', 'ldam']:
            target = batch_data.y
            loss = criterion(out, target, epoch)
            loss.backward()
            optimizer.step()
        elif loss_type in ['auroc']:
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            loss = AUROC_loss(predScore, target, a, b, m, alpha, loss_param['pos_ratio'])
            curRegularizer = calculateRegularizerWeights(lr, model, state, loss_param['gamma'])
            loss.backward()
            optimizer.step()
            regularizeUpdate(model, curRegularizer)
            a, b, alpha = PESG_update_a_b_alpha_2(lr, a, a_0, b, b_0, alpha, m, predScore, target,
                                                  loss_param['pos_ratio'], loss_param['gamma'])
        elif loss_type in ['minmax']:
            target = batch_data.y
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        elif loss_type in ['smoothAP']:
            target = batch_data.y
            predScore = torch.sigmoid(out)
            loss = criterion(predScore, target, tau=loss_param['tau'])
            loss.backward()
            optimizer.step()
        elif loss_type in ['fastAP']:
            target = batch_data.y
            predScore = torch.sigmoid(out)
            loss = criterion(predScore, target, bins=loss_param['bins'])
            loss.backward()
            optimizer.step()

        # print('Iter {} | Loss {}'.format(i, loss.cpu().item()))
        losses.append(loss)
    return sum(losses).item() / len(losses)


def test_classification(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                                  num_tasks)

    return prc_results, roc_results