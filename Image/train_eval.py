import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from utils import compute_cla_metric, ave_prc, global_surrogate_loss_with_sqh
from torch.utils.data import DataLoader
import numpy as np
from SOAP import SOAPLOSS, AUPRCSampler
# from loss import *
# from auprc_hinge import *
# import wandb
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





### This is run function for classification tasks
def run_classification(i, train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, loss_type='ce', loss_param={}, ft_mode='fc_random', pre_train=None, save_dir=None, bins = 5, tau = 1, posNum = 1, surr_loss = 'sqh', dataset = 'cifar10', mv_gamma = 0.999, imb_factor = 0.02):

    if dataset == "melanoma":
        n_train = 26500
        n_train_pos = 467
    else:
        if imb_factor == 0.02:
            n_train = 20400
            n_train_pos = 400
        elif imb_factor == 0.2:
            n_train = 24000
            n_train_pos = 4000

    model = model.to(device)
    if pre_train is not None:
        print('we are loading pretrain model')
        state_key = torch.load(pre_train)
        print('pretrain model is loaded from {} epoch'.format(state_key['epoch']))
        filtered = {k:v for k,v in state_key['model'].items() if 'fc' not in k}
        model.load_state_dict(filtered, False)
    if ft_mode == 'frozen':
        for key,param in model.named_parameters():
            if 'fc' in key and 'gn' not in key:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif ft_mode == 'fc_random':
        model.fc.reset_parameters()

    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    global u, a, b, m, alpha
    #_1, u_2,
    labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos

    if loss_type == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif loss_type in ['auprc2']:
        labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos
        criterion = None
        u = torch.zeros([len(train_dataset)])
    elif loss_type in ['wce','focal','ldam']:
        n_pos = n_train_pos
        n_neg = n_train - n_train_pos
        cls_num_list = [n_neg, n_pos]
        if loss_type == 'wce':
            criterion = WeightedBCEWithLogitsLoss(cls_num_list=cls_num_list)
        elif loss_type == 'focal':
            criterion = FocalLoss(cls_num_list=cls_num_list)
        elif loss_type == 'ldam':
            criterion = BINARY_LDAMLoss(cls_num_list=cls_num_list)
    elif loss_type in ['auroc2']:
        criterion = None
        a, b, alpha, m = float(1), float(0), float(1), loss_param['m']
        loss_param['pos_ratio'] = n_train_pos / n_train
    elif loss_type in ['auprc_lang']:
        criterion = AUCPRHingeLoss()
    elif loss_type in ['fastAP']:
        criterion = fastAP
    elif loss_type in ['smoothAP']:
        criterion = smoothAP
    elif loss_type in ['expAP']:
        criterion = expAP
    elif loss_type in ['SOAP']:
        labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos
        criterion = SOAPLOSS(threshold=loss_param['threshold'], data_length = len(train_dataset) + len(val_dataset), loss_type=surr_loss, gamma = mv_gamma)
    # elif loss_type in ['SOAPINDI']:
    #     labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos
    #     criterion = SOAPLOSSINDICATOR(threshold=loss_param['threshold'], batch_size=batch_size, data_length = len(train_dataset) + len(val_dataset), loss_type=surr_loss)
    # elif loss_type in ['GENERALSOAPLOSS']:
    #     labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos
    #     criterion = GENERALSOAPLOSS(threshold=loss_param['threshold'], batch_size = batch_size)


    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    best_auprc_score = 0
    final_auprc = 0
    best_test_auprc_score = 0


    for epoch in range(1, epochs + 1):

        # if loss_type in ['ce', 'wce', 'focal', 'ldam', 'auroc2', 'auprc_lang','fastAP', 'smoothAP', 'expAP']:
        #     train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
        # elif loss_type in ['auprc2', 'SOAP', 'GENERALSOAPLOSS']:
        if loss_type == 'ce':
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=16,
                                      pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=AUPRCSampler(labels, batch_size, posNum=posNum), num_workers=16, pin_memory=True)
        #
        avg_train_loss = train_classification(model, optimizer, train_loader, lr_decay_step_size, num_tasks, device, epoch, lr, criterion, loss_type, loss_param, bins, tau, posNum, mv_gamma = mv_gamma)
        train_auprc, train_roc,  train_ap,  train_surr_loss = val_train_classification(model, train_loader, num_tasks, device, loss_param)
        val_auprc, val_roc,  val_ap, val_surr_loss = val_train_classification(model, val_loader, num_tasks, device, loss_param)
        test_auprc, test_roc, test_ap, test_surr_loss = test_classification(model, test_loader, num_tasks, device, loss_param, dataset = dataset)


        if best_auprc_score <= np.mean(val_auprc):
            best_auprc_score = np.mean(val_auprc)
            final_auprc = np.mean(test_auprc)
            if save_dir is not None:
                torch.save({'model':model.state_dict(), 'epoch':epoch}, os.path.join(save_dir, str(i) + '_best.ckpt'))


        if best_test_auprc_score <= np.mean(test_auprc):
            best_test_auprc_score = np.mean(test_auprc)

        print('Epoch: {:03d}, Training Loss: {:.6f}, Train AUPRC: {:.4f}, Val AUPRC (avg over multitasks): {:.4f}, Best AUPRC: {:.4f}, Test AUPRC: {:.4f} Final AUPRC: {:.4f} Best AUPRC: {:.4f}'
              .format(epoch, avg_train_loss, np.mean(train_auprc), np.mean(val_auprc), best_auprc_score, np.mean(test_auprc), final_auprc, best_test_auprc_score))
        print('Train AP {:.4f}, Val AP: {}, Test AP: {:.4f}\n'.format(train_ap, val_ap, test_ap))
        print('Train Surr Loss {:.4f}, Val Surr Loss: {}, Test Surr Loss: {:.4f}\n'.format(train_surr_loss, val_surr_loss, test_surr_loss))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if save_dir is not None:
            fp = open(os.path.join(save_dir, str(i)+'_res_auprc_avepre.txt'), 'a')
            fp.write('Train AUPRC {:.4f}, Val AUPRC: {}, Test AUPRC: {:.4f}, Final AUPRC: {:.4f}, Train avg loss: {:.4f}\n'.format(np.mean(train_auprc), np.mean(val_auprc), np.mean(test_auprc), final_auprc, avg_train_loss))
            fp.close()
            fp = open(os.path.join(save_dir, str(i)+'_res_auroc.txt'), 'a')
            fp.write('Train avg loss: {:.4f}, Val AUROC: {:.4f} Test AUCROC: {:.4f}\n'.format(avg_train_loss, np.mean(val_roc), np.mean(test_roc)))
            fp.close()
            fp = open(os.path.join(save_dir, str(i)+'_ap.txt'), 'a')
            fp.write(
                'Train AP {:.4f}, Val AP: {}, Test AP: {:.4f}\n'.format(
                    train_ap, val_ap, test_ap))
            fp.close()
            fp = open(os.path.join(save_dir, str(i) + '_surr_loss.txt'), 'a')
            fp.write(
                'Train Surr Loss {:.4f}, Val Surr Loss: {}, Test Surr Loss: {:.4f}\n'.format(
                    train_surr_loss, val_surr_loss, test_surr_loss))
            fp.close()




    if save_dir is not None:
        torch.save({'model':model.state_dict(), 'epoch':epochs}, os.path.join(save_dir, str(i) + '_last.ckpt'))



def train_classification(model, optimizer, train_loader, lr_decay_step_size, num_tasks, device, epoch, lr, criterion=None, loss_type=None, loss_param={}, bins = 5, tau = 1.0, posNum = 1, mv_gamma=0.999):
    model.train()
    
    global a, b, m, alpha
    if loss_type == 'auroc2' and epoch % 10 == 1:
        # Periordically update w_{ref}, a_{ref}, b_{ref}
        global state, a_0, b_0
        a_0, b_0 = a, b
        state = []
        for name, param in model.named_parameters():
            state.append(param.data)
    losses = []
    for i, (index, inputs, target) in enumerate(train_loader):

        if i%50 == 0:
            print(epoch, " : ", i, "/", len(train_loader))
        # warmup_learning_rate(epoch, i, lr, len(train_loader), optimizer)
        # print(index, target)
        optimizer.zero_grad()
        inputs = inputs.to(device)
        target = target.to(device).float()
        out = model(inputs)

        if loss_type == 'ce':
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, num_tasks))
            loss = criterion(out, target)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
        elif loss_type in ['wce','focal','ldam']:
            loss = criterion(out, target, epoch)
            loss.backward()
            optimizer.step()
        elif loss_type in ['auroc2']:
            predScore = torch.nn.Sigmoid()(out)
            loss = AUROC_loss(predScore, target, a, b, m, alpha, loss_param['pos_ratio'])
            curRegularizer = calculateRegularizerWeights(lr, model, state, loss_param['gamma'])
            loss.backward()
            optimizer.step()
            regularizeUpdate(model, curRegularizer)
            a, b, alpha = PESG_update_a_b_alpha_2(lr, a, a_0, b, b_0, alpha, m, predScore, target, loss_param['pos_ratio'], loss_param['gamma'])
        elif loss_type in ['auprc_lang']:
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        elif loss_type in ['smoothAP']:
            predScore = torch.sigmoid(out)
            loss = criterion(predScore, target, tau = tau)
            loss.backward()
            optimizer.step()
        elif loss_type in ['fastAP']:
            predScore = torch.sigmoid(out)
            # predScore = out/torch.norm(out)
            loss = criterion(predScore, target, bins = bins)
            loss.backward()
            optimizer.step()
        elif loss_type in ['expAP']:
            # predScore = out / torch.norm(out)
            predScore = torch.sigmoid(out)
            loss = criterion(predScore, target, tau = tau)
            loss.backward()
            optimizer.step()
        elif loss_type in ['SOAP']:
            predScore = torch.nn.Sigmoid()(out)
            loss = criterion(f_ps=predScore[0:posNum], f_ns=predScore[posNum:], index_s=index[0:posNum])

            loss.backward()
            optimizer.step()

        losses.append(loss)
    return sum(losses).item() / len(losses)


def val_train_classification(model, test_loader, num_tasks, device, loss_param):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    for (index, inputs, target)in test_loader:

        inputs = inputs.to(device)
        target = target.to(device).float()
        with torch.no_grad():
            out = model(inputs)
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, num_tasks))
        if out.shape[1] == 1:
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        else:
            pred = torch.softmax(out, dim=-1)[:, 1:2]
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, target], dim=0)


    auprc, auroc = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    ap = ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    surro_loss = global_surrogate_loss_with_sqh(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), loss_param['threshold'])



    return auprc, auroc, ap, surro_loss


def test_classification(model, test_loader, num_tasks, device, loss_param, dataset = 'cifar10'):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    for (inputs, target) in test_loader:
        inputs = inputs.to(device)
        target = target.to(device).float()
        if dataset == 'cifar10':
            target[target <= 4] = 0
            target[target > 4] = 1
        elif dataset == 'cifar100':
            target[target <= 49] = 0
            target[target > 49] = 1


        with torch.no_grad():
            out = model(inputs)
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, num_tasks))

        if out.shape[1] == 1:
            pred = torch.sigmoid(out) ### prediction real number between (0,1)
        else:
            pred = torch.softmax(out,dim=-1)[:,1:2]
        preds = torch.cat([preds,pred], dim=0)
        # print(preds)
        targets = torch.cat([targets, target], dim=0)
        # print(targets)
    auprc, auroc = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    ap = ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    surro_loss = global_surrogate_loss_with_sqh(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), loss_param['threshold'])

    
    return auprc, auroc, ap, surro_loss


def warmup_learning_rate(epoch, batch_id, lr, total_batches, optimizer):
    if epoch <= 5:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (5 * total_batches)
        lr = 0.01 + p * (lr - 0.01)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def plot_precision_recall_curve(model, vt_batch_size, test_dataset, saved_model, method, dataset = 'cifar10'):

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = model.to(device)
    # wandb.watch(model)


    state_key = torch.load(saved_model)
    print('pretrain model is loaded from {} epoch'.format(state_key['epoch']))
    model.load_state_dict(state_key['model'])

    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    for (inputs, target) in test_loader:
        inputs = inputs.to(device)
        target = target.to(device).float()
        if dataset == 'cifar10':
            target[target <= 4] = 0
            target[target > 4] = 1
        elif dataset == 'cifar100':
            target[target <= 49] = 0
            target[target > 49] = 1

        with torch.no_grad():
            out = model(inputs)

        if out.shape[1] == 1:
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        else:
            pred = torch.softmax(out, dim=-1)[:, 1:2]
        preds = torch.cat([preds, pred], dim=0)

        # print(preds)
        targets = torch.cat([targets, target], dim=0)
    precision, recall, _ = precision_recall_curve(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # precision_recall_plt = disp.plot()


    plt.plot(recall, precision, label = method, linewidth = 2)
    if dataset == 'cifar10':
        plt.title('CIFAR-10', fontsize = 25)
    else:
        plt.title('CIFAR-100', fontsize = 25)
    plt.xlabel('Recall', fontsize = 20)
    plt.ylabel('Precision', fontsize = 20)
    plt.hlines(0.5, -0.03, 1.03, colors='gray', linestyles = '--', linewidth = 2)
    plt.ylim(0.45,1)
    plt.legend(fontsize = 13)
    plt.savefig(os.path.join('results', dataset, dataset +'_'+method+'_precision_recall_curve.png'))
