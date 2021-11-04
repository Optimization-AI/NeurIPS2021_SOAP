import torch
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision.datasets.folder import ImageFolder
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1)
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)
    return model



class MyImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return index, sample, target


def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
#    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    return auc(recall, precision)


def prc_recall_curve(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)


def ave_prc(targets, preds):
    return average_precision_score(targets, preds)

def compute_cla_metric(targets, preds, num_tasks):
    
    prc_results = []
    roc_results = []
    for i in range(num_tasks):
        is_labeled = targets[:,i] == targets[:,i] ## filter some samples without groundtruth label
        target = targets[is_labeled,i]
        pred = preds[is_labeled,i]
        try:
            prc = prc_auc(target, pred)
        except ValueError:
            prc = np.nan
            print("In task #", i+1, " , there is only one class present in the set. PRC is not defined in this case.")
        try:
            roc = roc_auc_score(target, pred)
        except ValueError:
            roc = np.nan
            print("In task #", i+1, " , there is only one class present in the set. ROC is not defined in this case.")
        if not np.isnan(prc): 
            prc_results.append(prc)
        else:
            print("PRC results do not consider task #", i+1)
        if not np.isnan(roc): 
            roc_results.append(roc)
        else:
            print("ROC results do not consider task #", i+1)
    return prc_results, roc_results


def global_surrogate_loss_with_sqh(target, pred, threshold):


    posNum = np.sum(target)
    target, pred = target.reshape(-1), pred.reshape(-1)
    # print(target, pred)
    # print(posNum)
    loss = 0
    for t in range(len(target)):
        if target[t] == 1:
            # print(t)
            all_surr_loss = np.maximum(threshold - (pred[t] - pred), np.array([0]*len(target)))**2
            num = np.sum(all_surr_loss * (target == 1))
            dem = np.sum(all_surr_loss)

            loss += -num/dem


    return loss/posNum





