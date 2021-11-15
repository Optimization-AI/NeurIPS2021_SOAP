__author__ = 'Qi'
# Created by on 11/3/21.
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import random
import torch.nn as nn
# from loss import logistic_loss, sigmoid_loss

class AUPRCSampler(Sampler):

    def __init__(self, labels, batchSize, posNum=1):
        # positive class: minority class
        # negative class: majority class

        self.labels = labels
        self.posNum = posNum
        self.batchSize = batchSize

        self.clsLabelList = np.unique(labels)
        self.dataDict = {}

        for label in self.clsLabelList:
            self.dataDict[str(label)] = []

        for i in range(len(self.labels)):
            self.dataDict[str(self.labels[i])].append(i)

        self.ret = []


    def __iter__(self):
        minority_data_list = self.dataDict[str(1)]
        majority_data_list = self.dataDict[str(0)]

        # print(len(minority_data_list), len(majority_data_list))
        random.shuffle(minority_data_list)
        random.shuffle(majority_data_list)

        # In every iteration : sample 1(posNum) positive sample(s), and sample batchSize - 1(posNum) negative samples
        if len(minority_data_list) // self.posNum  >= len(majority_data_list)//(self.batchSize - self.posNum): # At this case, we go over the all positive samples in every epoch.
            # extend the length of majority_data_list from  len(majority_data_list) to len(minority_data_list)* (batchSize-posNum)
            majority_data_list.extend(np.random.choice(majority_data_list, len(minority_data_list) // self.posNum * (self.batchSize - self.posNum) - len(majority_data_list), replace=True).tolist()
            for i in range(len(minority_data_list) // self.posNum):
                if self.posNum == 1:
                    self.ret.append(minority_data_list[i])
                else:
                    self.ret.extend(minority_data_list[i*self.posNum:(i+1)*self.posNum])

                startIndex = i*(self.batchSize - self.posNum)
                endIndex = (i+1)*(self.batchSize - self.posNum)
                self.ret.extend(majority_data_list[startIndex:endIndex])

        else: # At this case, we go over the all negative samples in every epoch.
            # extend the length of minority_data_list from len(minority_data_list) to len(majority_data_list)//(batchSize-posNum) + 1

            minority_data_list.extend(np.random.choice(minority_data_list, (len(majority_data_list) // (self.batchSize - self.posNum) + 1 - len(minority_data_list)//self.posNum)*self.posNum, replace=True).tolist())
            for i in range(0, len(majority_data_list), self.batchSize - self.posNum):

                if self.posNum == 1:
                    self.ret.append(minority_data_list[i//(self.batchSize - self.posNum)])
                else:
                    self.ret.extend(minority_data_list[i//(self.batchSize- self.posNum)* self.posNum: (i//(self.batchSize-self.posNum) + 1)*self.posNum])

                self.ret.extend(majority_data_list[i:i + self.batchSize - self.posNum])

        return iter(self.ret)


    def __len__ (self):
        return len(self.ret)


class SOAPLOSS(nn.Module):
    def __init__(self, threshold, batch_size, data_length, loss_type = 'sqh'):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(SOAPLOSS, self).__init__()
        self.u_all = threshold**2/batch_size * torch.tensor([0]*data_length).view(-1, 1).cuda()
        self.u_pos = threshold**2/batch_size * torch.tensor([0]*data_length).view(-1, 1).cuda()
        self.threshold = threshold
        self.loss_type = loss_type
        print('The loss type is :', self.loss_type)


    def forward(self,f_ps, f_ns, index_s, gamma):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)

        vec_dat = torch.cat((f_ps, f_ns), 0)
        mat_data = vec_dat.repeat(len(f_ps), 1)

       #  print(mat_data.shape)

        f_ps = f_ps.view(-1, 1)

        neg_mask = torch.ones_like(mat_data)
        neg_mask[:, 0:f_ps.size(0)] = 0

        pos_mask = torch.zeros_like(mat_data)
        pos_mask[:, 0:f_ps.size(0)] = 1

        # test_tmp = f_ps- mat_data
        # print(f_ps.size(), mat_data.size(), test_tmp.size())

        # 3*1 - 3*64 ==> 3*64

        if self.loss_type == 'sqh':

            neg_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * neg_mask
            pos_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * pos_mask

        elif self.loss_type == 'lgs':

            neg_loss = logistic_loss(f_ps, mat_data, self.threshold) * neg_mask
            pos_loss = logistic_loss(f_ps, mat_data, self.threshold) * pos_mask

        elif self.loss_type == 'sgm':
            neg_loss = sigmoid_loss(f_ps, mat_data, self.threshold) * neg_mask
            pos_loss = sigmoid_loss(f_ps, mat_data, self.threshold) * pos_mask


        loss = pos_loss + neg_loss


        if f_ps.size(0) == 1:

            self.u_pos[index_s] = (1 - gamma) * self.u_pos[index_s] + gamma * (pos_loss.mean())
            self.u_all[index_s] = (1 - gamma) * self.u_all[index_s] + gamma * (loss.mean())
        else:
            # print(self.u_all[index_s], loss.size(), loss.sum(1, keepdim = 1))
            self.u_all[index_s] = (1 - gamma) * self.u_all[index_s] + gamma * (loss.mean(1, keepdim=True))
            self.u_pos[index_s] = (1 - gamma) * self.u_pos[index_s] + gamma * (pos_loss.mean(1, keepdim=True))



        p = (self.u_pos[index_s] - (self.u_all[index_s]) * pos_mask) / (self.u_all[index_s] ** 2)


        p.detach_()

        loss = torch.sum(p * loss)
   

        return loss







