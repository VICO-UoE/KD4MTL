from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import pdb


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWCML(object):
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, task: int):
        # pdb.set_trace()

        self.model = model
        self.dataset = dataset
        self.task = task

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.detach().data
        self._precision_matrices = self._diag_fisher()

        
            # self._means[n] = variable(p.data).detach()


    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.detach()
            # precision_matrices[n] = variable(p.data).detach()

        self.model.eval()
        # for input in self.dataset:
        for inputs, targets in enumerate(self.dataset):
            self.model.zero_grad()
            # input = variable(input)
            # targets = torch.zeros(targets.size(0), 10).scatter_(1, targets_x1.view(-1,1), 1)
            inputs, targets = inputs.cuda(), taregts.cuda()
            outputs = self.model(inputs)
            # output = self.model(input).view(1, -1)
            # label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
                else:
                    # precision_matrices[n].data = precision_matrices[n].data + 1.0
                    self._means[n] = self._means[n] * 0.0

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class EWCMeta(object):
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, task: int):
        # pdb.set_trace()

        self.model = model
        self.dataset = dataset
        self.task = task

        self.params = {n: p for n, p in self.model.named_params(self.model) if p.requires_grad}
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.detach()
        # pdb.set_trace()
        self.total = 0
        self.maximum = torch.zeros(1).cuda()
        self.feat_fisher = torch.zeros(50).cuda()
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._precision_matrices[n][self._precision_matrices[n]>=1.0] = 1.0
            self._precision_matrices[n][self._precision_matrices[n]<1.0] = self._precision_matrices[n][self._precision_matrices[n]<1.0] * 0.0



        # for n, p in deepcopy(self.params).items():
        #     self._means[n] = p.data.detach()
            # self._means[n] = variable(p.data).detach()


    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.detach()
            # precision_matrices[n] = variable(p.data).detach()
        self.model.eval()
        # for input in self.dataset:
        # pdb.set_trace()
        for idx, (inputs, targets) in enumerate(self.dataset):
        # dataset_iter = iter(self.dataset)
        # for idx in range(200):
            # try:
            #     inputs, targets = dataset_iter.next()
            # except:
            #     dataset_iter = iter(self.dataset)
            #     inputs, targets = dataset_iter.next()
            self.model.zero_grad()
            # input = variable(input)
            # targets = torch.zeros(targets.size(0), 10).scatter_(1, targets_x1.view(-1,1), 1)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = self.model(inputs, task=self.task)
            # feat = feat[1]
            feat.retain_grad()
            # output = self.model(input).view(1, -1)
            # label = outputs.max(1)[1]
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets)
            # loss = F.nll_loss(F.log_softmax(outputs, dim=1), label)
            loss.backward()
            # pdb.set_trace()

            # here for fc feat
            self.feat_fisher += (feat.detach() * feat.grad.detach()).pow(2).mean(0) * 0.5 / len(self.dataset)

            for n, p in self.model.named_params(self.model):
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
                    self.total = self.total + precision_matrices[n].data.sum()
                    self.maximum = torch.max(self.maximum, torch.max(precision_matrices[n].data))
                else:
                    self._means[n] = self._means[n] * 0.0

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_params(model):
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2 
            loss += _loss.sum()
            # loss += _loss.mean()
        return loss


class EWCMetaMl(object):
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, task: int):
        # pdb.set_trace()

        self.model = model
        self.dataset = dataset
        self.task = task

        self.params = {n: p for n, p in self.model.named_params(self.model) if p.requires_grad}
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.detach()
        # pdb.set_trace()
        self.total = 0
        self.maximum = torch.zeros(1).cuda()
        self.feat_fisher = torch.zeros(20).cuda()
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._precision_matrices[n][self._precision_matrices[n]>=1.0] = 1.0
            self._precision_matrices[n][self._precision_matrices[n]<1.0] = self._precision_matrices[n][self._precision_matrices[n]<1.0] * 0.0



        # for n, p in deepcopy(self.params).items():
        #     self._means[n] = p.data.detach()
            # self._means[n] = variable(p.data).detach()


    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.detach()
            # precision_matrices[n] = variable(p.data).detach()
        self.model.eval()
        # for input in self.dataset:
        # pdb.set_trace()
        for idx, (inputs, targets) in enumerate(self.dataset):
        # dataset_iter = iter(self.dataset)
        # for idx in range(200):
            # try:
            #     inputs, targets = dataset_iter.next()
            # except:
            #     dataset_iter = iter(self.dataset)
            #     inputs, targets = dataset_iter.next()
            self.model.zero_grad()
            # input = variable(input)
            # targets = torch.zeros(targets.size(0), 10).scatter_(1, targets_x1.view(-1,1), 1)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = self.model(inputs, task=self.task)
            feat = feat[1]
            # feat = feat[1]
            feat.retain_grad()
            # output = self.model(input).view(1, -1)
            # label = outputs.max(1)[1]
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets)
            # loss = F.nll_loss(F.log_softmax(outputs, dim=1), label)
            loss.backward()
            # pdb.set_trace()

            # here for fc feat
            self.feat_fisher += (feat.detach() * feat.grad.detach()).pow(2).sum(-1).sum(-1).mean(0) * 0.5 / len(self.dataset)

            for n, p in self.model.named_params(self.model):
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
                    self.total = self.total + precision_matrices[n].data.sum()
                    self.maximum = torch.max(self.maximum, torch.max(precision_matrices[n].data))
                else:
                    self._means[n] = self._means[n] * 0.0

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_params(model):
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2 
            loss += _loss.sum()
            # loss += _loss.mean()
        return loss


