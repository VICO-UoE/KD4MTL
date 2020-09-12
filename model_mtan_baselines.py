import os
import torch
import fnmatch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
import shutil

from dataset.nyuv2 import *
from torch.autograd import Variable
from model.mtan import SegNet

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from progress.bar import Bar as Bar
import pdb
from mgda.min_norm_solvers import MinNormSolver, gradient_normalizers

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task loss weighting: uniform, gradnorm, mgda, dwa')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--alpha', default=0.12, type=float, help='hyper params of GradNorm')
opt = parser.parse_args()

class Weight(torch.nn.Module):
    def __init__(self, tasks):
        super(Weight, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))

def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'mtan_weight_{}_'.format(opt.weight) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'mtan_weight_{}_'.format(opt.weight) + 'model_best.pth.tar'))



# define model, optimiser and scheduler
tasks = ['semantic', 'depth', 'normal']

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2'
logger = Logger(os.path.join(opt.out, 'mtan_weight_{}_log.txt'.format(opt.weight)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'Ws', 'Wd', 'Wn'])


# define model, optimiser and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
use_cuda = torch.cuda.is_available()

model = SegNet().cuda()

Weights = Weight(tasks).cuda()
params = []
params += model.parameters()
params += [Weights.weights]
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=True)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
T = opt.temp
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
lambda_weight = np.ones([3, total_epoch])
best_loss = 100
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
            w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
            lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
    

    # iteration for all batches
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    bar = Bar('Training', max=train_batch)
    for k in range(train_batch):
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()

        train_pred, logsigma, feat = model(train_data)

        optimizer.zero_grad()
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        w = torch.ones(len(tasks)).float().cuda()
        if opt.weight == 'mgda':
            W = []
            for t_idx in range(len(tasks)):
                W += [feat[t_idx][-1]]
                W[t_idx].retain_grad()
            gygw = []
            for i, t in enumerate(tasks):
                gygw.append(torch.autograd.grad(train_loss[i], W[i], retain_graph=True))
            sol, min_norm = MinNormSolver.find_min_norm_element(gygw)
            for i in range(len(tasks)):
                w[i] = float(sol[i])
        if opt.weight == 'gradnorm':
            W = []
            norms = []
            for t_idx in range(len(tasks)):
                W += [feat[t_idx][-1]]
                W[t_idx].retain_grad()
            for i, t in enumerate(tasks):
                gygw = torch.autograd.grad(train_loss[i], W[i], retain_graph=True)
                norms.append(torch.norm(torch.mul(Weights.weights[i], gygw[0])))
            norms = torch.stack(norms)
            task_loss = torch.stack(train_loss)
            if epoch ==0 and k == 0:
                initial_task_loss = task_loss
            loss_ratio = task_loss.data / initial_task_loss.data
            inverse_train_rate = loss_ratio / loss_ratio.mean()
            mean_norm = norms.mean()
            constant_term = mean_norm.data * (inverse_train_rate ** opt.alpha)
            grad_norm_loss = (norms - constant_term).abs().sum()
            w_grad = torch.autograd.grad(grad_norm_loss, Weights.weights)[0]
            for i in range(len(tasks)):
                w[i] = Weights.weights[i].data
        if opt.weight == 'dwa':
            for i in range(len(tasks)):
                w[i] = lambda_weight[i, index]
        loss = sum(w[i].data * train_loss[i] for i in range(len(tasks)))
        optimizer.zero_grad()
        loss.backward()
        if opt.weight == 'gradnorm':
            Weights.weights.grad = torch.zeros_like(Weights.weights.data)
            Weights.weights.grad.data = w_grad.data
        optimizer.step()
        if opt.weight == 'gradnorm':
            Weights.weights.data = len(tasks) * Weights.weights.data / Weights.weights.data.sum()



        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        bar.suffix  = '({batch}/{size}) | LossS: {loss_s:.4f} | LossD: {loss_d:.4f} | LossN: {loss_n:.4f} | Ws: {ws:.4f} | Wd: {wd:.4f}| Wn: {wn:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_s=cost[1],
                    loss_d=cost[3],
                    loss_n=cost[6],
                    ws=w[0].data,
                    wd=w[1].data,
                    wn=w[2].data,
                    )
        bar.next()
    bar.finish()

    loss_index = (avg_cost[index, 0] + avg_cost[index, 3] + avg_cost[index, 6]) / 3.0
    isbest = loss_index < best_loss

    # evaluating test data
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        nyuv2_test_dataset = iter(nyuv2_test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
            test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
            test_depth, test_normal = test_depth.cuda(), test_normal.cuda()

            test_pred, _, _ = model(test_data)
            test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            cost[12] = test_loss[0].item()
            cost[13] = model.compute_miou(test_pred[0], test_label).item()
            cost[14] = model.compute_iou(test_pred[0], test_label).item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = model.depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = model.normal_error(test_pred[2], test_normal)

            avg_cost[index, 12:] += cost[12:] / test_batch


    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23],
                lambda_weight[0, index], lambda_weight[1, index], lambda_weight[2, index]])
    if isbest:
        best_loss = loss_index
        print_index = index

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
print('The best results is:')
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11], avg_cost[print_index, 12], avg_cost[print_index, 13],
                avg_cost[print_index, 14], avg_cost[print_index, 15], avg_cost[print_index, 16], avg_cost[print_index, 17], avg_cost[print_index, 18],
                avg_cost[print_index, 19], avg_cost[print_index, 20], avg_cost[print_index, 21], avg_cost[print_index, 22], avg_cost[print_index, 23]))