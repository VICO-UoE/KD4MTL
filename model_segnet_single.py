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
from model.segnet import SegNet
import numpy as np
import pdb
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='Single Task Learning (SegNet)')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--task', default='semantic', type=str, help='task: semantic, depth, normal')
opt = parser.parse_args()

tasks = ['S', 'D', 'N']


def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'segnet_single_model_task_{}_'.format(opt.task) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'segnet_single_model_task_{}_'.format(opt.task) + 'model_best.pth.tar'))

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2'
logger = Logger(os.path.join(opt.out, 'segnet_single_model_task_{}_log.txt'.format(opt.task)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'Ws', 'Wd', 'Wn'])

# define model, optimiser and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
use_cuda = torch.cuda.is_available()
model = SegNet(type_=opt.type, class_nb=13).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
if opt.task == 'semantic':
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC\n')
elif opt.task == 'depth':
    print('LOSS FORMAT: DEPTH_LOSS ABS_ERR REL_ERR\n')
elif opt.task == 'normal':
    print('LOSS FORMAT: NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_train_val_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0, drop_last=True)


nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
lambda_weight = np.zeros([3, total_epoch])
best_loss = 100
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(train_batch):
        # pdb.set_trace()
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        

        train_pred, logsigma, _ = model(train_data)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)
        w = torch.zeros(len(tasks)).cuda()
        if opt.task == 'semantic':
            w[0] = 1.0
        elif opt.task == 'depth':
            w[1] = 1.0
        elif opt.task == 'normal':
            w[2] = 1.0
        loss = 0
        for i in range(3):
            loss = loss + train_loss[i] * w[i]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        if opt.task =='semantic':
            bar.suffix  = '({batch}/{size}) | LossS: {loss_s:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_s=avg_cost[index,1],
                    )
        elif opt.task == 'depth':
            bar.suffix  = '({batch}/{size}) | LossD: {loss_d:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_d=avg_cost[index,3],
                    )
        elif opt.task == 'normal':
            bar.suffix  = '({batch}/{size}) | LossN: {loss_n:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_n=avg_cost[index,6],
                    )
        bar.next()
    bar.finish()

    if opt.task == 'semantic':
        loss_index = cost[0]
    elif opt.task == 'depth':
        loss_index = cost[3]
    elif opt.task == 'normal':
        loss_index = cost[6]
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


    if opt.task == 'semantic':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
          'TEST: {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], 
                avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
    elif opt.task == 'depth':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
          'TEST: {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], 
                avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
    elif opt.task == 'normal':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
    # print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
    #       'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
    #       .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
    #             avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
    #             avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
    #             avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
    #             avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))


    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23],
                lambda_weight[0, index], lambda_weight[1, index], lambda_weight[2, index]])

    if isbest:
        best_loss = loss_index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
