#!/usr/bin/env python3
from belief_fill.dataloader import OccupancyDataset
from belief_fill.model import OccupancyRepair
import os
import random
import argparse
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import joblib

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('train_inputs', type=str, help='pattern to true occupancy grid')
parser.add_argument('train_labels', type=str, help='pattern to ablated occupancy grid')
parser.add_argument('test_inputs', type=str, help='pattern to true occupancy grid')
parser.add_argument('test_labels', type=str, help='pattern to ablated occupancy grid')
parser.add_argument('test_out_xs', type=str, help='pattern to true occupancy grid')
parser.add_argument('test_out_yhats', type=str, help='pattern to ablated occupancy grid')
parser.add_argument(
    '--nepoch', type=int, default=250, help='Number of training epochs')
parser.add_argument(
    '--batch_size', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--outf', type=str, default='vel_cmd', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--should_train', type=str,  default='True', help='should train')

opt = parser.parse_args()

def blue(x): 
  return '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

train_dataset = OccupancyDataset(opt.train_inputs, opt.train_labels)
test_dataset = OccupancyDataset(opt.test_inputs, opt.test_labels)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers))

model = OccupancyRepair(50, 100, 100)
if opt.model != '':
  model.load_state_dict(torch.load(opt.model))
model.cuda()

def my_loss(pred, target):
  return torch.sum((pred - target)**2)


def compute_loss(model, data):
  Xs, ys = data
  Xs, ys = Xs.cuda(), ys.cuda()
  yhats, _ = model(Xs)
  return my_loss(yhats, ys)


def train_network():
  optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  num_batch = len(train_dataset) / opt.batch_size
  for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(train_dataloader, 0):
      optimizer.zero_grad()
      loss = compute_loss(model.train(), data)
      loss.backward()
      optimizer.step()
      print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))
      if i % 10 == 0: 
        j, data = next(enumerate(test_dataloader, 0))
        loss = compute_loss(model.eval(), data)
        print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
    torch.save(model.state_dict(), '%s/vel_model_%d.pth' % (opt.outf, epoch))
  torch.save(model.state_dict(), '%s/vel_final_model.pth' % (opt.outf))

if opt.should_train == 'True':
  train_network()

def evaluate_results():
  with torch.no_grad():
    total_loss = 0
    total_testset = 0
    Xs_lst = []
    yhats_lst = []    
    for i, data in enumerate(test_dataloader, 0):
      Xs, ys = data
      Xs, ys = Xs.cuda(), ys.cuda()
      yhats, _ = model(Xs)
      total_testset += yhats.shape[0]
      total_loss += my_loss(yhats, ys)
      Xs_lst.append(Xs.cpu().numpy())
      yhats_lst.append(yhats.cpu().numpy())
      
    Xs_stack = np.concatenate(Xs_lst, axis=0)
    yhats_stack = np.concatenate(yhats_lst, axis=0) 
    joblib.dump(Xs_stack, opt.test_out_xs)
    joblib.dump(Xs_stack, opt.test_out_yhats)
    print("average test set loss {}".format(total_loss / float(total_testset)))

evaluate_results()