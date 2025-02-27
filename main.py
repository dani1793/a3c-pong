from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn")

import my_optim
from model import ActorCritic
from test import test
from train import train
from pong import Pong
import time

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument("--headless", default=True, action="store_true", help="Run in headless mode")
parser.add_argument("--load_checkpoint", default=False, help="Load the checkpoint and run from there")
parser.add_argument("--save_progress", default=False, help="Save the model parameters when test is run")

def load_checkpoint(model, optimizer, filename='/output/checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def showTestResults(testValue):
    print('inside show Test Results')
    result = testValue.get()
    print(result[0])
        
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    testValue = mp.Queue()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shared_model = ActorCritic(
        1, 3)
    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    if args.load_checkpoint:
       load_checkpoint(shared_model, optimizer, args.load_checkpoint)

        
    shared_model.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, optimizer, testValue))
    p.start()
    processes.append(p)
    #showTestResults(testValue);
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        print('starting training')
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

