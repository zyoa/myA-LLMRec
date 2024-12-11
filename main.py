import os
import sys
import argparse
from transformers import set_seed

from utils import *
from train_model import *
from pre_train.sasrec.data_preprocess import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=1)

    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--emb", type=str, default='sbert')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--phase1_epoch", type=int, default=10)
    parser.add_argument("--phase2_epoch", type=int, default=5)

    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')

    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')

    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.device = 'cuda:0'

    set_seed(args.seed)

    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)
