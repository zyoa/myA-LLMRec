import os
import torch
import random
import time
import os

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.a_llmrec_model import *
from pre_train.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_model_phase1(args):
    print('A-LLMRec start train phase-1\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(train_model_phase1_, args=(world_size, args), nprocs=world_size)
    else:
        train_model_phase1_(0, 0, args)


def train_model_phase2(args):
    print('A-LLMRec strat train phase-2\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(train_model_phase2_, args=(world_size, args), nprocs=world_size)
    else:
        train_model_phase2_(0, 0, args)


def inference(args):
    print('A-LLMRec start inference\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(inference_, args=(world_size, args), nprocs=world_size)
    else:
        inference_(0, 0, args)


def train_model_phase1_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = f'cuda:{str(rank)}'

    model = A_llmrec_model(args).to(args.device)

    # preprocess data
    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    user_train, user_valid, user_test, usernum, itemnum = dataset
    print('user num:', usernum, 'item num:', itemnum)

    if args.multi_gpu:
        num_batch = (len(user_train) // args.batch_size1) // world_size
    else:
        num_batch = len(user_train) // args.batch_size1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # Init Dataloader, Model, Optimizer

    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_data_set, shuffle=True)
        train_data_loader = DataLoader(train_data_set,
                                       batch_size=args.batch_size1,
                                       sampler=train_sampler,
                                       pin_memory=True)
        model = DDP(model, device_ids=[args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size1, pin_memory=True)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.98))

    epoch_start_idx = 1

    model.train()
    t0 = time.time()

    if rank == 0:
        epoch_iterator = tqdm(range(epoch_start_idx, args.num_epochs + 1))
    else:
        epoch_iterator = range(epoch_start_idx, args.num_epochs + 1)

    for epoch in epoch_iterator:
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(tqdm(train_data_loader)):
            with torch.cuda.amp.autocast():
                model(
                    [x.numpy() for x in data],
                    optimizer=adam_optimizer,
                    batch_iter=[epoch, args.num_epochs + 1, step, num_batch],
                    mode='phase1',
                )

                if step % max(10, num_batch // 100) == 0 and rank == 0:
                    if args.multi_gpu:
                        model.module.save_model(args, epoch1=epoch)
                    else:
                        model.save_model(args, epoch1=epoch)

        if rank == 0:
            if args.multi_gpu:
                model.module.save_model(args, epoch1=epoch)
            else:
                model.save_model(args, epoch1=epoch)

    print(f'train time : {time.time() - t0}')

    if args.multi_gpu:
        destroy_process_group()

    return


def train_model_phase2_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = f'cuda:{str(rank)}'
    random.seed(0)

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = args.phase1_epoch
    model.load_model(args, phase1_epoch=phase1_epoch)
    # model = model.half()
    dataset = data_partition(
        args.rec_pre_trained_data,
        path=f'./data/amazon/{args.rec_pre_trained_data}.txt'
    )
    user_train, user_valid, user_test, usernum, itemnum = dataset
    print('user num:', usernum, 'item num:', itemnum)

    if args.multi_gpu:
        num_batch = (len(user_train) // args.batch_size2) // world_size
    else:
        num_batch = len(user_train) // args.batch_size2

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])

    print('average sequence length: %.2f' % (cc / len(user_train)))

    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size2, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids=[args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size2, pin_memory=True, shuffle=True)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))

    epoch_start_idx = 1

    model.train()
    t0 = time.time()

    if rank == 0:
        epoch_iterator = tqdm(range(epoch_start_idx, args.num_epochs + 1))
    else:
        epoch_iterator = range(epoch_start_idx, args.num_epochs + 1)

    for epoch in epoch_iterator:
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(tqdm(train_data_loader)):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u, seq, pos, neg], optimizer=adam_optimizer, batch_iter=[epoch, args.num_epochs + 1, step, num_batch], mode='phase2')
            if step % max(10, num_batch//50) == 0 and rank == 0:
                if args.multi_gpu:
                    model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch, step=step)
                else:
                    model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
        if rank == 0:
            if args.multi_gpu:
                model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
            else:
                model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)

    print('phase2 train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return


def inference_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = f'cuda:{str(rank)}'

    phase1_epoch = args.phase1_epoch
    phase2_epoch = args.phase2_epoch

    model = A_llmrec_model(args).to(args.device)
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    user_train, user_valid, user_test, usernum, itemnum = dataset
    num_batch = len(user_train) // args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    model.eval()

    if usernum > 1000:
        users = random.sample(range(1, usernum + 1), 1000)
    else:
        users = range(1, usernum + 1)

    user_list = [
        u for u in users if len(user_train[u]) >= 1 and len(user_test[u]) >= 1
    ]
    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)

    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids=[args.device], static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, pin_memory=True)

    for data in tqdm(inference_data_loader):
        u, seq, pos, neg = data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        model([u, seq, pos, neg, rank], mode='generate')
