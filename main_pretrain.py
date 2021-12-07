import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import time
import random
import argparse

from model.infonce import MoCo
from data.augmentation import get_train_val_augment
from data.vDataLoader import get_dataloader

from utils.utils import AverageMeter, ProgressMeter, save_checkpoint, \
    calc_topk_accuracy, Logger, neq_load_customized, adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser()

    # - basic
    parser.add_argument('--dataset', default='ucf101', type=str, choices=['hmdb51', 'ucf101', 'k400', 'k200'])
    parser.add_argument('--db_root', default='', type=str, help='root path of mp4 videos')
    parser.add_argument('--setting', default='ssl', type=str, choices=['ssl', 'sup'])

    parser.add_argument('--net', default='r3d18', type=str, choices=['s3d', 'r3d18'])
    parser.add_argument('--img_dim', default=112, type=int, help='input spatial size')
    parser.add_argument('--seq_len', default=16, type=int, help='sequence length')
    parser.add_argument('-bsz', '--batch_size', default=32, type=int, help='batch size per gpu')
    parser.add_argument('--double_sample', action='store_false')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')

    # - Model
    parser.add_argument('--aug_type', default=1, type=int)
    parser.add_argument('-t', '--threshold', default=0.5, type=float, help='hyper-parameter for augment')

    # - training
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[400, 450], nargs='*', type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--reset_lr', action='store_true')

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ttl_epoch', default=500, type=int)

    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('-j', '--workers', default=8, type=int,
                        help='number of data loading workers per gpu')

    # - log & save
    parser.add_argument('--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # - DDP
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    # use ddp mode if there are multiple gpus.
    if args.gpu is None:
        raise ValueError('No GPU is Available.')

    args.ddp = len(args.gpu.split(',')) > 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # - save folder & filename
    if args.resume:
        args.exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        args.exp_path = f'log-{args.setting}/{args.net}-{args.dataset}-{args.img_dim}_{args.seq_len}_bsz{args.batch_size}'
        args.exp_path += f'_aug{args.aug_type}_t{args.threshold}'

    args.img_path = os.path.join(args.exp_path, 'img')
    args.model_path = os.path.join(args.exp_path, 'model')

    if not os.path.exists(args.img_path):
        os.makedirs(args.img_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args


def main(args):
    # Setting ===============================
    best = 1e10
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

    # Initialization ========================
    # - DDP
    if args.ddp:
        dist.init_process_group(backend='nccl')

    # - Model
    if args.dataset in ['k400', 'k200']:
        args.mocok = 16384
        args.keep_all = True
    else:
        args.mocok = 2048
        args.keep_all = False

    model = MoCo(args.net, use_ddp=args.ddp, rank=args.local_rank,
                 threshold=args.threshold, aug_type=args.aug_type, K=args.mocok).cuda()

    if args.ddp:
        model = DDP(model, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    rank = args.local_rank

    # ================= Restart Training ==================
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            state_dict = checkpoint['state_dict']

            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp, state_dict, verbose=True)
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.reset_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
    else:
        print("=> train from scratch")

    # - Data
    (null_transform, base_transform), _ = get_train_val_augment(args)
    train_loader, train_sampler, val_loader = get_dataloader(args, train_transform=(null_transform, base_transform))

    args.logger = Logger(args.img_path)
    args.logger.log('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

    title = '=' * 10 + 'Parameter check' + '=' * 10
    args.logger.log(title)
    for name, p in model.named_parameters():
        if p.requires_grad:
            args.logger.log(name)
    args.logger.log('=' * len(title))

    # main loop
    for epoch in range(args.start_epoch, args.ttl_epoch):
        if args.ddp: train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        avg_loss = train(args, epoch, model, train_loader, optimizer)
        is_best = avg_loss < best
        if is_best: best = avg_loss

        save_state_dict = dict()
        for k in list(model_without_ddp.state_dict().keys()):
            if 'INN.' not in k:
                save_state_dict[k] = model_without_ddp.state_dict()[k]
        save_dict = {
            'epoch': epoch,
            'state_dict': save_state_dict,
            'best': best,
            'optimizer': optimizer.state_dict()}
        save_checkpoint(save_dict, is_best, gap=1,
                        filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                        keep_all=args.keep_all)


def train(args, epoch, model, train_loader, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1_meter = AverageMeter('Acc@1', ':.4f')
    top5_meter = AverageMeter('Acc@5', ':.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    tic = time.time()
    end = time.time()
    for i, (x, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        x = x.cuda(non_blocking=True)

        output, target = model(x)
        loss = F.cross_entropy(output, target)

        top1, top5 = calc_topk_accuracy(output, target, (1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = x.shape[0]
        losses.update(loss.item(), bsz)
        top1_meter.update(top1.item(), bsz)
        top5_meter.update(top5.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    args.logger.log('train Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} T-epoch:{t:.2f}\t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, t=time.time() - tic))

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    main(args)
