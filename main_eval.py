import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import data.videotransforms as vT

import os
import sys

import argparse
import time
import numpy as np
import random
import pickle
from tqdm import tqdm
import json

from model.classifier import LinearClassifier
from utils.utils import AverageMeter, ProgressMeter, save_checkpoint, \
    calc_topk_accuracy, Logger, neq_load_customized, adjust_learning_rate

from data.augmentation import get_train_val_augment
from data.vDataset import VideoDataset
from data.vDataLoader import get_dataloader, data_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='ucf101', type=str,
                        choices=['hmdb51', 'ucf101'])
    parser.add_argument('--db_root', default='', type=str, help='root path of mp4 videos')
    parser.add_argument('--setting', default='sup', type=str, choices=['ssl', 'sup'])
    parser.add_argument('--task', default='lincls', type=str, choices=['lincls'])

    parser.add_argument('--net', default='r3d18', type=str, choices=['s3d', 'r3d18'])
    parser.add_argument('--img_dim', default=224, type=int, help='input spatial size')
    parser.add_argument('--seq_len', default=64, type=int, help='sequence length')
    parser.add_argument('-bsz', '--batch_size', default=32, type=int, help='batch size per gpu')
    parser.add_argument('--double_sample', action='store_true')  # default False
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.9, type=float, help='dropout')
    parser.add_argument('--warmup', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--ttl_epoch', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers per gpu')

    parser.add_argument('--train_what', default='', type=str)
    parser.add_argument('--val_all', action='store_true')

    parser.add_argument('--print_freq', default=10, type=int, help='frequency of printing output during training')
    parser.add_argument('--eval_freq', default=1, type=int)

    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--retrieval', action='store_true', help='path of model to nn retrieval')

    parser.add_argument('--dirname', default=None, type=str, help='dirname for feature')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--five_crop', action='store_true')
    parser.add_argument('--ten_crop', action='store_true')

    # DDP
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
    elif args.test:
        args.exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        args.exp_path = f'log-{args.task}/{args.net}-{args.dataset}-{args.img_dim}_{args.seq_len}_bsz{args.batch_size}'
        args.exp_path += f'_pt={args.pretrain.replace("/", "-") if args.pretrain else "None"}'

    args.img_path = os.path.join(args.exp_path, 'img')
    args.model_path = os.path.join(args.exp_path, 'model')

    if not os.path.exists(args.img_path):
        os.makedirs(args.img_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args


def main(args):
    best = 0
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    data_config(args)
    print(f'=> Effective BatchSize = {args.batch_size}')

    # - DDP
    if args.ddp:
        dist.init_process_group(backend='nccl')

    # - Model
    if args.train_what == 'last':  # for linear probe
        args.final_bn = True
        args.final_norm = True
        args.use_dropout = False
    else:  # for training the entire network
        args.final_bn = False
        args.final_norm = False
        args.use_dropout = True

    if args.task == 'lincls':
        model = LinearClassifier(
            network=args.net,
            num_class=args.num_class,
            dropout=args.dropout,
            use_dropout=args.use_dropout,
            use_final_bn=args.final_bn,
            use_l2_norm=args.final_norm)
    else:
        raise NotImplementedError

    model.cuda()

    # - optimizer
    if args.train_what == 'last':
        print('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})

    elif args.train_what == 'ft':
        print('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})

    else:  # train all
        params = []
        print('=> [optimizer] train all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})

    if args.train_what == 'last':
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # - scheduler
    if args.train_what == 'last':
        args.ttl_epoch = 150
        args.dropout = 0.5
        if args.dataset in ['hmdb51']:
            args.schedule = [40, 60]
        elif args.dataset in ['ucf101']:
            args.schedule = [60, 80]
        else:
            raise NotImplementedError
    else:
        args.ttl_epoch = 400
        if args.dataset in ['hmdb51']:
            args.schedule = [100, 150]
        elif args.dataset in ['ucf101']:
            args.schedule = [130, 180]
        else:
            raise NotImplementedError
    print('=> Using scheduler at {} epochs'.format(args.schedule))

    if args.ddp:
        model = DDP(model, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # ================ test: higher priority ====================
    if args.test:
        for name, param in model.named_parameters():
            param.requires_grad = False  # freeze model
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k, v in state_dict.items():
                if '_k' in k: continue
                k = k.replace('module.base_q.', 'backbone.')
                k = k.replace('module.encoder_q.0.', 'backbone.')
                k = k.replace('base_q.', 'backbone.')
                k = k.replace('encoder_q.0.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict

            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_ddp, state_dict, verbose=True)

        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0
            print("[Warning] if test random init weights, press c to continue")
            import ipdb
            ipdb.set_trace()

        torch.cuda.empty_cache()

        if args.retrieval:
            args.dirname = '_'.join(args.test.split('/')[-1].split('_')[:-1]) + '_feature'
            if not os.path.exists(os.path.join(os.path.dirname(args.test), args.dirname)):
                os.makedirs(os.path.join(os.path.dirname(args.test), args.dirname))
            args.logger = Logger(path=os.path.join(os.path.dirname(args.test), args.dirname))
            args.logger.log('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

            test_retrieval(args, model)
            print(f'Save feature to dirname: {args.dirname}')
        elif args.center_crop or args.five_crop or args.ten_crop:
            args.logger = Logger(path=os.path.dirname(args.test))
            args.logger.log('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

            _, val_transform = get_train_val_augment(args)
            val_set = VideoDataset(args.val_list, args.video_path, double_sample=False,
                                   transform=val_transform, mode='test',
                                   seq_len=args.seq_len, ds=args.ds,
                                   return_vpath=True,
                                   return_label=True)
            test_10crop(args, model, epoch, val_set)
        else:
            raise NotImplementedError

        sys.exit(0)

    # - Data
    train_transform, val_transform = get_train_val_augment(args)
    train_loader, train_sampler, val_loader = get_dataloader(args, train_transform, val_transform)

    args.logger = Logger(path=args.img_path)
    args.logger.log('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

    args.logger.log('===================================')
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
                args.logger.log('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp, state_dict, verbose=True)
            args.logger.log("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                args.logger.log('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            args.logger.log("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k, v in state_dict.items():
                if '_k' in k: continue
                if 'fc' in k: continue
                k = k.replace('module.base_q.', 'backbone.')
                k = k.replace('module.encoder_q.0.', 'backbone.')
                k = k.replace('base_q.', 'backbone.')
                k = k.replace('encoder_q.0.', 'backbone.')
                k = k.replace('module.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict

            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_ddp, state_dict, verbose=True)
            args.logger.log("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            args.logger.log("[Warning] no checkpoint found at '{}', use random init".format(args.pretrain))
            raise NotImplementedError

    else:
        args.logger.log("=> train from scratch")

    if args.val_all:  # when do evaluation, no train
        _, val_acc = validate(args, args.start_epoch, model, val_loader)
        exit(0)

    # main loop
    for epoch in range(args.start_epoch, args.ttl_epoch):
        if args.ddp: train_sampler.set_epoch(epoch)
        np.random.seed(epoch)
        random.seed(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train(args, epoch, model, train_loader, optimizer)
        if epoch % args.eval_freq == 0:
            _, val_acc = validate(args, epoch, model, val_loader)

            # save check_point
            is_best = val_acc > best
            best = max(val_acc, best)
            state_dict = model_without_ddp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best': best,
                'optimizer': optimizer.state_dict()}
            save_checkpoint(save_dict, is_best, 1,
                            filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                            keep_all=False)

    args.logger.log('Training from ep %d to ep %d finished' % (args.start_epoch, args.ttl_epoch))
    sys.exit(0)


def train(args, epoch, model, data_loader, optimizer):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    top1_meter = AverageMeter('Acc@1', ':.4f')
    top5_meter = AverageMeter('Acc@5', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))

    if args.train_what == 'last':
        model.eval()  # totally freeze BN in backbone
    else:
        model.train()

    if args.task == 'lincls' and args.final_bn:
        try:
            model.final_bn.train()
        except:
            model.module.final_bn.train()

    end = time.time()
    tic = time.time()

    for idx, (x, target) in enumerate(data_loader):
        data_time.update(time.time() - end)
        B = x.shape[0]

        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logit, _ = model(x)
        loss = F.cross_entropy(logit, target)
        top1, top5 = calc_topk_accuracy(logit, target, (1, 5))

        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)

    args.logger.log('train Epoch: [{0}][{1}/{2}]\t'
                    'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time() - tic))

    return losses.avg, top1_meter.avg


def validate(args, epoch, model, data_loader):
    batch_time = AverageMeter('Time', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    top1_meter = AverageMeter('Acc@1', ':.4f')
    top5_meter = AverageMeter('Acc@5', ':.4f')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (x, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            B = x.size(0)

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logit, _ = model(x)
            loss = F.cross_entropy(logit, target)
            top1, top5 = calc_topk_accuracy(logit, target, (1, 5))

            losses.update(loss.item(), B)
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

    args.logger.log('val Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    return losses.avg, top1_meter.avg


def test_10crop(args, model, epoch, dataset):
    prob_dict = {}
    model.eval()

    # aug_list: 1,2,3,4,5 = topleft, topright, bottomleft, bottomright, center
    # flip_list: 0,1 = raw, flip
    if args.center_crop:
        print('Test using center crop')
        args.logger.log('Test using center_crop\n')
        aug_list = [5]
        flip_list = [0]
        title = 'center'
    if args.five_crop:
        print('Test using 5 crop')
        args.logger.log('Test using 5_crop\n')
        aug_list = [5, 1, 2, 3, 4]
        flip_list = [0]
        title = 'five'
    if args.ten_crop:
        print('Test using 10 crop')
        args.logger.log('Test using 10_crop\n')
        aug_list = [5, 1, 2, 3, 4]
        flip_list = [0, 1]
        title = 'ten'

    with torch.no_grad():
        end = time.time()
        # for loop through 10 types of augmentations, then average the probability
        for flip_idx in flip_list:
            for aug_idx in aug_list:
                print('Aug type: %d flip: %d' % (aug_idx, flip_idx))

                if flip_idx == 0:
                    transform = transforms.Compose([
                        vT.ToPILImage(),
                        vT.RandomHorizontalFlip(p=0),
                        vT.FiveCrop(size=(224, 224), where=aug_idx),
                        vT.Resize(size=(args.img_dim, args.img_dim)),
                        transforms.RandomApply([vT.ColorJitter(0.2, 0.2, 0.2, 0.1, per_img=False)], p=0.3),
                        vT.ToTensor(),
                        vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        vT.ToClip()
                    ])
                else:
                    transform = transforms.Compose([
                        vT.ToPILImage(),
                        vT.RandomHorizontalFlip(p=1.),
                        vT.FiveCrop(size=(224, 224), where=aug_idx),
                        vT.Resize(size=(args.img_dim, args.img_dim)),
                        transforms.RandomApply([vT.ColorJitter(0.2, 0.2, 0.2, 0.1, per_img=False)], p=0.3),
                        vT.ToTensor(),
                        vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        vT.ToClip()
                    ])

                dataset.transform = transform
                dataset.return_label = True
                test_sampler = data.SequentialSampler(dataset)
                data_loader = data.DataLoader(dataset,
                                              batch_size=1,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

                for idx, (x, (target, vpath)) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    x = x.cuda(non_blocking=True)
                    logit, _ = model(x)

                    # average probability along the temporal window
                    prob_mean = F.softmax(logit, dim=-1).mean(0, keepdim=True)

                    vpath = vpath[0]
                    if vpath not in prob_dict.keys():
                        prob_dict[vpath] = {'mean_prob': [], 'target': target[0].item()}
                    prob_dict[vpath]['mean_prob'].append(prob_mean)

                if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
                    print('center-crop result:')
                    acc_1 = summarize_probability(prob_dict, 'center')
                    args.logger.log('center-crop:')
                    args.logger.log('test Epoch: [{0}]\t'
                                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                                    .format(epoch, acc=acc_1))

            if (title == 'ten') and (flip_idx == 0):
                print('five-crop result:')
                acc_5 = summarize_probability(prob_dict, 'five')
                args.logger.log('five-crop:')
                args.logger.log('test Epoch: [{0}]\t'
                                'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                                .format(epoch, acc=acc_5))

    print('%s-crop result:' % title)
    acc_final = summarize_probability(prob_dict, 'ten')
    args.logger.log('%s-crop:' % title)
    args.logger.log('test Epoch: [{0}]\t'
                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                    .format(epoch, acc=acc_final))
    sys.exit(0)


def summarize_probability(prob_dict, title):
    acc = [AverageMeter(), AverageMeter()]
    stat = {}
    for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        target = item['target']
        mean_prob = torch.stack(item['mean_prob'], 0).mean(0)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob, torch.LongTensor([target]).cuda(), (1, 5))
        stat[vname] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

    print('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
          .format(acc=acc))

    with open(os.path.join(os.path.dirname(args.test),
                           '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def test_retrieval(args, model):
    model.eval()

    with torch.no_grad():
        test_transform = transforms.Compose([
            vT.ToPILImage(),
            vT.CenterCrop(size=(224, 224)),
            vT.Resize(size=(args.img_dim, args.img_dim)),
            transforms.RandomApply([vT.ColorJitter(0.2, 0.2, 0.2, 0.1, per_img=False)], p=0.3),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.ToClip()
        ])

        train_dataset = VideoDataset(args.train_list, args.video_path, double_sample=False,
                                     transform=test_transform, mode='test',
                                     seq_len=args.seq_len, ds=args.ds,
                                     return_vpath=True,
                                     return_label=True)
        train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                       num_workers=args.workers, pin_memory=True, drop_last=False)

        test_dataset = VideoDataset(args.val_list, args.video_path, double_sample=False,
                                    transform=test_transform, mode='test',
                                    seq_len=args.seq_len, ds=args.ds,
                                    return_vpath=True,
                                    return_label=True)
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.workers, pin_memory=True, drop_last=False)

        if args.dirname is None:
            dirname = 'feature'
        else:
            dirname = args.dirname

        if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)):
            test_feature = torch.load(
                os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)).cuda()
            test_label = torch.load(
                os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset)).cuda()
        else:
            try:
                os.makedirs(os.path.join(os.path.dirname(args.test), dirname))
            except:
                pass

            print('Computing test set feature ... ')
            test_feature = None
            test_label = []
            test_vname = []
            sample_id = 0
            for idx, (x, (target, vname)) in tqdm(enumerate(test_loader), total=len(test_loader)):
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                logit, feature = model(x)
                if test_feature is None:
                    test_feature = torch.zeros(len(test_dataset), feature.size(-1), device=feature.device)

                test_feature[sample_id, :] = feature.mean(0)
                test_label.append(target)
                test_vname.append(vname)
                sample_id += 1

            print(test_feature.size())
            test_label = torch.cat(test_label, dim=0)
            torch.save(test_feature,
                       os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset))
            torch.save(test_label,
                       os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset))
            with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset),
                      'wb') as fp:
                pickle.dump(test_vname, fp)

        if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)):
            train_feature = torch.load(
                os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)).cuda()
            train_label = torch.load(
                os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset)).cuda()
        else:
            print('Computing train set feature ... ')
            train_feature = None
            train_label = []
            train_vname = []
            sample_id = 0
            for idx, (x, (target, vname)) in tqdm(enumerate(train_loader), total=len(train_loader)):
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                logit, feature = model(x)
                if train_feature is None:
                    train_feature = torch.zeros(len(train_dataset), feature.size(-1), device=feature.device)

                train_feature[sample_id, :] = feature.mean(0)
                train_label.append(target)
                train_vname.append(vname)
                sample_id += 1
            print(train_feature.size())
            train_label = torch.cat(train_label, dim=0)
            torch.save(train_feature,
                       os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset))
            torch.save(train_label,
                       os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset))
            with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset),
                      'wb') as fp:
                pickle.dump(train_vname, fp)

        ks = [1, 5, 10, 20, 50]
        NN_acc = []

        # centering
        test_feature = test_feature - test_feature.mean(dim=0, keepdim=True)
        train_feature = train_feature - train_feature.mean(dim=0, keepdim=True)

        # normalize
        test_feature = F.normalize(test_feature, p=2, dim=1)
        train_feature = F.normalize(train_feature, p=2, dim=1)

        # dot product
        sim = test_feature.matmul(train_feature.t())

        torch.save(sim, os.path.join(os.path.dirname(args.test), dirname, '%s_sim.pth.tar' % args.dataset))

        for k in ks:
            topkval, topkidx = torch.topk(sim, k, dim=1)
            acc = torch.any(train_label[topkidx] == test_label.unsqueeze(1), dim=1).float().mean().item()
            NN_acc.append(acc)
            # print('%dNN acc = %.4f' % (k, acc))

        args.logger.log('NN-Retrieval on %s:' % args.dataset)
        for k, acc in zip(ks, NN_acc):
            args.logger.log('\t%dNN acc = %.4f' % (k, acc))

        sys.exit(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)
