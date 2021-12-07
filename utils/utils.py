import torch

import os
import glob
import shutil
import time


class Logger(object):
    def __init__(self, path):
        self.birth_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.filepath = os.path.join(path, self.birth_time + '.log')  # create a log file named as localtime in log_path
        with open(self.filepath, 'a+') as f:
            print('Log file Created')

    def log(self, info, verbose=True):
        """ set verbose=False if you don't want the log info be print """
        with open(self.filepath, 'a+') as f:
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(time_stamp + '\t' + info + '\n')
        if verbose:
            print('Log:', info)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    maxk = min(maxk, output.shape[-1])

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False):
    try:
        torch.save(state, filename, _use_new_zipfile_serialization=False)  # Compatible with older torch version
    except:
        torch.save(state, filename)

    last_epoch_path = os.path.join(os.path.dirname(filename), 'epoch%s.pth.tar' % str(state['epoch']-gap))
    if not keep_all:
        try: os.remove(last_epoch_path)
        except: pass

    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 3:
            try: os.remove(past_best[0])
            except: pass
        try:
            torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])),
                       _use_new_zipfile_serialization=False)
        except:
            torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))


def neq_load_customized(model, pretrained_dict, verbose=True):
    """ load pre-trained model in a not-equal way,
    when new model has been partially modified """
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                print(k)
        print('---------------------------')
        print('Weights not loaded into new model:')
        cnt = 20
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                if 'INN' not in k: cnt -= 1
                if cnt <= 0: raise ValueError('Too much unloaded params, do a double check?')
                print(k)
        print('===================================\n')
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def adjust_learning_rate(optimizer, epoch, args):
    """ Decay the learning rate based on schedule """
    print('lr: {}'.format(optimizer.param_groups[0]['lr']), end=' -> ')

    if args.warmup > 0 and epoch < args.warmup:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (args.lr / args.warmup) * (epoch + 1)
    elif args.warmup > 0 and args.warmup == epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    else:
        ratio = 0.1 if epoch in args.schedule else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * ratio

    print('{}'.format(optimizer.param_groups[0]['lr']))
    try:
        args.logger.log('new lr: {}'.format(optimizer.param_groups[0]['lr']))
    except:
        pass
