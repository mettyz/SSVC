import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import numpy as np
from backbone.select_backbone import select_model
import data.videotransforms as vT

import random

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, backbone,
                 use_ddp, rank,
                 threshold, aug_type,
                 dim_out=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.dim_out = dim_out
        self.K = K
        self.m = m
        self.T = T
        self.ddp = use_ddp
        self.rank = rank
        self.threshold = threshold
        self.aug_type = aug_type

        print(f'aug_type {aug_type}, threshold: {threshold}')

        # create the encoders
        self.base_q, param = select_model(backbone)
        self.base_k, _ = select_model(backbone)
        feature_size = param['feature_size']

        self.encoder_q = nn.Sequential(
            self.base_q,
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(feature_size, self.dim_out, kernel_size=1, bias=True)
        )
        self.encoder_k = nn.Sequential(
            self.base_k,
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(feature_size, self.dim_out, kernel_size=1, bias=True)
        )

        import model.advflow.advflow as advflow
        self.flow_pth = './flow/ImageNet_6_blocks_Simple_Gaussian.pt'
        advflow.load(self.flow_pth)

        self.INN = advflow.model
        for p in self.INN.parameters():
            p.requires_grad = False

        # copy initialize parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim_out, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.ddp:
            # gather keys before updating queue
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _change_background(self, x):
        self.INN.to(x.device)
        bsz, c, seq_len, h, w = x.shape

        # Downsample
        orig = x.permute(0, 2, 1, 3, 4).contiguous().reshape(bsz * seq_len, c, h, w)
        orig = F.adaptive_avg_pool2d(orig, (64, 64))
        encode = self.INN(orig)

        # Find temporal invariant channels
        encode = encode.reshape(bsz, seq_len, -1)
        tmp = encode.std(dim=1)

        # Suppress Static Visual Cues
        if self.aug_type == 1:
            # constant alpha
            static_cues = (tmp < self.threshold).unsqueeze_(1).expand(-1, seq_len, -1)
        elif self.aug_type == 2:
            # auto alpha keep Top k% active channels
            threshold, _ = tmp.topk(int(self.threshold * tmp.shape[1]), dim=1)  # B x 12288
            threshold = threshold[:, -1]  # B x 1
            static_cues = (tmp < threshold.unsqueeze_(1)).unsqueeze_(1).expand(-1, seq_len, -1)
        else:
            raise NotImplementedError('Unrecognized Augment Type')

        encode[static_cues] = 0.

        # Reverse & UpSample ============================================
        encode = encode.reshape(bsz * seq_len, -1)
        fp = self.INN(encode.data, rev=True)

        # residual-like process
        diff = fp - orig
        diff = F.interpolate(diff, (h, w))
        fp = x + diff.reshape(bsz, seq_len, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        return fp

    def forward(self, x):
        bsz = x.shape[0]
        im_q, im_k = x[:, 0].contiguous(), x[:, 1].contiguous()

        # change bkgd
        with torch.no_grad():
            im_k = self._change_background(im_k)
            im_k = vT.clip_normalize(im_k)

        # compute query features
        q = self.encoder_q(im_q)  # queries: BxC
        q = nn.functional.normalize(q, dim=1)
        q = q.reshape(bsz, self.dim_out).contiguous()

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.ddp:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: BxC
            k = nn.functional.normalize(k, dim=1)

            if self.ddp:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.reshape(bsz, self.dim_out).contiguous()

        # contrastive learning
        l_pos = (q * k).sum(1, keepdim=True)  # positive logits: Bx1
        l_neg = q.matmul(self.queue.clone().detach())  # negative logits: BxK
        logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Bx(1+K)

        # apply temperature
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(non_blocking=True)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
