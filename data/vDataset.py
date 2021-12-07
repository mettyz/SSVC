import torch
from torch.utils.data import Dataset

import random
import numpy as np

import os
import json
import lintel
from tqdm import tqdm
import data.videotransforms as vT


class VideoDataset(Dataset):
    def __init__(self, list_file, video_path, double_sample,
                 transform=None, mode='val', val_all=False,
                 seq_len=32, ds=1,
                 return_vpath=False,
                 return_label=False):
        self.list_file = list_file
        self.video_path = video_path
        self.transform = transform
        self.mode = mode
        self.seq_len = seq_len
        self.ds = ds
        self.double_sample = double_sample

        self.return_vpath = return_vpath
        self.return_label = return_label

        print('='*20)
        print(f'Loading Dataset from {self.video_path}, list file: {self.list_file}')

        invalid_vpath_cnt = 0
        video_info = []
        too_short_vpath = []
        with open(self.list_file, 'r', encoding='utf-8') as f:
            rows = f.readlines()
        for row in rows:
            vpath, vlabel, vlen = row.strip().split()
            if not os.path.exists(os.path.join(self.video_path, vpath)):
                invalid_vpath_cnt += 1
            else:
                if int(vlen) - self.seq_len // 2 * self.ds - 1 <= 0:  # allow max padding = half video
                    too_short_vpath.append(vpath)
                video_info.append((vpath, int(vlabel), int(vlen) - 1))

        if mode == 'val' and not val_all:
            video_info = random.sample(video_info, int(0.3 * len(video_info)))
        self.video_info = video_info

        print(f'\t{len(self.video_info)} samples, {invalid_vpath_cnt} missing, {len(too_short_vpath)} Too short.')

    def frame_sampler(self, total):
        if self.mode == 'test':  # half overlap - 1
            if total - self.seq_len * self.ds <= 0:  # pad left, only sample once
                sequence = np.arange(self.seq_len) * self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total - self.seq_len * self.ds
                start = np.expand_dims(np.arange(0, available + 1, self.seq_len * self.ds // 2 - 1), 1)
                seq_idx = np.expand_dims(np.arange(self.seq_len) * self.ds, 0) + start  # [test_sample, seq_len]
                seq_idx = seq_idx.reshape(-1)
        else:  # train or val
            if total - self.seq_len * self.ds <= 0:  # pad left
                sequence = np.arange(self.seq_len) * self.ds + np.random.choice(range(self.ds), 1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total - self.seq_len * self.ds), 1)
                seq_idx = np.arange(self.seq_len) * self.ds + start
        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, vlabel, vlen = self.video_info[index]
        if self.double_sample:
            frame_index = self.double_sampler(vlen)
        else:
            frame_index = self.frame_sampler(vlen)
            
        with open(os.path.join(self.video_path, vpath), 'rb') as f:
            video = f.read()
        indices = np.arange(vlen) + 1
        video, width, height = lintel.loadvid_frame_nums(video, frame_nums=indices, should_seek=True)
        video = np.frombuffer(video, dtype=np.uint8)
        video = np.reshape(video, newshape=(vlen, height, width, 3))

        seq = [video[i] for i in frame_index]

        if self.double_sample and isinstance(self.transform, tuple) and len(self.transform) == 2:
            null_transform, base_transform = self.transform
            seq1 = base_transform(seq[:self.seq_len])
            seq2 = null_transform(seq[self.seq_len:])
            seq = torch.stack([seq1, seq2])

        elif self.transform is not None:
            seq = self.transform(seq)
        else:
            raise NotImplementedError('Invalid Transformation')

        if self.return_label:
            if self.return_vpath:
                return seq, (vlabel, vpath)
            else:
                return seq, vlabel
        return seq

    def __len__(self):
        return len(self.video_info)

