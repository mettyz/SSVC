from data.vDataset import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def data_config(args):
    if args.dataset == 'ucf101':
        args.num_class = 101
        args.video_path = F'{args.db_root}/ucf101/mp4'
        args.class_list = './data/lists/ucf101/ClassInd.txt'
        args.train_list = './data/lists/ucf101/train_split1.txt'
        args.val_list = './data/lists/ucf101/test_split1.txt'
    elif args.dataset == 'hmdb51':
        args.num_class = 51
        args.video_path = F'{args.db_root}/hmdb51/mp4'
        args.class_list = './data/lists/hmdb51/ClassInd.txt'
        args.train_list = './data/lists/hmdb51/train_split1.txt'
        args.val_list = './data/lists/hmdb51/test_split1.txt'
    elif args.dataset == 'k400':
        args.num_class = 400
        args.video_path = F'{args.db_root}/Kinetics/compress'
        args.train_list = './data/lists/k400/trainlist.txt'
        args.val_list = './data/lists/k400/vallist.txt'
    elif args.dataset == 'k200':
        """
        This subset of Kinetics dataset consists of the 200 categories with most training examples; 
        for each category, we randomly sample 
        400 examples from the training set, and 25 examples from the validation set, 
        resulting in 80K training examples and 5K validation examples in total.
        """
        args.num_class = 200
        args.video_path = F'{args.db_root}/Kinetics/compress'
        args.train_list = './data/lists/k200/trainlist.txt'
        args.val_list = './data/lists/k200/vallist.txt'
    else:
        raise ValueError('Unknown dataset ' + args.dataset)


def get_dataloader(args, train_transform=None, val_transform=None):
    data_config(args)

    train_loader = None
    train_sampler = None
    val_loader = None

    if train_transform is not None:
        train_set = VideoDataset(args.train_list, args.video_path, args.double_sample,
                                 transform=train_transform, mode='train',
                                 seq_len=args.seq_len, ds=args.ds,
                                 return_vpath=False,
                                 return_label=True)

        train_sampler = DistributedSampler(train_set) if args.ddp else None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)

    if val_transform is not None:
        val_set = VideoDataset(args.val_list, args.video_path, double_sample=False,
                               transform=val_transform, mode='val', val_all=args.val_all,
                               seq_len=args.seq_len, ds=args.ds,
                               return_vpath=False,
                               return_label=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader, train_sampler, val_loader
