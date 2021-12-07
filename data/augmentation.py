from torchvision.transforms import transforms
import data.videotransforms as vT


def get_train_val_augment(args):
    if args.setting == 'ssl':
        null_transform = transforms.Compose([
            vT.ToPILImage(),
            vT.RandomResizedCrop(224, scale=(0.2, 1.)),
            vT.Resize((args.img_dim, args.img_dim)),
            vT.RandomHorizontalFlip(p=0.5),
            vT.ToTensor(),
            vT.ToClip()
        ])

        base_transform = transforms.Compose([
            vT.ToPILImage(),
            vT.RandomResizedCrop(224, scale=(0.2, 1.)),
            vT.Resize((args.img_dim, args.img_dim)),
            vT.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([vT.ColorJitter(0.4, 0.4, 0.4, 0.1, per_img=True)], p=0.8),
            vT.RandomGrayscale(p=0.2),
            transforms.RandomApply([vT.GaussianBlur()], p=0.5),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.ToClip()
        ])
        train_transform = (null_transform, base_transform)
        val_transform = None

    elif args.setting == 'sup':
        train_transform = transforms.Compose([
            vT.ToPILImage(),
            vT.RandomResizedCrop(224, scale=(0.2, 1.)),
            vT.Resize((args.img_dim, args.img_dim)),
            vT.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([vT.ColorJitter(0.4, 0.4, 0.4, 0.1, per_img=False)], p=0.3),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.ToClip()
        ])
        val_transform = transforms.Compose([
            vT.ToPILImage(),
            vT.RandomResizedCrop(224, scale=(0.2, 1.)),
            vT.Resize((args.img_dim, args.img_dim)),
            vT.ToTensor(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vT.ToClip()
        ])
    else:
        raise NotImplementedError

    return train_transform, val_transform
