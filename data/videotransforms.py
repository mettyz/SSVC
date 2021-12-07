import torch
import random
import numpy as np
from PIL import ImageFilter
import torchvision.transforms.transforms as T
import torchvision.transforms.functional as TF


def clip_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # BCTHW
    mean = torch.as_tensor(mean, dtype=torch.float32, device=x.device).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32, device=x.device).view(1, -1, 1, 1, 1)
    return (x - mean) / std


class BatchClipTransforms(object):
    def __init__(self, transform):
        self.transform = T.Compose([
            ToPILImage(),
            transform.transforms,
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToClip()
        ])

    def __call__(self, clip):
        assert len(clip.shape) == 5
        # BCTHW -> B[TCHW]
        batch_clips = torch.split(clip.permute((0, 2, 1, 3, 4)), 1, dim=0)  # tensor -> list of tensor, len=bsz
        batch_clips = [self.transform(clip) for clip in batch_clips]
        return torch.stack(batch_clips)


class TransformController(object):
    def __init__(self, transform_list, weights):
        self.transform_list = transform_list
        self.weights = weights
        self.num_transform = len(transform_list)
        assert self.num_transform == len(self.weights)

    def __call__(self, q, k):
        idx = random.choices(range(self.num_transform), weights=self.weights)[0]
        return self.transform_list[idx](q, k)

    def __str__(self):
        string = 'TransformController: %s with weights: %s' % (str(self.transform_list), str(self.weights))
        return string


class Dummy(object):
    def __call__(self, pics):
        return pics


class ToClip(object):
    """
    list of tensor -> tensor  CTHW
    """
    def __call__(self, pics):
        return torch.stack(pics, dim=1)


class ToTensor(T.ToTensor):
    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [TF.to_tensor(pic) for pic in pics]  # (TxCHW, range[0,1])


class Normalize(T.Normalize):
    def __call__(self, pics):
        return [TF.normalize(pic, self.mean, self.std, self.inplace) for pic in pics]


class ToPILImage(T.ToPILImage):
    def __call__(self, pics):
        return [TF.to_pil_image(pic, self.mode) for pic in pics]


class Resize(T.Resize):
    def __call__(self, pics):
        return [TF.resize(pic, self.size, self.interpolation) for pic in pics]


class CenterCrop(T.CenterCrop):
    def __call__(self, imgs):
        return [TF.center_crop(img, self.size) for img in imgs]


class RandomCrop(T.RandomCrop):
    def __call__(self, imgs):
        if self.padding is not None:
            imgs = [TF.pad(img, self.padding, self.fill, self.padding_mode) for img in imgs]

        # pad the width if needed
        if self.pad_if_needed and imgs[0].size[0] < self.size[1]:
            imgs = [TF.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode) for img in imgs]
        # pad the height if needed
        if self.pad_if_needed and imgs[0].size[1] < self.size[0]:
            imgs = [TF.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode) for img in imgs]

        i, j, h, w = self.get_params(imgs[0], self.size)

        return [TF.crop(img, i, j, h, w) for img in imgs]


class RandomResizedCrop(T.RandomResizedCrop):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [TF.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in imgs]


class RandomRotation(T.RandomRotation):
    def __call__(self, imgs):
        angle = self.get_params(self.degrees)

        return [TF.rotate(img, angle, self.resample, self.expand, self.center) for img in imgs]


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, imgs):
        """
        Default self.p=0.5, can be modified while init
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [TF.hflip(img) for img in imgs]
        return imgs


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, per_img=False):
        super().__init__(brightness, contrast, saturation, hue)
        self.per_img = per_img  # decide whether do the same augment for the whole clip or not

    def __call__(self, imgs):
        if self.per_img:
            trans = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            trans = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return [trans(img) for img in imgs]


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, imgs):
        sigma = np.random.uniform(0.1, 2.0)
        return [img.filter(ImageFilter.GaussianBlur(radius=sigma)) for img in imgs]


class RandomGrayscale(T.RandomGrayscale):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if imgs[0].mode == 'L' else 3
        if random.random() < self.p:
            return [TF.to_grayscale(img, num_output_channels=num_output_channels) for img in imgs]
        return imgs


class FiveCrop(object):
    def __init__(self, size, where=1):
        # 1=topleft, 2=topright, 3=botleft, 4=botright, 5=center
        self.size = size
        self.where = where

    def __call__(self, imgs):
        image_width, image_height = imgs[0].size
        crop_height, crop_width = self.size, self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        if self.where == 1:
            return [TF.crop(img, 0, 0, crop_height, crop_width) for img in imgs]
        elif self.where == 2:
            return [TF.crop(img, 0, image_width - crop_width, crop_height, crop_width) for img in imgs]
        elif self.where == 3:
            return [TF.crop(img, image_height - crop_height, 0, crop_height, crop_width) for img in imgs]
        elif self.where == 4:
            return [TF.crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width) for img in imgs]
        elif self.where == 5:
            return [TF.center_crop(img, [crop_height, crop_width]) for img in imgs]
        else:
            raise NotImplementedError('Wrong crop position')
