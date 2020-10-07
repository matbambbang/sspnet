from PIL import Image
import os
import os.path
import numpy as np
import sys
from torchvision.datasets.vision import VisionDataset # check whether this successfully imports

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import subset_sampler

# This code is helped by 'meliketoy'(https://github.com/meliketoy).

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_test_dataset(dir, class_to_idx, extensions=None):
    images = []
    dir = os.path.expanduser(dir)

    if not os.path.exists('{}/val_annotations.txt'.format(dir)):
        raise RuntimeError("Validation data is corrupted! Download the data again!")

    d = os.path.join(dir, 'images')
    with open('{}/val_annotations.txt'.format(dir), 'r') as annot:
        for line in annot.readlines():
            image_name, class_name, _, _, _, _ = line.split("\t")
            path = os.path.join(d, image_name)
            if is_image_file(path):
                item = (path, class_to_idx[class_name])
                images.append(item)

    return images

class TINY_IMAGENET(VisionDataset):
    """
    Tiny ImageNet Dataset.
    Args:
        root (string): Root directory of dataset directory
        train (bool, optional): If True, creates dataset from training set, otherwise creates from val set.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform  (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional):  If true, donwloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not donwloaded again.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TINY_IMAGENET, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = (split == 'train')
        self.loader = default_loader
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. ' + ' You can use download=True to download it')
        self.base = '{}/tiny-imagenet-200'.format(root)
        classes, class_to_idx = self._find_classes()
        self.classes = classes
        self.class_to_idx = class_to_idx
        print("| Preparing Tiny-ImageNet {} dataset with {} classes...".format(split, len(self.classes)))

        if self.train:
            samples = make_dataset('{}/train'.format(self.base), class_to_idx, extensions='JPEG')
        else:
            samples = make_test_dataset('{}/val'.format(self.base), class_to_idx, extensions='JPEG')

        self.samples = samples
        self.targets = [s[1] for s in samples]

        print("| {} dataset with {} samples.".format(split, len(samples)))

    def _check_integrity(self):
        root = self.root
        if os.path.isdir('{}/tiny-imagenet-200/'.format(self.root)):
            if os.path.exists('{}/tiny-imagenet-200/wnids.txt'.format(self.root)):
                if os.path.isdir('{}/tiny-imagenet-200/train'.format(self.root)):
                    if os.path.isdir('{}/tiny-imagenet-200/train'.format(self.root)):
                        return True
        return False

    def download(self):
        if not os.path.exists('{}/tiny-imagenet-200.zip'.format(self.root)):
            os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P {}'.format(self.root)) # move to data directory
        if not os.path.isdir('{}/tiny-imagenet-200/'.format(self.root)):
            os.system('unzip {}/tiny-imagenet-200.zip -d {}'.format(self.root, self.root))
        return

    def _find_classes(self):
        with open('{}/wnids.txt'.format(self.base), 'r') as label_txt:
            classes = [l[:-1] for l in label_txt.readlines()]
        classes.sort()
        class_to_idx = {classes[i]:i for i in range(len(classes))}

        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

def get_tinyimagenet_loaders(data_aug=True, batch_size=128, test_batch_size=500, path="./data/tiny_imagenet") :
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    if data_aug :
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomAffine(degrees=5.),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=0, scale=(0.1, 0.9)),
            transforms.RandomAffine(degrees=0, shear=5.),
            transforms.RandomPerspective(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ]

        transform_train = transforms.Compose([
            transforms.RandomApply(transform_list),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    else :
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

    train_loader = DataLoader(
        TINY_IMAGENET(root=path, split='train', download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True
    )
    eval_loader = DataLoader(
        TINY_IMAGENET(root=path, split='train', download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=10, drop_last=True
    )

    test_loader = DataLoader(
        TINY_IMAGENET(root=path, split='validation', download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=10, drop_last=True
    )

    return train_loader, test_loader, eval_loader
