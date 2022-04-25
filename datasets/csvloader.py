# data loader for multitask-learning

from __future__ import print_function, division
from os.path import join, isfile
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
import math

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(im_path, csv_path, extensions=None, is_valid_file=None):
    # IMG_ID,IMG_PATH,BBOX_ID,XMIN,YMIN,XMAX,YMAX,LOCAL_ID
    samples = pd.read_csv(csv_path, header=0)
    image_paths_ = samples['IMG_PATH'].tolist()
    labels_ = samples['LOCAL_ID'].tolist()
    xmins_ = samples['XMIN'].tolist()
    ymins_ = samples['YMIN'].tolist()
    xmaxs_ = samples['XMAX'].tolist()
    ymaxs_ = samples['YMAX'].tolist()
    image_paths, labels, xmins, ymins, xmaxs, ymaxs = [], [], [], [], [], []

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions) and isfile(x)

    # Believe the given path is correct
    labels = labels_
    image_paths = image_paths_
    xmins = xmins_
    ymins = ymins_
    xmaxs = xmaxs_
    ymaxs = ymaxs_

    return image_paths, labels, xmins, ymins, xmaxs, ymaxs


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


class CSVDataset(Dataset):
    def __init__(
            self,
            image_root=None,
            csv_root=None,
            split='train',
            transform=None
    ):
        self.split = split
        self.image_root = image_root
        self.transform = transform

        if self.split == 'train':
            self.csv_path = join(csv_root, 'train.csv')
        elif self.split == 'test':
            self.csv_path = join(csv_root, 'test.csv')

        samples = make_dataset(self.image_root, self.csv_path, extensions=IMG_EXTENSIONS)
        image_paths = samples[0]
        labels = samples[1]
        self.imgs = [(p, l) for p, l in zip(image_paths, labels)]
        self.xmins = samples[2]
        self.ymins = samples[3]
        self.xmaxs = samples[4]
        self.ymaxs = samples[5]

        # TODO: If some label has zero samples, it is removed from the list (and it makes number of classes different)
        self.classes = sorted(set(labels))

        self.loader = default_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # try:
        image_path = join(self.image_root, self.imgs[idx][0])
        labels = self.imgs[idx][1]
        xmin = self.xmins[idx]
        ymin = self.ymins[idx]
        xmax = self.xmaxs[idx]
        ymax = self.ymaxs[idx]

        image = self.loader(image_path)

        if xmin == xmax:
            xmin = 0
            xmax = image.size[0]
        if ymin == ymax:
            ymin = 0
            ymax = image.size[1]

        # crop_image = self.crop(image, [xmin, ymin, xmax, ymax], self.margin)
        crop_image = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            crop_image = self.transform(crop_image)

        return crop_image, labels
        # except:
        #     return self.__getitem__(math.floor(random.random()*len(self.imgs)))

    # def crop(self, image, bboxes, margin):
    #     w, h = image.size

    #     bbox_w = bboxes[2] - bboxes[0]
    #     bbox_h = bboxes[3] - bboxes[1]

    #     if isinstance(margin, tuple):
    #         margin_w = bbox_w * margin[0]
    #         margin_h = bbox_h * margin[1]
    #     elif margin == 'bottom-full':
    #         margin_w = 0.0
    #         margin_h = h - bboxes[3]

    #     xmin = max(bboxes[0] - margin_w, 0)
    #     ymin = max(bboxes[1] - margin_h, 0)
    #     xmax = min(bboxes[2] + margin_w, w)
    #     ymax = min(bboxes[3] + margin_h, h)

    #     crop_image = image.crop((xmin, ymin, xmax, ymax))
    #     return crop_image


if __name__ == '__main__':
    dataset = CSVDataset(
        image_root='/data/shared/images',
        csv_root='/data/shared/tdc/tagger_2.0_tdc_dataset/print/csv',
        split='train',
        transform=None
    )
