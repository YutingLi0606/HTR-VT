import numpy as np
import torch
import skimage
import os
import itertools
from PIL import Image
from torch.utils.data import Dataset
from utils import utils
from data import transform as transform
from torchvision.transforms import ColorJitter


def SameTrCollate(batch, args):

    images, labels = zip(*batch)
    images = [Image.fromarray(np.uint8(images[i][0] * 255)) for i in range(len(images))]

    # Apply data augmentations with 90% probability
    if np.random.rand() < 0.5:
        images = [transform.RandomTransform(args.proj)(image) for image in images]

    if np.random.rand() < 0.5:
        kernel_h = utils.randint(1, args.dila_ero_max_kernel + 1)
        kernel_w = utils.randint(1, args.dila_ero_max_kernel + 1)
        if utils.randint(0, 2) == 0:
            images = [transform.Erosion((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in images]
        else:
            images = [transform.Dilation((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in images]

    if np.random.rand() < 0.5:
        images = [ColorJitter(args.jitter_brightness, args.jitter_contrast, args.jitter_saturation,
                              args.jitter_hue)(image) for image in images]

    # Convert images to tensors

    image_tensors = [torch.from_numpy(np.array(image, copy=True)) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    image_tensors = image_tensors.unsqueeze(1).float()
    image_tensors = image_tensors / 255.
    return image_tensors, labels


class myLoadDS(Dataset):
    def __init__(self, flist, dpath, img_size=[512, 32], ralph=None, fmin=True, mln=None):
        self.fns = get_files(flist, dpath)
        self.tlbls = get_labels(self.fns)
        self.img_size = img_size

        if ralph == None:
            alph = get_alphabet(self.tlbls)
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph

        if mln != None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        timgs = get_images(self.fns[index], self.img_size[0], self.img_size[1])
        timgs = timgs.transpose((2, 0, 1))

        return (timgs, self.tlbls[index])


def get_files(nfile, dpath):
    fnames = open(nfile, 'r').readlines()
    fnames = [dpath + x.strip() for x in fnames]
    return fnames


def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int(y * max_h / x), max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y, x)))
    return img


def get_images(fname, max_w=500, max_h=500, nch=1):  # args.max_w args.max_h args.nch

    try:

        image_data = np.array(Image.open(fname).convert('L'))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)

        image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant',
                            constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)

    return image_data


def get_labels(fnames):
    labels = []
    for id, image_file in enumerate(fnames):
        fn = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        lbl = ' '.join(lbl.split())  # remove linebreaks if present

        labels.append(lbl)

    return labels


def get_alphabet(labels):
    coll = ''.join(labels)
    unq = sorted(list(set(coll)))
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    alph = dict(zip(unq, range(len(unq))))

    return alph


def cycle_dpp(iterable):
    epoch = 0
    iterable.sampler.set_epoch(epoch)
    while True:
        for x in iterable:
            yield x
        epoch += 1
        iterable.sampler.set_epoch(epoch)


def cycle_data(iterable):
    while True:
        for x in iterable:
            yield x

