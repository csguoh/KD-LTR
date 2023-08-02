#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import cv2
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
from utils.util import str_filt
import imgaug.augmenters as iaa
from setup import CharsetMapper
from model.parseq.parseq_tokenizer import get_parseq_tokenize


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


class lmdbDataset_real(Dataset):
    def __init__(
            self,
            config,
            args,
            root=None,
            voc_type='all',
            max_len=100,
    ):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("nSamples:", nSamples)
        self.voc_type = voc_type
        self.cfg = config
        self.args =args
        self.max_len = max_len+1

        if self.args.rec_backbone in ['PARSeq']:
            self.charset= get_parseq_tokenize()
        else:
            self.charset =CharsetMapper(self.cfg.ABINet.dataset_charset_path, max_length=self.max_len)



    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = ""
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        post_pro = self.charset.label_postprocessing(word)

        return img_HR, img_lr, post_pro['label'], post_pro['length'],label_str


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples



class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC, aug=None):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.aug = aug

    def __call__(self, img):
        size = self.size
        img = img.resize(size, self.interpolation)
        if not self.aug is None:
            img_np = np.array(img)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])
        img_tensor = self.toTensor(img)
        return img_tensor


class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 train=True
                 ):

        sometimes = lambda aug: iaa.Sometimes(0.25, aug)
        aug = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(
                d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=3),
            iaa.MeanShiftBlur(),
            iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ]

        self.aug = iaa.Sequential([sometimes(a) for a in aug], random_order=True)

        self.imgH = imgH
        self.imgW = imgW
        self.down_sample_scale = down_sample_scale
        self.train = train

class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, str_one_hot, str_len, str_label = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW*self.down_sample_scale, imgH*self.down_sample_scale))
        transform2 = resizeNormalize((imgW, imgH)) # resize only
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        gt_dict = dict()
        gt_dict['label'] = torch.stack(str_one_hot,dim=0)
        gt_dict['length'] = torch.tensor(str_len)

        return images_HR, images_lr, gt_dict, str_label


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes




class lmdbDataset_realIC15(Dataset):
    def __init__(
            self,
            config,
            args,
            root=None,
            voc_type='all',
            max_len=100,
    ):
        super(lmdbDataset_realIC15, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("nSamples:", nSamples)
        self.voc_type = voc_type
        self.cfg = config
        self.args = args
        self.max_len = max_len+1

        if self.args.rec_backbone in ['PARSeq']:
            self.charset= get_parseq_tokenize()
        else:
            self.charset =CharsetMapper(self.cfg.ABINet.dataset_charset_path, max_length=self.max_len)

    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = ""
        img_key = b'image-%09d' % index  # 128*32
        try:
            img_HR = buf2PIL(txn, img_key, 'RGB')
            img_lr = None
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        post_pro = self.charset.label_postprocessing(word)

        return img_HR, img_lr, post_pro['label'], post_pro['length'],label_str



class alignCollate_IC15(alignCollate_syn):
    def __call__(self, batch):
        images_HR_origin, images_lr, str_one_hot, str_len, str_label = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # resize only
        transform = resizeNormalize((imgW*self.down_sample_scale, imgH*self.down_sample_scale))
        transform_pseudoLR = resizeNormalize((imgW, imgH),aug=self.aug)

        images_HR = [transform(image) for image in images_HR_origin]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform_pseudoLR(image) for image in images_HR_origin]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        gt_dict = dict()
        gt_dict['label'] = torch.stack(str_one_hot,dim=0)
        gt_dict['length'] = torch.tensor(str_len)

        return images_HR, images_lr, gt_dict, str_label
