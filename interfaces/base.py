import logging
import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
import string
import ptflops
from dataset import alignCollate_real, lmdbDataset_real,lmdbDataset_realIC15,alignCollate_IC15
from loss.DistillLoss import Distill_Loss,PARSeq_distill_loss

from model.ABINet.ABINet import ABINet_model,BaseVision
from model.ABINet.LowRes_VisionRecognizer import LowRes_BaseVision,LowRes_ABINet
from model.MATRN.matrn import MATRN,LowRes_MATRN
from utils import ssim_psnr
import dataset.dataset as dataset
from setup import CharsetMapper
from model.parseq.parseq_tokenizer import get_parseq_tokenize

from model.parseq.parseq import PARSeq
from model.parseq.parseq_LR import LowRes_PARSeq

class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale

        self.align_collate = alignCollate_real
        self.load_dataset = lmdbDataset_real

        # 1 IC15
        if 'IC15' in config.TRAIN.VAL.val_data_dir[0]:
            self.align_collate_val =  alignCollate_IC15
            self.load_dataset_val = lmdbDataset_realIC15
        # 2 real TextZoom
        else:
            self.align_collate_val = alignCollate_real
            self.load_dataset_val = lmdbDataset_real

        self.resume = config.TRAIN.resume
        self.batch_size = self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
        }
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.ckpt_path = self.config.TRAIN.ckpt_dir

        if self.args.rec_backbone in ['PARSeq']:
            self.charsetMapper= get_parseq_tokenize()
        else:
            self.charsetMapper = CharsetMapper(self.config.ABINet.dataset_charset_path, max_length=self.config.TRAIN.max_len)



    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(config=self.config,
                                      args=self.args,
                                      root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len,
                ))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          train=True), drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN

        test_dataset = self.load_dataset_val(root=dir_,
                                            config=self.config,
                                            args=self.args,
                                            voc_type=cfg.voc_type,
                                            max_len=cfg.max_len)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=48,# bug for multii-gpu
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width,
                                              down_sample_scale=cfg.down_sample_scale,
                                              train=False),
            drop_last=False)
        return test_dataset, test_loader


    def HRteacher_init(self):
        # tea model
        if self.args.rec_backbone == 'BaseVision':
            cfg = self.config.ABINet
            tea = BaseVision(cfg)

        elif self.args.rec_backbone == 'ABINet':
            cfg = self.config.ABINet
            tea = ABINet_model(cfg)

        elif self.args.rec_backbone == 'MATRN':
            cfg = self.config.MATRN
            tea = MATRN(cfg)

        elif self.args.rec_backbone == 'PARSeq':
            cfg = self.config.PARSeq
            tea = PARSeq(cfg)


        tea = tea.to(self.device)
        if self.config.TRAIN.ngpu > 1:
            tea = torch.nn.DataParallel(tea, device_ids=range(self.config.TRAIN.ngpu))
        for p in tea.parameters():
            p.requires_grad = False
        tea.eval()
        return tea


    def generator_init(self):
        # load student model

        if self.args.rec_backbone == 'BaseVision':
            model = LowRes_BaseVision(config=self.config,args=self.args)
            image_crit = Distill_Loss(self.config)

        elif self.args.rec_backbone == 'ABINet':
            model = LowRes_ABINet(config=self.config,args=self.args)
            image_crit = Distill_Loss(self.config)

        elif self.args.rec_backbone == 'MATRN':
            model = LowRes_MATRN(config=self.config,args=self.args)
            image_crit = Distill_Loss(self.config)

        elif self.args.rec_backbone == 'PARSeq':
            model = LowRes_PARSeq(config=self.config.PARSeq)
            image_crit = PARSeq_distill_loss(self.config)

        model = model.to(self.device)
        image_crit = image_crit.to(self.device)


        if self.config.TRAIN.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(self.config.TRAIN.ngpu))
            image_crit = torch.nn.DataParallel(image_crit, device_ids=range(self.config.TRAIN.ngpu))

        resume = self.resume
        if not resume == '' and self.args.go_test:
            logging.info('loading pre-trained model from %s ' % resume)
            if self.config.TRAIN.ngpu == 1:
                if os.path.isdir(resume):
                    model_dict = torch.load(resume)['state_dict_G']
                    model.load_state_dict(model_dict, strict=False)
                else:
                    loaded_state_dict = torch.load(resume)
                    if 'state_dict_G' in loaded_state_dict:
                        model.load_state_dict(torch.load(resume)['state_dict_G'])
                    else:
                        model.load_state_dict(torch.load(resume))
            else:
                model_dict = torch.load(resume)['state_dict_G']

                if os.path.isdir(resume):
                    model.load_state_dict({'module.' + k: v for k, v in model_dict.items()}, strict=False)
                else:
                    model.load_state_dict({'module.' + k: v for k, v in torch.load(resume)['state_dict_G'].items()})


        return {'model': model, 'crit': image_crit}


    def optimizer_init(self, model,loss):
        cfg = self.config.TRAIN
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and 'patch_embed.proj.weight' not in n]}, # ABINet
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and 'patch_embed.proj.weight' in n],"lr":1e-4},
            {"params": [p for n, p in loss.named_parameters() if p.requires_grad],"lr":1e-4},  # loss proj
        ]
        optimizer = optim.Adam(param_dicts, lr=cfg.Recognizer_lr,
                               betas=(cfg.beta1, 0.999))

        return optimizer

    def save_checkpoint(self, netG, best_model_info, is_best):

        ckpt_path = self.ckpt_path
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        if self.config.TRAIN.ngpu > 1:
            netG = netG.module
        else:
            netG = netG

        save_dict = {
            'state_dict_G': netG.state_dict(),
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.parameters()]),
        }

        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best_' + '.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))