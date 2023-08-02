import torch
import torch.nn as nn
from model.ABINet.ABINet import BaseVision,BaseAlignment,BCNLanguage,Model
from model.ABINet.LowRes_VisionRecognizer import LowRes_BaseVision
from model.MATRN.sematic_visual_backbone import BaseSemanticVisual_backbone_feature
import torch.nn.functional as F
import logging

class MATRN(Model):
    def __init__(self, config):
        super().__init__(config)
        self.iter_size = 3
        self.test_bh = None
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.semantic_visual = BaseSemanticVisual_backbone_feature(config)
        self.max_length = config.dataset_max_length + 1  # additional stop token

        # load full model--> Vision Language Align
        if config.full_ckpt is not None:
            logging.info(f'Read full ckpt model from {config.full_ckpt}.')
            self.load(config.full_ckpt)


    def forward(self, images,input_lr=False,normalize=True,common_info=None):
        v_res = self.vision(images,input_lr,normalize)
        a_res = v_res
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)
            l_res = self.language(tokens, lengths)

            lengths_l = l_res['pt_lengths']
            lengths_l.clamp_(2, self.max_length)

            v_attn_input = v_res['attn_scores'].clone().detach()
            l_logits_input = None
            texts_input = None

            a_res = self.semantic_visual(l_res['feature'], v_res['backbone_feature'], lengths_l=lengths_l, v_attn=v_attn_input, l_logits=l_logits_input, texts=texts_input, training=self.training)


        v_res['logits'] = a_res['logits']

        return v_res


class LowRes_MATRN(Model):
    def __init__(self, config, args):
        super().__init__(config.MATRN)
        self.iter_size = 3
        self.test_bh = None
        self.vision = LowRes_BaseVision(config, args)
        config = config.MATRN
        self.language = BCNLanguage(config)
        self.semantic_visual = BaseSemanticVisual_backbone_feature(config)

        # load full model--> Vision Language Align
        if config.full_ckpt is not None:
            logging.info(f'Student (MATRN-full) model reads pretrained param from {config.full_ckpt}.')
            MATRN_state_dict = torch.load(config.full_ckpt)['model']
            MATRN_LR_state_dict = self.state_dict()
            pretrained_state_dict = {k: v for k, v in MATRN_state_dict.items() if k in MATRN_LR_state_dict}
            MATRN_LR_state_dict.update(pretrained_state_dict)
            self.load_state_dict(MATRN_LR_state_dict)


    def forward(self, images,common_info=None):
        v_res = self.vision(images)
        a_res = v_res
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)
            l_res = self.language(tokens, lengths)

            lengths_l = l_res['pt_lengths']
            lengths_l.clamp_(2, self.max_length)

            v_attn_input = v_res['attn_scores'].clone().detach()
            l_logits_input = None
            texts_input = None

            a_res = self.semantic_visual(l_res['feature'], v_res['backbone_feature'], lengths_l=lengths_l, v_attn=v_attn_input, l_logits=l_logits_input, texts=texts_input, training=self.training)

        v_res['logits'] = a_res['logits']

        return v_res