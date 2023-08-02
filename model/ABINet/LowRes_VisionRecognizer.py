import torch
import logging
from model.ABINet.attention import *
from model.ABINet.backbone import ResTranformer,ResTranformer_LR
from model.ABINet.resnet import resnet45
from torch.nn import functional as F
from model.ABINet.ABINet import Model,BCNLanguage,BaseVision,BaseAlignment

_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu')

class LowRes_BaseVision(Model):
    def __init__(self,config,args):
        super().__init__(config.ABINet)
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        config = config.ABINet
        self.out_channels = config.vision.d_model if config.vision.d_model is not None else 512

        # ====== Backbone-- Resnet45+Encoder  for  Feature Exaction=============
        if config.vision.backbone == 'transformer':
            self.backbone = ResTranformer_LR(config)
        else:
            self.backbone = resnet45()
        # =============== get attn map ======================================
        if config.vision.attention == 'position':
            mode = config.model_vision_attention_mode if config.model_vision_attention_mode is not None else 'nearest'
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.vision.attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )

        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.vision.checkpoint is not None:
            logging.info(f'Student model reads pretrained param from {config.vision.checkpoint}.')
            ABINet_state_dict = torch.load(config.vision.checkpoint)['model']
            ABINet_LR_state_dict = self.state_dict()
            pretrained_state_dict = {k:v for k,v in ABINet_state_dict.items() if k in ABINet_LR_state_dict}
            ABINet_LR_state_dict.update(pretrained_state_dict)
            self.load_state_dict(ABINet_LR_state_dict)


    def forward(self, images,normalize=True,common_info=None):
        device = images.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if normalize:
            images = (images - self.mean[..., None, None])/self.std[..., None, None]

        features, fpn_feature = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs)  # (N, T, C)
        pt_lengths = self._get_length(logits)
        return {'visual_feat': fpn_feature, 'sematic_feat': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores,'backbone_feature':features}




class LowRes_ABINet(Model):
    def __init__(self, config,args):
        super().__init__(config.ABINet)
        self.vision = LowRes_BaseVision(config,args)
        config = config.ABINet
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        # load full model--> Vision Language Align
        if config.full_ckpt is not None:
            logging.info(f'Student (ABINet-full) model reads pretrained param from {config.full_ckpt}.')
            ABINet_state_dict = torch.load(config.full_ckpt)['model']
            ABINet_LR_state_dict = self.state_dict()
            pretrained_state_dict = {k: v for k, v in ABINet_state_dict.items() if k in ABINet_LR_state_dict}
            ABINet_LR_state_dict.update(pretrained_state_dict)
            self.load_state_dict(ABINet_LR_state_dict)

    def logits_to_string(self, output):
        # {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,'attn_scores': attn_scores, 'name': 'vision',' backbone_feature': features}
        logit = output['logits']  # .cpu()
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, self.charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    def forward(self, images,common_info=None):
        v_res = self.vision(images)
        # new here!
        v_tokens = torch.softmax(v_res['logits'], dim=-1)
        v_lengths = v_res['pt_lengths'].clamp_(2, self.charset.max_length)
        l_res = self.language(v_tokens, v_lengths)
        l_feature, v_feature = l_res['feature'], v_res['sematic_feat']
        a_res = self.alignment(l_feature, v_feature)

        v_res['logits'] = a_res['logits']

        return v_res
