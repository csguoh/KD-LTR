import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.helpers import named_apply
from model.parseq.modules import DecoderLayer, Decoder, Encoder, TokenEmbedding
import logging
from model.parseq.parseq_tokenizer import get_parseq_tokenize
from torchvision import transforms


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class PARSeq(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.tokenizer = get_parseq_tokenize()

        img_size= config.img_size
        patch_size= config.patch_size
        embed_dim=config.embed_dim
        enc_depth=config.enc_depth
        enc_num_heads=config.enc_num_heads
        enc_mlp_ratio=config.enc_mlp_ratio

        self.max_label_length = 25
        self.decode_ar = True
        self.refine_iters = 1
        self.bos_id = 95
        self.eos_id = 0
        self.pad_id = 96

        dec_num_heads = config.dec_num_heads
        dec_mlp_ratio=config.dec_mlp_ratio
        dropout = config.dropout
        dec_depth = config.dec_depth
        perm_num = config.perm_num
        perm_mirrored = config.perm_mirrored
        max_label_length = config.max_label_length


        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)#

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)


        if config.full_ckpt is not None:
            logging.info(f'Read full ckpt parseq model from {config.full_ckpt}.')
            state = torch.load(config.full_ckpt)
            self.load_state_dict(state, strict=True)



    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self,images, common_info=None,input_lr=False):
        self._device = images.device
        trans = transforms.Normalize(0.5, 0.5)
        images = trans(images)
        is_train = True if common_info is not None else False
        if is_train:
            return self.training_step(images,parseq_info=common_info)
        else:
            return self.test_step(images,input_lr)



    def test_step(self,images,input_lr):
        if input_lr:
            images = F.interpolate(images, scale_factor=2, mode='bicubic', align_corners=True)

        max_length = None
        testing = not self.training
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1
        memory = self.encode(images)#2,384,128

        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)#N,8(max_len),95
        else:
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return {'logits':logits}

    def generate_attn_masks(self, perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, images, parseq_info):
        output_dict = {'semantic_feat': [], 'logits': [], 'visual_feat': None}
        memory = self.encode(images)
        output_dict['visual_feat'] = memory
        tgt = parseq_info['target']
        tgt_perms = parseq_info['tgt_perms'][0]
        tgt_in = tgt[:, :-1]
        assert tgt_in.shape[1] <= 26, 'meet too long instance!>26'
        assert tgt_perms.shape[1] == tgt.shape[1]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            output_dict['semantic_feat'].append(out)
            logits = self.head(out)  # [N,T, C]
            output_dict['logits'].append(logits)

        return output_dict
