import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.ProjectionFunc import BidirectionalLSTM
from loss.losses import MultiCELosses
from loss.beam_search import seq_modeling



class Visual_Attention_Loss(nn.Module):
    def __init__(self):
        super(Visual_Attention_Loss, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def _get_padding_mask(self, length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    def forward(self,stu,tea,gt_dict,layer_num=3):
        stu_feat = stu['visual_feat']['layer' + str(layer_num)]  # N,C,H,W
        tea_feat = tea['visual_feat']['layer' + str(layer_num)].detach()
        N, C, H, W = stu_feat.shape

        tea_attn = tea['attn_scores'].detach()
        pt_lengths = gt_dict['length'].squeeze()
        pos_mask = (~self._get_padding_mask(pt_lengths, 26))[:, :, None, None].float()
        soft_attn_mask = (tea_attn * pos_mask).max(dim=1, keepdim=True)[0].detach()  # [N,1,8,32] soft_mask

        stu_feat = stu_feat.view(N,C,-1)
        tea_feat=tea_feat.view(N,C,-1)

        stu_mean = torch.mean(stu_feat,dim=-1,keepdim=True)
        tea_mean = torch.mean(tea_feat,dim=-1,keepdim=True)

        stu_std = torch.std(stu_feat,dim=-1,keepdim=True)
        tea_std = torch.std(tea_feat,dim=-1,keepdim=True)

        diff_stu_feat = (stu_feat-stu_mean)/stu_std
        diff_tea_feat = (tea_feat-tea_mean)/tea_std

        diff_stu_feat = diff_stu_feat.view(N,C,H,W)
        diff_tea_feat = diff_tea_feat.view(N,C,H,W)

        diff_stu_feat = (soft_attn_mask*diff_stu_feat).view(N,C,-1)
        diff_tea_feat = (soft_attn_mask*diff_tea_feat).view(N,C,-1)
        loss = 1 - torch.cosine_similarity(diff_stu_feat,diff_tea_feat,dim=-1)
        return loss.mean()



# ==============================================
class Sematic_Loss(nn.Module):
    def __init__(self,proj_dim=512):
        super(Sematic_Loss, self).__init__()
        self.proj = BidirectionalLSTM(input_size=proj_dim,hidden_size=256,output_size=512,proj_img=False)
        self.temp = 0.1
        self.l1=nn.L1Loss()
        self.l2=nn.MSELoss()

    def vec_contrastive_loss(self,anchor_embed, pos_embed, n_embed_per_batch,gt_length):
        """
        :param anchor_embed: N*L_i,C
        :param pos_embed: N*L_i,C
        """
        instances = torch.cat((anchor_embed, pos_embed), dim=0)
        normalized_instances = F.normalize(instances, dim=1)
        similarity_matrix = normalized_instances @ normalized_instances.T
        similarity_matrix_exp = (similarity_matrix / self.temp).exp_()
        cross_entropy_denominator = similarity_matrix_exp.sum(
            dim=1) - similarity_matrix_exp.diag()
        cross_entropy_nominator = torch.cat((
            similarity_matrix_exp.diagonal(offset=n_embed_per_batch)[:n_embed_per_batch],
            similarity_matrix_exp.diagonal(offset=-n_embed_per_batch)
        ), dim=0)
        cross_entropy_similarity = cross_entropy_nominator / cross_entropy_denominator
        loss = - cross_entropy_similarity.log()
        loss = loss.mean()
        return loss

    def forward(self,stu_vec,tea_vec,gt_dict):
        """
        :param stu_vec: N,T,C
        :param tea_vec: N,T,C
        :param gt_dict:
        :return:
        """
        gt_length = gt_dict['length']
        stu_vec=self.proj(stu_vec)
        tea_vec=self.proj(tea_vec)
        stu_vec = torch.cat([v[:l] for v, l in zip(stu_vec, gt_length)])
        tea_vec = torch.cat([v[:l] for v, l in zip(tea_vec,gt_length)])
        # stu_vec = stu_vec.flatten(0,1)
        # tea_vec=tea_vec.flatten(0,1).detach()
        loss = self.vec_contrastive_loss(stu_vec,tea_vec,stu_vec.shape[0],gt_length)
        return loss





class SoftLogits_Loss(nn.Module):
    def __init__(self, soft_teacher=True, KD_tmp=1,seq_modeling=True):
        super(SoftLogits_Loss, self).__init__()
        self.soft_teacher = soft_teacher
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.KD_tmp = KD_tmp
        self.seq_modeling = seq_modeling
        self.alpha = 0.2
        self.path_thred = 0.1

    def _get_padding_mask(self, length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    def forward(self,stu_logits,tea_logits,gt_dict):
        stu_logits = F.log_softmax(stu_logits/self.KD_tmp,dim=-1)
        stu_logits=torch.clamp_min(stu_logits,-10)
        tea_logits = (tea_logits.detach() / self.KD_tmp).softmax(dim=-1)
        if self.seq_modeling:
            seq_logits, seq_len, alpha_weight = seq_modeling(tea_logits,alpha=self.alpha,path_thred=self.path_thred)
            tea_logits = alpha_weight*seq_logits + (1-alpha_weight)*tea_logits
            kl_loss = 0.
            for idx in range(seq_len.shape[0]):
                kl_loss += self.KL(stu_logits[idx][:seq_len[idx]],tea_logits[idx][:seq_len[idx]])
            kl_loss = kl_loss/seq_len.shape[0] # batch-mean
        else:
            kl_loss = self.KL(stu_logits,tea_logits)

        return kl_loss



class CrossEentropyLoss(nn.Module):
    def __init__(self):
        super(CrossEentropyLoss, self).__init__()
        self.CE = MultiCELosses()

    def forward(self,stu_logits,tea_logits,gt_dict):
        ce_loss = self.CE(stu_logits, gt_dict)
        return ce_loss


class Distill_Loss(nn.Module):
    def __init__(self,config):
        super(Distill_Loss, self).__init__()
        self.l2=nn.MSELoss()
        self.visual_loss=Visual_Attention_Loss()
        self.sematic_loss=Sematic_Loss()
        self.cross_entropy = CrossEentropyLoss()
        self.soft_tea_loss=SoftLogits_Loss(soft_teacher=True,
                                           KD_tmp=4,
                                           seq_modeling=True)

        self.loss_wight = {'visual': 4.,'sematic': 2,'ce':0.025,'kl':20}

    def forward(self,stu_outpout,tea_output,gt_dict):
        # ==== CLASSIC
        visual_loss= self.visual_loss(stu_outpout,tea_output,gt_dict,3) # select the 3-rd feat for distillation
        sematic_loss=self.sematic_loss(stu_outpout['sematic_feat'],tea_output['sematic_feat'],gt_dict)
        ce_loss=self.cross_entropy(stu_outpout['logits'],tea_output['logits'],gt_dict)
        kl_loss = self.soft_tea_loss(stu_outpout['logits'],tea_output['logits'],gt_dict)

        loss = self.loss_wight['visual']*visual_loss+\
               self.loss_wight['sematic']*sematic_loss+\
               self.loss_wight['ce']*ce_loss+\
               self.loss_wight['kl']*kl_loss


        return loss, visual_loss, sematic_loss, ce_loss,kl_loss



class PARSeq_distill_loss(nn.Module):
    def __init__(self,config):
        # Since PARSeq use the permuted auto-regressive technique,
        # we thus give an separate implementation for this.
        super(PARSeq_distill_loss, self).__init__()
        self.l2 = nn.MSELoss()
        self.sematic_loss = Sematic_Loss(proj_dim=384)
        self.cross_entropy = CrossEentropyLoss()
        self.soft_tea_loss=SoftLogits_Loss(soft_teacher=True,
                                           KD_tmp=4,
                                           seq_modeling=True)

        self.loss_wight = {'visual': 4.,'sematic': 2,'ce':0.025,'kl':20}


    def forward(self,stu_outpout,tea_output,gt_dict):
        stu_visual,stu_semantic,stu_logits = stu_outpout['visual_feat'],stu_outpout['semantic_feat'],stu_outpout['logits']
        tea_visual,tea_semantic,tea_logits = tea_output['visual_feat'],tea_output['semantic_feat'],tea_output['logits']

        # ===1. viusal loss
        N,T,C = stu_visual.shape
        stu_mean = torch.mean(stu_visual, dim=1, keepdim=True)
        tea_mean = torch.mean(tea_visual, dim=1, keepdim=True)
        stu_std = torch.std(stu_visual, dim=1, keepdim=True)
        tea_std = torch.std(tea_visual, dim=1, keepdim=True)
        diff_stu_feat = (stu_visual - stu_mean)/stu_std
        diff_tea_feat = (tea_visual - tea_mean)/tea_std
        visual_loss = 1 - torch.cosine_similarity(diff_stu_feat, diff_tea_feat, dim=1)
        visual_loss = visual_loss.mean()
        #== 2. semantic loss + 3. logits_loss
        # since there are K! permutation results, we calcaculate the mean of them.
        sematic_loss=0.
        ce_loss=0.
        kl_loss = 0.
        for permu_id in range(len(stu_semantic)):
            stu_semantic_one,tea_semantic_one = stu_semantic[permu_id],tea_semantic[permu_id]
            stu_logits_one,tea_logits_one = stu_logits[permu_id],tea_logits[permu_id]
            sematic_loss += self.sematic_loss(stu_semantic_one,tea_semantic_one,gt_dict)
            ce_loss += self.cross_entropy(stu_logits_one, tea_logits_one, gt_dict)
            kl_loss += self.soft_tea_loss(stu_logits_one, tea_logits_one, gt_dict)

        sematic_loss = sematic_loss/len(stu_semantic)
        ce_loss = ce_loss/len(stu_semantic)
        kl_loss = kl_loss/len(stu_semantic)

        # === 4. sumition over all losses
        loss = self.loss_wight['visual']*visual_loss+\
               self.loss_wight['sematic']*sematic_loss+\
               self.loss_wight['ce']*ce_loss+\
               self.loss_wight['kl']*kl_loss

        return loss, visual_loss, sematic_loss, ce_loss,kl_loss
