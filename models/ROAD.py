import pickle
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
from .model import BaseModel

class Modality_Confidence(nn.Module):
    def __init__(self, s_dim, i_dim, t_dim, out_dim):
        super(Modality_Confidence, self).__init__()
        
        self.gm_s = nn.Sequential(
            nn.Linear(s_dim,1),
            # nn.Sigmoid()
        )
        
        self.gm_t = nn.Sequential(
            nn.Linear(t_dim,1),
            # nn.Sigmoid()
        )
        
        self.gm_i = nn.Sequential(
            nn.Linear(i_dim,1),
            # nn.Sigmoid()
        )
        
        self.gm_mm = nn.Sequential(
            nn.Linear(out_dim,1),
            # nn.Sigmoid()
        )
        
    def forward(self, s_emb, i_emb, t_emb, mm_emb):             
        gm_s = self.gm_s(s_emb.detach())
        gm_i = self.gm_i(i_emb.detach())
        gm_t = self.gm_t(t_emb.detach())
        gm_mm = self.gm_mm(mm_emb.detach())
        return [gm_s, gm_i, gm_t, gm_mm]

class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(InformationBottleneck, self).__init__()
        self.latent_dim = latent_dim
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 2*latent_dim),
            nn.ReLU()
        ) 
        self.mu_fc = nn.Linear(2*latent_dim, latent_dim)
        self.logvar_fc = nn.Linear(2*latent_dim, latent_dim)
        self.relu = nn.ReLU()

    def kld_gauss(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-20, max=20)  # 防止exp爆炸
        sigma = torch.exp(0.5 * logvar)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - sigma.pow(2), dim=1)/self.latent_dim
        return kl.mean()
    
    def forward(self, x,training,is_structure):
        h = self.shared_fc(x)
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        logvar = torch.clamp(logvar, min=-15, max=15)  
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)
        z = mu + eps * std * 0.01
        z = self.relu(z) + x
        kl_loss = self.kld_gauss(mu, logvar)
        return z, mu, std, kl_loss,logvar

class ModalFusionDCKI(nn.Module):
    def __init__(self, in_dim, out_dim, img_dim, txt_dim,temperature=0.1):
        super(ModalFusionDCKI, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        self.modal1 = nn.Linear(in_dim, out_dim)
        self.modal2 = nn.Linear(img_dim, out_dim)
        self.modal3 = nn.Linear(txt_dim, out_dim)
        
        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        self.temperature = temperature
    
    def contrastive_loss(self, anchor, positive, heads, temperature):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        logits = torch.matmul(anchor, positive.T) / temperature  # [B, B]
        B = logits.size(0)
        device = logits.device
        heads_i = heads.view(-1, 1).expand(B, B)  # [B, B]
        heads_j = heads.view(1, -1).expand(B, B)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        different_heads = (heads_i != heads_j)
        mask = different_heads | eye  
        logits = logits.masked_fill(~mask,0.0)
        labels = torch.arange(B).to(device)
        loss1 = F.cross_entropy(logits, labels)
        return loss1
        
    def forward(self, modal1_emb, modal2_emb, modal3_emb, rel_emb,head):

        modal1_emb = self.modal1(modal1_emb)
        modal2_emb = self.modal2(modal2_emb) 
        modal3_emb = self.modal3(modal3_emb)

        z_modal1 = torch.tanh(modal1_emb) 
        z_modal2 = torch.tanh(modal2_emb) 
        z_modal3 = torch.tanh(modal3_emb)
        z_stack = torch.stack((z_modal1, z_modal2, z_modal3), dim=1)     
        attention_scores = self.ent_attn(z_stack).squeeze(-1)  
        rel_xi = torch.sigmoid(rel_emb)
        attention_weights = torch.softmax(attention_scores / rel_xi, dim=-1)
        fused = torch.sum(attention_weights.unsqueeze(-1) * z_stack, dim=1)
        
        cl_s = self.contrastive_loss(fused, z_modal1, head,self.temperature)
        cl_i = self.contrastive_loss(fused, z_modal2, head,self.temperature)
        cl_t = self.contrastive_loss(fused, z_modal3, head,self.temperature)
        cl_f = self.contrastive_loss(fused, fused,head,self.temperature)
        contrastive_loss = (cl_s + cl_i + cl_t + cl_f) / 4
        
        return fused, attention_weights, contrastive_loss


    
class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        # x_mm = torch.relu(x_mm)
        return x_mm, attention_weights


    def relation_gated_fuse(self, modal1_emb, modal2_emb, modal3_emb, rel):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores / rel, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.mean(1).view(batch_size, self.out_dim)
        x_mm = torch.relu(x_mm)
        return x_mm
    
    def gated_fusion(self, emb, rel):
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel



class ROAD(BaseModel):
    def __init__(self, args):
        super(ROAD, self).__init__(args)
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id), 
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        if args.dataset == "DB15K":
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            # multi-modal information for MKG
            img = args.img.to(self.device).view(args.img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
            txt = txt.view(txt.size(0), -1)
        elif "TIVA" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 32, 64))
            img = img.view(img.size(0), -1)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
        elif "Kuai" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "WN9" in args.dataset:
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            img = torch.tensor(img).to(torch.float32)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
            txt = torch.tensor(txt).to(torch.float32)
        elif "FB15K-237" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=True)
        self.img_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)
        self.att_relation_fc = nn.Linear(args.r_dim, 1, bias=False)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=True)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)
        
        self.kl_loss_s = None 
        self.kl_loss_i = None
        self.kl_loss_t = None
        self.cl_loss = None
        
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.ib_dim =200        
        self.fuse_out_dim = 200
        self.img_dim_ib = self.img_dim 
        self.txt_dim_ib = self.txt_dim  
        
        self.structure_ib = InformationBottleneck(self.dim, self.dim)
        self.visual_ib = InformationBottleneck(self.img_dim, self.img_dim)
        self.text_ib = InformationBottleneck(self.txt_dim, self.txt_dim)
        self.mm_ib = InformationBottleneck(self.fuse_out_dim, self.fuse_out_dim)
        self.TuckER_S = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.img_dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.txt_dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(self.fuse_out_dim, self.fuse_out_dim)
        self.fuse_e = ModalFusionLayer(
            in_dim=self.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.img_dim,
            txt_dim=self.txt_dim
        )
        self.fuse_dcki = ModalFusionDCKI(
            in_dim=self.dim,
            out_dim=self.fuse_out_dim,
            img_dim=self.img_dim,
            txt_dim=self.txt_dim
        )
        
        self.fuse_r = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim
        )
        
        self.modal_conf = Modality_Confidence(
            s_dim=self.dim,
            i_dim=self.img_dim,
            t_dim=self.txt_dim,
            out_dim=self.fuse_out_dim
        )
        self.bceloss = nn.BCELoss()
        self.dropout = nn.Dropout(p=0.2)
        
        self.Linear_s = nn.Linear(self.ib_dim, self.dim)
        self.Linear_i = nn.Linear(self.img_dim, self.img_dim)
        self.Linear_t = nn.Linear(self.txt_dim, self.txt_dim)

        
        
    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)
        training = self.training
    
        e_embed = self.Linear_s(self.entity_embeddings(head))
        e_embed_ib, mu_s, std_s, kl_loss_s,logvar_s = self.structure_ib(e_embed,training,1)
        r_embed = self.relation_embeddings(relation)

        e_img_embed = self.Linear_i(self.img_entity_embeddings(head))
        e_img_embed = self.dropout(e_img_embed)
        e_img_embed_ib, mu_i, std_i, kl_loss_i ,logvar_i= self.visual_ib(e_img_embed,training,0)
        r_img_embed = self.img_relation_embeddings(relation)

        e_txt_embed = self.Linear_t(self.txt_entity_embeddings(head))    
        e_txt_embed = self.dropout(e_txt_embed)
        e_txt_embed_ib, mu_t, std_t, kl_loss_t ,logvar_t = self.text_ib(e_txt_embed,training,0)
        r_txt_embed = self.txt_relation_embeddings(relation)

        
        e_mm_embed, attn_f,cl_loss = self.fuse_dcki(
            e_embed_ib, e_img_embed_ib, e_txt_embed_ib, 
            rel_gate,head
        )

        self.cl_loss = cl_loss
        r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)
        
        conf_score = self.modal_conf(
                    e_embed_ib, 
                    e_img_embed_ib, 
                    e_txt_embed_ib, 
                    e_mm_embed
                ) 

        self.kl_loss_s = kl_loss_s
        self.kl_loss_i = kl_loss_i
        self.kl_loss_t = kl_loss_t

        p_s = self.TuckER_S(e_embed_ib, r_embed)
        p_i = self.TuckER_I(e_img_embed_ib, r_img_embed)
        p_d = self.TuckER_D(e_txt_embed_ib, r_txt_embed)
        p_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        
        all_s = self.Linear_s(self.entity_embeddings.weight)
        all_v = self.Linear_i(self.img_entity_embeddings.weight)
        all_t = self.Linear_t(self.txt_entity_embeddings.weight)
        all_f, _ = self.fuse_e(all_s, all_v, all_t)


        
        logits_s = torch.mm(p_s, all_s.transpose(1, 0))
        logits_i = torch.mm(p_i, all_v.transpose(1, 0))
        logits_t = torch.mm(p_d, all_t.transpose(1, 0))
        logits_mm = torch.mm(p_mm, all_f.transpose(1, 0))

        pred_s = F.softmax(logits_s, dim=1)
        pred_i = F.softmax(logits_i, dim=1)
        pred_d = F.softmax(logits_t, dim=1)
        pred_mm = F.softmax(logits_mm, dim=1)
        
        if not self.training:
            return [pred_s, pred_i, pred_d,pred_mm], [_, _, _,_],conf_score
        else:
            return [pred_s, pred_i, pred_d,pred_mm], [e_embed_ib, e_img_embed_ib, e_txt_embed_ib,e_mm_embed], conf_score
           
            
    
    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        _, disen_str, _ = self.structure_moe(self.entity_embeddings(head))
        _, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head))
        _, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head))
        return [disen_str, disen_img, disen_txt]


    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)     
        return loss_s, loss_i, loss_d, loss_mm,self.kl_loss_s,self.kl_loss_i, self.kl_loss_t,self.cl_loss

    
    

 
 
