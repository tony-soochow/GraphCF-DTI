import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

from drug_feature_H import *
from rdkit.Chem import AllChem
from rdkit import Chem
import os

def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)
\
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1) # 双线性层

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):   # (h_os, x2_os, x2_os_a)
        c_x = c.expand_as(h_pl)
        # print('c_x:{}'.format(c_x.size()))        c_x:torch.Size([9495, 128])
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.max(seq, 0).values
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)



class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim/2), bias=True)
        self.fc2 = nn.Linear(int(dim/2), 2, bias=False)
        self.dropout = nn.Dropout(0.2)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion in SeHGNN.
    '''
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels     torch.Size([726, 61, 512])
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]    torch.Size([726, 1, 61, 128])
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]    torch.Size([726, 1, 128, 61])
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]    torch.Size([726, 1, 61, 512])

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]  torch.Size([726, 1, 61, 61])
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]     torch.Size([726, 1, 61, 512])
        return o.permute(0,2,1,3).reshape((B, M, C)) + x    # [726, 61, 1, 512]->[726, 61, 512]

class GraphCF(nn.Module):
    def __init__(self, args):
        super(GraphCF, self).__init__()

        self.data_name = args.in_file
        self.in_dim = args.dimensions
        self.hidden1 = args.hidden1
        self.batch = args.batch
        
        self.encoder_o1 = GCNConv(self.in_dim, self.hidden1)
        self.encoder_o2 = GCNConv(self.hidden1 * 2, self.hidden1)
        self.encoder_s1 = GCNConv(self.in_dim, self.hidden1)
        self.encoder_s2 = GCNConv(self.hidden1 * 2, self.hidden1)
        
        # 对比学习
        self.disc = Discriminator(self.hidden1 * 2)   # 64
        self.classifier = linear_module(dim=128*2)
        self.dropout = args.dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        # 分子图卷积 两层
        self.char_dim = 64
        self.conv = 64
        self.drug_MAX_LENGTH = 100
        self.drug_kernel = [2, 4]
        self.drug_vocab_size = 65
        self.drug_embed = nn.Embedding(self.drug_vocab_size, self.char_dim, padding_idx=0)
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - self.drug_kernel[0] - self.drug_kernel[1] + 2
        
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.char_dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        
        
        # 蛋白质序列嵌入
        self.protein_MAX_LENGTH = 1000
        self.protein_kernel = [2, 4]
        self.protein_vocab_size = 26
        self.protein_embed = nn.Embedding(self.protein_vocab_size, self.in_dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.in_dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
        )
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - self.protein_kernel[0] - self.protein_kernel[1] + 2
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)
        
        # 特征融合
        self.out_dim = 16
        self.num_heads = args.num_heads
        attn_vec_dim = self.in_dim
        self.fc1 = nn.Linear(self.out_dim * self.num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        
        self.semantic_fusion = Transformer(self.in_dim, num_heads=args.num_heads, att_drop=args.dropout, act='relu')
        self.fc_after_concat = nn.Linear(3 * self.in_dim, self.in_dim)
        
        # 3D特征
        self.latent_size = 128
        self.pos_mask_prob = 0.15
        self.raw_with_pos = True
        self.mode = 'mask'
        self.use_random_conformer = True
        self.pos_embedding = PosEmbeddingwithMask(self.latent_size, self.pos_mask_prob, raw_with_pos=self.raw_with_pos)

        self.drug_H_dims = [self.latent_size, 64, self.latent_size]
        self.drug_H_gcn = nn.ModuleList([GCNConv(self.drug_H_dims[i], self.drug_H_dims[i+1]) for i in range(len(self.drug_H_dims)-1)])

        self.protein_H_dims = [41, self.latent_size, self.latent_size]
        self.protein_H_gcn = nn.ModuleList([GCNConv(self.protein_H_dims[i], self.protein_H_dims[i+1]) for i in range(len(self.protein_H_dims)-1)])
        
    def get_drug_HF(self, drug_ids, pos_dict, dis_dict, edge_dict):
        
        batch_feature = []
        for id in drug_ids:
            id = int(id)
            pos = pos_dict.item().get(id)
            
            if len(pos) == 0:
                feature = torch.zeros(self.latent_size).to('cuda')
                batch_feature.append(feature)
                continue
            
            pos = torch.tensor(pos, dtype=torch.float32)
            pos = pos.to('cuda')
            
            distance = dis_dict.item().get(id)
            distance = torch.tensor(distance, dtype=torch.float32)
            distance = distance.to('cuda')
            
            last_pos_pred = pos.new_zeros((pos.shape[0], 3)).uniform_(-1, 1)
            pos_mask_idx = self.pos_embedding.get_mask_idx(pos, mode=self.mode)

            drug_node_feature, drug_edge_feature = self.pos_embedding(pos, distance, last_pred=last_pos_pred, mask_idx=pos_mask_idx, mode=self.mode)
            
            edge_index = edge_dict.item().get(id)
            edge_index = torch.tensor(edge_index)
            edge_index = edge_index.to('cuda')
            
            for drug_gcn in self.drug_H_gcn:
                drug_node_feature = F.relu(drug_gcn(drug_node_feature, edge_index))
            drug_node_feature = torch.mean(drug_node_feature, dim=0)
            drug_edge_feature = torch.mean(drug_edge_feature, dim=0)
            global_feature = drug_node_feature + drug_edge_feature
            
            batch_feature.append(global_feature)

        return torch.stack(batch_feature)


    def get_protein_HF(self, protein_ids, protein_dict, edge_dict):
        
        if self.data_name == 'drugbank':
            error_list = [2162, 2444, 2589, 599, 1987, 2603, 3076, 3007, 2437, 85, 1153, 2356, 1986, 1791, 1396, 2100, 787, 1498]
        else:
            error_list = [374]  
        
        batch_feature = []
        for id in protein_ids:
            id = int(id)
            protein_node_feature = protein_dict.item().get(id)

            # zero_count = np.count_nonzero(protein_node_feature == 0)
            if len(protein_node_feature) == 0 or id in error_list :
                feature = torch.zeros(self.latent_size).to('cuda')
                batch_feature.append(feature)
                continue
            
            protein_node_feature = torch.from_numpy(protein_node_feature).to(torch.float32).to('cuda')
                
            edge_index = edge_dict.item().get(id)
            edge_index = torch.from_numpy(edge_index).to('cuda')
            
            for protein_gcn in self.protein_H_gcn:
                protein_node_feature = F.relu(protein_gcn(protein_node_feature, edge_index))
            protein_feature = torch.mean(protein_node_feature, dim=0)
            
            batch_feature.append(protein_feature)
        
        return torch.stack(batch_feature)
        
        
    def get_dti_seq_feature(self, drug, protein):
        
        # get drug feature
        drugembed = self.drug_embed(drug)       # 128 100  64
        drugembed = drugembed.permute(0, 2, 1)          #  128 64 100
        drugConv = self.Drug_CNNs(drugembed)                #  128  128  85
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)      # 128  128  
        
        proteinembed = self.protein_embed(protein)       # 128 1200 128 
        proteinembed = proteinembed.permute(0, 2, 1)            # 128  128  1200 
        proteinConv = self.Protein_CNNs(proteinembed)               # 128 128 1179
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        
        return drugConv, proteinConv

    
    def feature_fusion(self, metapath_outs):
        beta = []
        for metapath_out in metapath_outs:          # [2, 32, 128]
            fc1 = torch.tanh(self.fc1(metapath_out))    
            fc1_mean = torch.mean(fc1, dim=0)  # metapath specific vector
            fc2 = self.fc2(fc1_mean)  # metapath importance
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
    
    
    def forward(self, data_o, data_s, data_a, idx, drug_vec, target_vec, drug_H_data, protein_H_data):
        x_o, adj = data_o.x, data_o.edge_index
        adj2 = data_s.edge_index
        x_a = data_a.x

        x1_o = F.relu(self.encoder_o1(x_o, adj))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_s = F.relu(self.encoder_s1(x_o, adj2))
        x1_s = F.dropout(x1_s, self.dropout, training=self.training)

        x1_os = torch.cat((x1_o, x1_s), dim=1)

        x2_o = self.encoder_o2(x1_os, adj)
        x2_s = self.encoder_s2(x1_os, adj2)

        x2_os = torch.cat((x2_o, x2_s), dim=1)

        x1_o_a = F.relu(self.encoder_o1(x_a, adj))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x1_s_a = F.relu(self.encoder_s1(x_a, adj2))
        x1_s_a = F.dropout(x1_s_a, self.dropout, training=self.training)

        x1_os_a = torch.cat((x1_o_a, x1_s_a), dim=1)

        x2_o_a = self.encoder_o2(x1_os_a, adj)
        x2_s_a = self.encoder_s2(x1_os_a, adj2)

        x2_os_a = torch.cat((x2_o_a, x2_s_a), dim=1)

        # graph representation
        h_os = self.read(x2_os)
        h_os = self.sigm(h_os)

        h_os_a = self.read(x2_os_a)
        h_os_a = self.sigm(h_os_a)

        # adversarial learning
        ret_os = self.disc(h_os, x2_os, x2_os_a)
        ret_os_a = self.disc(h_os_a, x2_os_a, x2_os)

        # 2D feature
        target_to_feature = x2_os[idx[0]]   # 128
        drug_to_feature = x2_os[idx[1]]

        # 1D feature
        drug_st_feature, target_st_feature = self.get_dti_seq_feature(drug_vec, target_vec)

        # 3D feature
        drug_H_feature = self.get_drug_HF(idx[1], drug_H_data[0], drug_H_data[1], drug_H_data[2])
        protein_H_feature = self.get_protein_HF(idx[0], protein_H_data[0], protein_H_data[1])

        # Fused feature
        target_fu_feature = self.semantic_fusion(torch.stack([target_to_feature, target_st_feature, protein_H_feature], dim=1))
        target_fu_feature = self.fc_after_concat(target_fu_feature.reshape(self.batch, -1))  
        
        drug_fu_feature = self.semantic_fusion(torch.stack([drug_to_feature, drug_st_feature, drug_H_feature], dim=1))
        drug_fu_feature = self.fc_after_concat(drug_fu_feature.reshape(self.batch, -1)) 


        x = torch.cat([target_fu_feature, drug_fu_feature], dim=1)
        x_out = self.classifier(x)

        return F.softmax(x_out, dim=-1), ret_os, ret_os_a, x2_os