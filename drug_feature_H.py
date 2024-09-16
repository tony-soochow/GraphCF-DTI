import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
        pos_drop=0.0
    ):
        super().__init__()
        self.drop = nn.Dropout(p=pos_drop)
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
            x = self.drop(x)
        return x


class PosEmbeddingwithMask(nn.Module):
    def __init__(self, latent_size, mask_prob, raw_with_pos=True):
        super().__init__()
        self.latent_size = latent_size
        self.pos_embedding = MLP(3, [latent_size, latent_size], pos_drop=0.0)
        # self.pos_embedding_schnet = SchNet(hidden_channels=latent_size, out_channels=latent_size)
        self.dis_embedding = MLP(1, [latent_size, latent_size], pos_drop=0.0)
        self.mask_prob = mask_prob
        self.raw_with_pos = raw_with_pos

    def get_mask_idx(self, pos, mode="mask"):
        if mode == "mask":
            random_variables = pos.new_zeros((pos.shape[0], 1)).uniform_()
            mask_idx = random_variables < self.mask_prob
            return mask_idx
        else:
            return None

    def forward(self, pos, distance, last_pred=None, mask_idx=None, mode="mask", z=None, batch=None):
        if mode == "mask":
            pos = self.mask(pos, last_pred, mask_idx)
        elif mode == "mol2conf":
            pos = self.mol2conf(pos, last_pred, mask_idx)
        elif mode == "conf2mol":
            pos = self.conf2mol(pos, last_pred, mask_idx)
        elif mode == "raw":
            pos = self.raw(pos, last_pred, mask_idx)

        extended_x = self.pos_embedding(pos)

        extended_edge_attr = self.dis_embedding(distance)
        return extended_x, extended_edge_attr

    def mask(self, pos, last_pred, mask_idx):
        pos = torch.where(mask_idx, last_pred, pos)
        return pos

    def mol2conf(self, pos, last_pred, mask_idx):
        return last_pred

    def conf2mol(self, pos, last_pred, mask_idx):
        return pos

    def raw(self, pos, last_pred: torch.Tensor, mask_idx):
        if self.raw_with_pos:
            return pos
        else:
            return last_pred

