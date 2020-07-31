from torch import nn
import torch
import math
from PIL import Image
import numpy as np
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_classes, slots_per_class, dim, iters=3, eps=1e-8, vis=False, vis_id=0, loss_status=1):
        super().__init__()
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        self.num_slots = num_classes * slots_per_class
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.loss_status = loss_status

        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        # self.to_q = nn.Linear(dim, dim, bias=False)
        # self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(dim, dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(dim, dim),
        )
        self.gru = nn.GRU(dim, dim)

        # self.channel_weights = nn.Parameter(torch.ones(1, 1, dim))

        self.vis = vis
        self.vis_id = vis_id

    def forward(self, inputs, inputs_x):
        b, n, d = inputs.shape
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs), inputs

        # channel_weights = torch.relu(self.channel_weights.expand(b, self.num_slots, -1))z

        for _ in range(self.iters):
            slots_prev = slots

            # q = self.to_q(slots)
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2,0,1])).permute([1,2,0])) * dots.sum(2).sum(1).expand_as(dots.permute([1,2,0])).permute([2,0,1])# * 10
            attn = torch.sigmoid(dots)
            updates = torch.einsum('bjd,bij->bid', inputs_x, attn)
            updates = updates / inputs_x.size(2)
            self.gru.flatten_parameters()
            slots, _ = self.gru(
                updates.reshape(1, -1, d),
                slots_prev.reshape(1, -1, d)
            )

            slots = slots.reshape(b, -1, d)

            if self.vis:
                slots_vis = attn.clone()
                # print(slots_vis.shape)

        if self.vis:
            # print("hehe")
            # print(slots_vis.max())
            # slots_vis = torch.relu(slots_vis)
            if self.slots_per_class > 1:
                new_slots_vis = torch.zeros((slots_vis.size(0), self.num_classes, slots_vis.size(-1)))
                for slot_class in range(self.num_classes):
                    new_slots_vis[:, slot_class] = torch.sum(torch.cat([slots_vis[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)]], dim=1), dim=1, keepdim=False)
                slots_vis = new_slots_vis.to(updates.device)

            slots_vis = slots_vis[self.vis_id]
            slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max()-slots_vis.min()) * 255.).reshape(slots_vis.shape[:1]+(int(slots_vis.size(1)**0.5), int(slots_vis.size(1)**0.5)))
            slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
            for id, image in enumerate(slots_vis):
                image = Image.fromarray(image, mode='L')
                image.save(f'sloter/vis/slot_{id:d}.png')
            # print(slots_vis.shape)
            print(self.loss_status*torch.sum(attn.clone(), dim=2, keepdim=False))
            print(self.loss_status*torch.sum(updates.clone(), dim=2, keepdim=False))

        if self.slots_per_class > 1:
            new_updates = torch.zeros((updates.size(0), self.num_classes, updates.size(-1)))
            for slot_class in range(self.num_classes):
                new_updates[:, slot_class] = torch.sum(updates[:, self.slots_per_class*slot_class: self.slots_per_class*(slot_class+1)], dim=1, keepdim=False)
            updates = new_updates.to(updates.device)

        attn_relu = torch.relu(attn)
        slot_loss = torch.sum(attn_relu) / attn.size(0) / attn.size(1) / attn.size(2)# * self.slots_per_class

        # updates_weighted_sum = torch.einsum('bid,bid->bi', updates, channel_weights)
        # return self.loss_status*updates_weighted_sum, slot_loss
        return self.loss_status*torch.sum(updates, dim=2, keepdim=False), slot_loss


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionEmbeddingSine, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 1).float() * (-math.log(10000.0) / d_model))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe = torch.sin(position)
        # pe = torch.sin(position * div_term)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        pe = pe.transpose(0, 1)
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # pe = pe.squeeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(2), :x.size(3)]#.expand(x.shape[-2:])
        # pe = pe.transpose(0,1)
        x = x + pe
        return self.dropout(x)