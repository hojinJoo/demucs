from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn
import numpy as np



class SoftPositionEmbed(nn.Module):
    """Builds the soft position embedding layer with learnable projection.

    Args:
        hid_dim (int): Size of input feature dimension.
        resolution (tuple): Tuple of integers specifying width and height of grid.
    """

    def __init__(
        self,
        hid_dim: int = 64,
        resolution: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        self.grid = self.build_grid(resolution)

    def forward(self, inputs):
        self.grid = self.grid.to(inputs.device)
        grid = self.embedding(self.grid).to(inputs.device)
        return inputs + grid

    def build_grid(self, resolution):
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

class SlotAttention(nn.Module):
    """Slot Attention module.

    Args:
        num_slots: int - Number of slots in Slot Attention.
        iterations: int - Number of iterations in Slot Attention.
        num_attn_heads: int - Number of multi-head attention in Slot Attention,
    """

    def __init__(
        self,
        num_slots: int = 4,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        slot_dim: int = 768,
        hid_dim: int = 768,
        mlp_hid_dim: int = 1536,
        eps: float = 1e-8,
        ctr = False
    ):
        super().__init__()

        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_attn_heads = num_attn_heads
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.eps = eps

        self.scale = (num_slots // num_attn_heads) ** -0.5


        self.slots = nn.Parameter(torch.rand(1, 1, self.slot_dim))


        self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_slot = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hid_dim, self.slot_dim),
        )

    def forward(self, inputs, num_slots=None, train=False,ctr=False):
        outputs = dict()

        B, N_in, D_in = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_attn_heads

        
        slots = self.slots.expand(B, K, -1)
        inputs = self.norm_input(inputs)

        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        # k, v: (B, N_heads, N_in, D_slot // N_heads).

        if not train:
            attns = list()

        for iter_idx in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slot(slots)

            q = self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
            # q: (B, N_heads, K, slot_D // N_heads)

            attn_logits = torch.einsum("bhid, bhjd->bhij", k, q) * self.scale

            attn = attn_logits.softmax(dim=-1) + self.eps  # Normalization over slots
            # attn: (B, N_heads, N_in, K)

            if not train:
                attns.append(attn)

            attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Weighted mean
            # attn: (B, N_heads, N_in, K)

            updates = torch.einsum("bhij,bhid->bhjd", attn, v)
            # updates: (B, N_heads, K, slot_D // N_heads)
            updates = updates.transpose(1, 2).reshape(B, K, -1)
            # updates: (B, K, slot_D)

            slots = self.gru(updates.reshape(-1, D_slot), slots_prev.reshape(-1, D_slot))

            slots = slots.reshape(B, -1, D_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))

        outputs["slots"] = slots
        outputs["attn"] = attn
        if not train:
            outputs["attns"] = torch.stack(attns, dim=1)
            # attns: (B, T, N_heads, N_in, K)

        return outputs    


class Decoder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        slot_dim: int = 768,
        dec_hid_dim: int = 64,
        dec_init_size_f: int = 8,
        dec_init_size_t: int = 2,
        dec_depth: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.dec_init_size_f = dec_init_size_f
        self.dec_init_size_t = dec_init_size_t
        self.decoder_pos = SoftPositionEmbed(slot_dim, (dec_init_size_f, dec_init_size_t))

        D_slot = slot_dim
        D_hid = dec_hid_dim
        upsample_step = int(np.log2(img_size // dec_init_size_t))

        deconvs = nn.ModuleList()
        count_layer = 0
        for _ in range(upsample_step):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot,
                        D_hid,
                        5,
                        stride=(2, 2),
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        for _ in range(dec_depth - upsample_step - 1):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot, D_hid, 5, stride=(1, 1), padding=2
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(D_hid, 4, 3, stride=(1, 1), padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x,ft):
        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        # print(f"x size : {x.size()}")
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.dec_init_size_f, self.dec_init_size_t, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconvs(x)
        # print(f"after deconv ln 207 : {x.size()}")
        x = x[:, :, : ft[0], : ft[1]]
        # print(f"ft : {ft}")
        x = x.permute(0, 2, 3, 1)
        # print(f"final x size : {x.size()}")
        return x



class SlotDecoder(nn.Module) : 
    def __init__(
        self,
        num_slots: int = 4,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        slot_dim: int = 768,
        hid_dim: int = 768,
        mlp_hid_dim: int = 768,
        eps: float = 1e-8,
        t_size: int = 256,
        dec_hid_dim: int = 64,
        dec_init_size_f: int = 16,
        dec_init_size_t: int = 2,
        dec_depth: int = 6,
        ctr = False
    ):
        super().__init__()
        self.num_slots = num_slots
        self.ctr= ctr
        self.slot_attention = SlotAttention(num_slots,num_iterations,num_attn_heads,slot_dim,hid_dim,mlp_hid_dim,eps,ctr=ctr)
        self.decoder = Decoder(t_size,slot_dim,dec_hid_dim,dec_init_size_f,dec_init_size_t,dec_depth)
    def forward(self,inputs,ft,num_slots=None,train=True) :
        # ft는 fianl output size
        B,C,T = inputs.shape
        inputs = inputs.permute(0,2,1)
        out = self.slot_attention(inputs,num_slots,train,ctr=self.ctr)
        if train and self.ctr :
            slots = out['slots']
        # B,n_slots,slot_dim = out['slots'].shape
        out = self.decoder(out['slots'],ft)
        # print(f"ft : {ft}")
        out = out.reshape(B,self.num_slots,ft[0],ft[1],4).permute(0,1,4,2,3)
        if train and self.ctr : 
            return out, slots
        return out
        
if __name__ == "__main__" :
    a = torch.rand(3,768,15)
    slot = SlotDecoder()
    out = slot(a)
    
    print(out.shape)