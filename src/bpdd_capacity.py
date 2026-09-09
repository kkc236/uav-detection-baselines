"""Prediction-conditioned local visual expert for FDR distributions."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def boundary_grid(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Forty boundary-neighborhood points and one center, in grid_sample coordinates."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError('boxes must have shape B,Q,4')
    boxes = boxes.detach().float()
    t = torch.linspace(-.5, .5, 5, device=boxes.device)
    points = []
    for radius in (.4, .6):
        points.extend((torch.stack((torch.full_like(t,-radius),t),-1),
                       torch.stack((torch.full_like(t,radius),t),-1),
                       torch.stack((t,torch.full_like(t,-radius)),-1),
                       torch.stack((t,torch.full_like(t,radius)),-1)))
    offsets = torch.cat([*points, torch.zeros(1,2,device=boxes.device)])
    positions = boxes[...,None,:2] + offsets * boxes[...,None,2:]
    valid = ((positions >= 0) & (positions <= 1)).all(-1) & torch.isfinite(positions).all(-1)
    # Invalid samples are masked separately; never feed NaNs into grid_sample.
    grid = torch.nan_to_num(positions * 2 - 1, nan=3., posinf=3., neginf=-3.)
    return grid, valid


def sample_local_features(feature: Tensor, grid: Tensor) -> Tensor:
    """B,C,H,W x B,Q,K,2 -> B,Q,K,C; FP32 sampling has CUDA/CPU parity."""
    with torch.autocast(device_type=feature.device.type, enabled=False):
        sampled = F.grid_sample(feature.float(), grid.float(), mode='bilinear',
                                padding_mode='zeros', align_corners=False)
    return sampled.permute(0,2,3,1).to(feature.dtype)


class LocalAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ff: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden)
        self.kv_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, ff), nn.GELU(), nn.Linear(ff, hidden))

    def forward(self, query: Tensor, tokens: Tensor, valid: Tensor) -> Tensor:
        # A query token always remains available, including fully off-image boxes.
        tokens = torch.cat((tokens, query[:,None]),1)
        valid = torch.cat((valid, torch.ones_like(valid[:,:1])),1)
        q = self.q_norm(query)[:,None]
        kv = self.kv_norm(tokens)
        update = self.attn(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0][:,0]
        query = query + update
        return query + self.ff(self.ff_norm(query))


class LocalBoundaryExpert(nn.Module):
    def __init__(self, channels=(256,256,256), hidden=256, nc=10, heads=8,
                 ff=1024, private_seed=30000, chunk_size=32):
        super().__init__()
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0 or len(channels) != 3:
            raise ValueError('requires three feature scales and positive chunk_size')
        # Constructor must not consume the public training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(private_seed)
            self.projections = nn.ModuleList(nn.Conv2d(c,hidden,1) for c in channels)
            self.position = nn.Linear(3, hidden)
            self.distribution = nn.Linear(8, hidden)
            self.blocks = nn.ModuleList(LocalAttentionBlock(hidden,heads,ff) for _ in range(2))
            self.norm = nn.LayerNorm(hidden)
            self.box_out = nn.Linear(hidden,132)
            self.class_out = nn.Linear(hidden,nc)
            for layer in (self.box_out,self.class_out):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, query, corners, classes, boxes, features):
        if query.shape[:2] != corners.shape[:2] or corners.shape[-1] != 132:
            raise ValueError('query/corner contract changed')
        if len(features) != 3:
            raise ValueError('expert requires P2/P3/P4')
        if query.shape[1] == 0:
            return corners, classes
        projected = [p(f) for p,f in zip(self.projections,features)]
        grid,valid = boundary_grid(boxes)
        p = corners.detach().float().reshape(*corners.shape[:2],4,33).softmax(-1)
        entropy = -(p*p.clamp_min(1e-9).log()).sum(-1)
        mean = (p*torch.linspace(-1,1,33,device=p.device)).sum(-1)
        # Mean here is only an index-space feature; geometry uses the pinned support.
        query = query + self.distribution(torch.cat((entropy,mean),-1).to(query.dtype))
        outputs=[]
        for start in range(0,query.shape[1],self.chunk_size):
            stop=start+self.chunk_size
            local_grid=grid[:,start:stop]
            tokens=[]
            for scale,f in enumerate(projected):
                position=torch.cat((local_grid,torch.full_like(local_grid[...,:1],scale/2)), -1)
                tokens.append(sample_local_features(f,local_grid) + self.position(position.to(query.dtype)))
            token=torch.cat(tokens,2)
            mask=valid[:,start:stop].repeat(1,1,3)
            batch,count,_,width=token.shape
            q=query[:,start:stop].reshape(batch*count,width)
            token=token.reshape(batch*count,-1,width)
            mask=mask.reshape(batch*count,-1)
            for block in self.blocks:
                q=block(q,token,mask)
            outputs.append(q.reshape(batch,count,width))
        h=self.norm(torch.cat(outputs,1))
        return corners + self.box_out(h), classes + self.class_out(h)
