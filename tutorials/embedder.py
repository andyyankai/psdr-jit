import torch
import math
import numpy as np
from torch import nn

# reference: https://github.com/Ending2015a/hash-grid-encoding

PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

class Frequency(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 10
  ):
    super().__init__()
    self.n_levels = n_levels
    assert self.n_levels > 0
    freqs = 2. ** torch.linspace(0., n_levels-1, n_levels)
    self.register_buffer('freqs', freqs, persistent=False)
    # ---
    self.input_dim = dim
    self.output_dim = dim * n_levels * 2
  
  def forward(self, x: torch.Tensor):
    x = x.unsqueeze(dim=-1) # (..., dim, 1)
    x = x * self.freqs # (..., dim, L)
    x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1) # (..., dim, L*2)
    return x.flatten(-2, -1) # (..., dim * L * 2)


@torch.no_grad()
def fast_hash(ind: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
  d = ind.shape[-1]
  ind = (ind * primes[:d]) & 0xffffffff  # uint32
  for i in range(1, d):
    ind[..., 0] ^= ind[..., i]
  return ind[..., 0] % hashmap_size

class _HashGrid(nn.Module):
  def __init__(
    self,
    dim: int,
    n_features: int,
    hashmap_size: int,
    resolution: float
  ):
    super().__init__()
    self.dim = dim
    self.n_features = n_features
    self.hashmap_size = hashmap_size
    self.resolution = resolution

    # you can add more primes for supporting more dimensions
    assert self.dim <= len(PRIMES), \
      f"HashGrid only supports < {len(PRIMES)}-D inputs"

    # create look-up table
    self.embedding = nn.Embedding(hashmap_size, n_features)
    nn.init.uniform_(self.embedding.weight, a=-0.0001, b=0.0001)

    primes = torch.tensor(PRIMES, dtype=torch.int64)
    self.register_buffer('primes', primes, persistent=False)

    # create interpolation binary mask
    n_neigs = 1 << self.dim
    neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
    dims = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
    bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool) # (neig, dim)
    self.register_buffer('bin_mask', bin_mask, persistent=False)

  def forward(self, x: torch.Tensor):
    # x: (b..., dim), torch.float32, range: [0, 1]
    bdims = len(x.shape[:-1])
    x = x * self.resolution
    xi = x.long()
    xf = x - xi.float().detach()
    xi = xi.unsqueeze(dim=-2) # (b..., 1, dim)
    xf = xf.unsqueeze(dim=-2) # (b..., 1, dim)
    # to match the input batch shape
    bin_mask = self.bin_mask.reshape((1,)*bdims + self.bin_mask.shape) # (1..., neig, dim)
    # get neighbors' indices and weights on each dim
    inds = torch.where(bin_mask, xi, xi+1) # (b..., neig, dim)
    ws = torch.where(bin_mask, 1-xf, xf) # (b...., neig, dim)
    # aggregate nehgibors' interp weights
    w = ws.prod(dim=-1, keepdim=True) # (b..., neig, 1)
    # hash neighbors' id and look up table
    hash_ids = fast_hash(inds, self.primes, self.hashmap_size) # (b..., neig)
    neig_data = self.embedding(hash_ids) # (b..., neig, feat)
    return torch.sum(neig_data * w, dim=-2) # (b..., feat)

class MultiResHashGrid(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 15,
    base_resolution: int = 16,
    finest_resolution: int = 512,
  ):
    # https://nvlabs.github.io/instant-ngp/
    super().__init__()
    self.dim = dim
    self.n_levels = n_levels
    self.n_features_per_level = n_features_per_level
    self.log2_hashmap_size = log2_hashmap_size
    self.base_resolution = base_resolution
    self.finest_resolution = finest_resolution

    # from paper eq (3)
    b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(base_resolution-1))

    levels = []
    for level_idx in range(n_levels):
      resolution = math.floor(base_resolution * (b ** level_idx))
      hashmap_size = min(resolution ** dim, 2 ** log2_hashmap_size)
      levels.append(_HashGrid(
        dim = dim,
        n_features = n_features_per_level,
        hashmap_size = hashmap_size,
        resolution = resolution
      ))
    self.levels = nn.ModuleList(levels)

    self.input_dim = dim
    self.output_dim = n_levels * n_features_per_level

  def forward(self, x: torch.Tensor):
    return torch.cat([level(x) for level in self.levels], dim=-1)

def get_hashgrid_embedder(input_dims):
    embedder_obj = MultiResHashGrid(input_dims)
    embedder_obj = embedder_obj.to(device='cuda', dtype=torch.float32)
    return embedder_obj, embedder_obj.output_dim

def get_position_embedder(input_dims):
    embedder_obj = Frequency(input_dims)
    embedder_obj = embedder_obj.to(device='cuda', dtype=torch.float32)
    return embedder_obj, embedder_obj.output_dim
