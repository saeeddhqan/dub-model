
from einops import rearrange, repeat, einsum
import torch, math, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple


from util import config, mel_params



class NonLinear(nn.Module):
	def __init__(self, dim: int = config.dim, coef: int = 2) -> NoReturn:
		super().__init__()
		self.dim = dim
		self.w1 = nn.Linear(self.dim, coef * self.dim, bias=False)
		self.w2 = nn.Linear(self.dim, coef * self.dim, bias=False)
		self.w3 = nn.Linear(coef * self.dim, self.dim, bias=False)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor) -> Tensor:
		return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MixedActor(nn.Module):
	def __init__(self, dim=32, num_experts=4):
		super().__init__()
		self.num_experts = 4
		self.experts = nn.ModuleList([NonLinear(dim, idx * 2) for idx in range(1, num_experts + 1)])
		# Gating network
		self.gate = nn.Linear(dim, 4)
		self.aux_loss_weight = 0.5

	def forward(self, x):
		coefficients = F.softmax(self.gate(x), dim=-1)
		output = torch.zeros_like(x)
		for i, layer in enumerate(self.experts):
			output = output + layer(x) * coefficients[:,:, i].unsqueeze(-1)

		return output, self.compute_auxiliary_loss(coefficients)

	def compute_auxiliary_loss(self, gate_scores):
		# gate_scores shape: (B*T, num_experts)
		avg_usage = (gate_scores.view(-1, self.num_experts)).mean(dim=0)
		return (((avg_usage - (1 / self.num_experts)) ** 2).sum()) * self.aux_loss_weight


test = torch.randn(2, 4, 32)
moe = MixedActor()
x, gate_prob = moe(test)
print(gate_prob.shape, x.shape)
print(moe.compute_auxiliary_loss(gate_prob))
