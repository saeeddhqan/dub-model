
from einops import rearrange, repeat, einsum
import torch, math, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple


from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from util import config, mel_params
from timm.models.layers import trunc_normal_
from mamba_ssm import Mamba2



class NonLinear(nn.Module):
	def __init__(self, dim: int = config.dim, coef: int = 2) -> NoReturn:
		super().__init__()
		self.dim = dim
		self.w1 = nn.Linear(self.dim, coef * self.dim, bias=False)
		self.w2 = nn.Linear(self.dim, coef * self.dim, bias=False)
		self.w3 = nn.Linear(coef * self.dim, self.dim, bias=False)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor, aux_loss: Tensor = None) -> Tensor:
		return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x))), None


class MixedActor(nn.Module):
	def __init__(self, dim=32, num_experts=4):
		super().__init__()
		self.num_experts = 4
		self.experts = nn.ModuleList([NonLinear(dim, idx * 2) for idx in range(1, num_experts + 1)])
		# Gating network
		self.gate = nn.Linear(dim, 4)
		self.aux_loss_weight = 0.01

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


class GaussianNoise(nn.Module):
	"""Gaussian noise regularizer."""

	def __init__(self, sigma=0.1):
		super().__init__()
		self.sigma = sigma
		self.register_buffer('noise', torch.tensor(0))

	def forward(self, x):
		if self.training and self.sigma != 0:
			scale = self.sigma * x.detach()
			sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
			x = x + sampled_noise
		return x 


class ConvNeXtBlock(nn.Module):
	def __init__(
		self,
		dim: int,
		kernel, dilation,
		layer_scale_init_value: float = 1e-6,
	):
		# ConvNeXt Block copied from Vocos.
		super().__init__()
		self.dwconv = nn.Conv1d(dim, dim, 
								kernel_size=kernel, padding=dilation*(kernel//2), 
								dilation=dilation, groups=dim
							)  # depthwise conv
		
		self.norm = nn.LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Linear(dim, dim * 4)  # pointwise/1x1 convs, implemented with linear layers
		self.act = nn.SiLU()
		self.pwconv2 = nn.Linear(dim * 4, dim)
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		x = self.dwconv(x)
		x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
		x = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(x))))
		x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
		x = residual + x
		return x


class DVAEDecoder(nn.Module):
	def __init__(self, idim=512, odim=512,
				 n_layer=12, bn_dim=512, hidden=512, 
				 kernel=7, dilation=2, dim=768
				):
		super().__init__()
		self.dim = dim
		self.conv_in = nn.Sequential(
			nn.Conv1d(idim, bn_dim, 3, 1, 1), nn.SiLU(),
			nn.Conv1d(bn_dim, hidden, 3, 1, 1)
		)
		self.decoder_block = nn.ModuleList([
			ConvNeXtBlock(hidden, kernel, dilation,)
			for _ in range(n_layer)])

		self.out_ln = nn.LayerNorm(dim)
		self.out_proj = nn.Linear(dim, dim)
		self.apply(self._init_weights)


	def _init_weights(self, m):
		if isinstance(m, (nn.Conv1d, nn.Linear)):
			trunc_normal_(m.weight, std=.02)
			nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.conv_in(x)
		for f in self.decoder_block:
			x = f(x)
		return self.out_ln(self.out_proj(x))


class RMSNorm(nn.Module):
	def __init__(self, dim: int, no_mul: bool = False):
		super().__init__()
		self.eps = 1e-6
		self.gamma = nn.Parameter(torch.ones(dim))

	def forward(self, x: Tensor) -> Tensor:
		return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.gamma


class RMSNorm2(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.eps = 1e-6

	def forward(self, x: Tensor) -> Tensor:
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MHA(nn.Module):
	def __init__(self, dim: int = 1024,
		head_dim: int = 128, nheads: int = 8,
		kv_heads: int = 1, window_size: int = 1024):
		super().__init__()
		self.nheads = nheads
		self.head_dim = dim // self.nheads
		self.window_size = window_size
		self.rotary = RotaryEmbedding(dim=self.head_dim)
		self.qkv = nn.Linear(dim, dim * 3, bias=False)
		self.output = nn.Linear(dim, dim, bias=False)
		self.drop = config.dropout
		self.resid_dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor, xmel: Tensor) -> Tensor:
		B, T, C = x.shape
		x = self.rotary(self.qkv(x).view(B, T, 3, self.nheads, self.head_dim))
		x = flash_attn_qkvpacked_func(x, dropout_p=self.drop if self.training else 0.0, causal=False, window_size=(-1, -1))
		return self.resid_dropout(self.output(x.view(B, T, -1)))


class MHACross(nn.Module):
	def __init__(self, dim: int = 1024,
		head_dim: int = 128, nheads: int = 8,
		kv_heads: int = 1, window_size: int = 1024):
		super().__init__()
		self.nheads = nheads
		self.head_dim = dim // self.nheads
		self.window_size = window_size
		self.rotary = RotaryEmbedding(dim=self.head_dim)
		self.q = nn.Linear(dim, dim, bias=False)
		self.kv = nn.Linear(dim, dim * 2, bias=False)
		self.output = nn.Linear(dim, dim, bias=False)
		self.drop = config.dropout
		self.resid_dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor, xmel: Tensor) -> Tensor:
		B, T, C = x.shape
		q = self.q(x).view(B, T, self.nheads, -1)
		kv = self.kv(xmel).view(B, xmel.size(1), 2, self.nheads, -1)

		q, kv = self.rotary(q, kv)

		scale = (C // self.nheads) ** -0.25
		q = q.view(*q.shape[:2], self.nheads, -1).permute(0, 2, 1, 3) * scale
		k = kv[:,:,0].view(*kv.shape[:2], self.nheads, -1).permute(0, 2, 3, 1) * scale
		v = kv[:,:,1].view(*kv.shape[:2], self.nheads, -1).permute(0, 2, 1, 3)

		# self.dropout(q @ k)

		x = F.softmax(q @ k, dim=-1)
		x = (x @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

		return self.resid_dropout(self.output(x))


class TransformerBlock(nn.Module):
	def __init__(self, dim: int = config.dim, cross: bool = True, moe: bool = True):
		super().__init__()
		self.dim = dim
		self.moe = moe
		self.cross = cross
		self.norm1 = RMSNorm(dim=self.dim)
		self.norm2 = RMSNorm(dim=self.dim)
		self.mha = MHACross(self.dim) if cross else MHA(self.dim)
		self.ffn = MixedActor(self.dim) if moe else NonLinear(self.dim)
		if cross:
			self.cnorm1 = RMSNorm(dim=self.dim)
			self.cnorm2 = RMSNorm(dim=self.dim)
			self.cmha = MHA(self.dim)
			self.cffn = MixedActor(self.dim, num_experts=2) if moe else NonLinear(self.dim)

	def forward(self, x: Tensor, xmel: Tensor, aux_loss: Tensor) -> Tensor:
		if self.cross:
			xmel = xmel + self.cmha(self.cnorm1(xmel), None)
			out = self.cffn(self.cnorm2(xmel))
			xmel = xmel + out[0]
			aux_loss = aux_loss + out[1] if self.moe else 0
		x = x + self.mha(self.norm1(x), xmel)
		out = self.ffn(self.norm2(x))
		x = x + out[0]
		aux_loss = aux_loss + out[1] if self.moe else 0
		return x, xmel, aux_loss


class TranslateBlocks(nn.Module):
	def __init__(self, dim, n_layers, cross: bool = True, moe: bool = True) -> NoReturn:
		super().__init__()
		self.dim = dim
		self.cross = cross
		self.cross_freq = 4
		self.dropout = nn.Dropout(config.dropout)
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		# cross attention once upon a time
		self.blocks = nn.ModuleList([
			TransformerBlock(
				self.dim,
				cross=bool(idx % self.cross_freq == self.cross_freq - 1) if cross else False,
				moe=moe
			)
			for idx in range(n_layers)
		])

	def forward(self, x: Tensor, xmel: Tensor, aux_loss: Tensor) -> Tensor:
		x = self.ln1(self.dropout(x))
		for i, block in enumerate(self.blocks):
			x, xmel, aux_loss = block(x, xmel, aux_loss)
		return self.ln2(x), aux_loss


class ToneBlocks(nn.Module):
	def __init__(self, dim, n_layers, moe: bool = True) -> NoReturn:
		super().__init__()
		self.dim = dim
		self.dropout = nn.Dropout(config.dropout)
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		self.blocks = nn.ModuleList([
			TransformerBlock(self.dim, cross=True, moe=moe)
			for idx in range(n_layers)
		])

	def forward(self, x: Tensor, xmel: Tensor, aux_loss: Tensor) -> Tensor:
		x = self.ln1(self.dropout(x))
		for i, block in enumerate(self.blocks):
			x, xmel, aux_loss = block(x, xmel, aux_loss)
		return self.ln2(x), aux_loss


class BackgroundBlocks(nn.Module):
	def __init__(self, dim, n_layers, signal_tokens: int = 1000, moe: bool = True) -> NoReturn:
		super().__init__()
		self.dim = dim
		self.signal_tokens = signal_tokens
		self.dropout = nn.Dropout(config.dropout)
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		self.blocks = nn.ModuleList([
			TransformerBlock(self.dim, cross=True, moe=moe)
			for idx in range(n_layers)
		])
		self.signal_encode = DVAEDecoder(idim=signal_tokens, odim=signal_tokens, n_layer=4, bn_dim=signal_tokens, hidden=signal_tokens, kernel=7, dilation=2, dim=dim)

	def forward(self, x: Tensor, signal: Tensor, aux_loss: Tensor) -> Tensor:
		x = self.ln1(self.dropout(x))
		signal = self.signal_encode(signal) # turn off this for experiment
		for i, block in enumerate(self.blocks):
			x, signal, aux_loss = block(x, signal, aux_loss)
		return self.ln2(x), aux_loss


class MelEncoder(nn.Module):
	def __init__(
		self, n_mels: int, n_ctx: int, n_state: int,
		n_head: int, n_layers: int, n_frames: int,
		dim: int, downsample: bool = True, moe: bool = True,
	):
		super().__init__()
		self.dim = dim
		self.ln1 = RMSNorm(n_frames)
		self.ln2 = RMSNorm(n_frames)
		self.ln3 = RMSNorm(n_frames // 2)
		# self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
		# self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
		# self.dropout = nn.Dropout(0.2)
		self.convs = DVAEDecoder(idim=n_mels, odim=n_mels, n_layer=n_layers, bn_dim=n_mels, hidden=n_mels, kernel=7, dilation=2, dim=2000)
		self.blocks = nn.ModuleList([TransformerBlock(self.dim, cross=False, moe=moe) for _ in range(n_layers)])
		self.downsample = nn.Linear(2000, self.dim, bias=False)
		self.ln_post = RMSNorm(self.dim)
		self.alpha = 0.5

	def forward(self, x: Tensor, aux_loss: Tensor):
		"""
		x : torch.Tensor, shape = (B, n_mels, n_ctx)
			the mel spectrogram of the audio
		"""
		x = F.silu(self.convs(x) * self.alpha)
		x = self.downsample(x)
		for block in self.blocks:
			x, _, aux_loss = block(x, None, aux_loss)
		return self.ln_post(x), aux_loss


class S2SModel(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim # 768
		self.mel_dim = mel_params.n_audio_state
		self.use_moe = config.moe
		self.moe = False
		self.block_size = config.block_size
		self.in_token_size = 1024
		self.translate_mel_encode = MelEncoder(
			mel_params.n_mels,
			mel_params.n_audio_ctx,
			mel_params.n_audio_state,
			mel_params.n_audio_head,
			4,
			mel_params.n_frames, dim=self.dim,
			moe=False
		)
		self.tone_mel_encode = MelEncoder(
			mel_params.n_mels,
			mel_params.n_audio_ctx,
			mel_params.n_audio_state,
			mel_params.n_audio_head,
			4,
			mel_params.n_frames, dim=self.dim,
			moe=False
		)
		self.mamba = nn.ModuleList([
			Mamba2(d_model=self.dim, d_state=64, d_conv=4, expand=2).to(config.device)
			for _ in range(4)
		])

		self.mamba_heads = nn.ModuleList([
			Mamba2(d_model=self.dim, d_state=64, d_conv=4, expand=2).to(config.device)
			for _ in range(8)
		])
		self.embs = nn.Embedding(self.in_token_size * 8, self.dim)

		self.translate = TranslateBlocks(self.dim, n_layers=4, cross=False, moe=False)
		self.tune_tone = ToneBlocks(self.dim, n_layers=2, moe=False)
		self.add_background = BackgroundBlocks(self.dim, n_layers=2, moe=False)
		self.signal_upsample = nn.Linear(480, 512, bias=False)
		self.signal_norm = RMSNorm2(self.dim)
		self.ln_res = RMSNorm(self.dim)
		self.head = [nn.Linear(self.dim, self.in_token_size, bias=False).to(config.device) for _ in range(8)]

		self.embs.weight = nn.Parameter(torch.cat([x.weight for x in self.head], dim=0))
		self.gaussian_noise = GaussianNoise()
		self.drop = nn.Dropout(0.1)
		self.count_params, self.wo_n_params = self.num_params()
		config.parameters = self.count_params  / 1e6
		print("Number of parameters: %.3fM, %.3fM" % (self.count_params  / 1e6,self.wo_n_params / 1e6))
		self.alpha = 0.5
		self.apply(self.norm_weights)


	def num_params(self) -> int:
		n_params = sum(p.numel() for p in self.parameters())
		n_params2 = n_params - self.embs.weight.numel()
		return n_params, n_params2


	def norm_weights(self, module):
		if isinstance(module, nn.Embedding):
			torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
		elif isinstance(module, nn.LayerNorm):
			nn.init.zeros_(module.bias)
			nn.init.ones_(module.weight)
		elif isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))


	def forward(self,
		seq: Tensor,
		y: Optional[Tensor] = None,
	) -> tuple[Tensor, Tensor]:

		seq, signal, mel = seq
		B, T, C = seq.shape

		signal = self.gaussian_noise(signal)
		mel = self.gaussian_noise(mel)
		seq = self.drop(self.embs(seq.view(B, -1)))

		seq = F.silu(self.ln_res(seq) * self.alpha)
		for f in self.mamba:
			seq = f(seq)
		aux_loss = torch.tensor(data=0.0).to(seq.device)
		mel1, aux_loss = self.translate_mel_encode(mel, aux_loss)
		mel2, aux_loss = self.tone_mel_encode(mel, aux_loss)
		loss = None
		predictions = []
		seq = seq.view(B, T, C, -1)
		signal = F.silu(self.signal_norm(self.signal_upsample(signal.view(B, 1000, 480)))  * self.alpha)
		for i in range(8):
			logits, aux_loss = self.translate(seq[:,i], mel1, aux_loss)
			logits, aux_loss = self.tune_tone(logits, mel2, aux_loss)
			logits, aux_loss = self.add_background(logits, signal, aux_loss)
			if y is not None:
				logits = self.head[i](self.mamba_heads[i](logits))
				predictions.append(torch.argmax(logits, dim=-1).unsqueeze(1))
				layer_loss = F.cross_entropy(logits.view(-1, self.in_token_size), y[:, i].flatten())
				if loss is None:
					loss = layer_loss
				else:
					loss = loss + layer_loss
			else:
				predictions.append(torch.argmax(self.head[i](self.mamba_heads[i](logits)), dim=-1).unsqueeze(1))
		predictions = torch.cat(predictions, dim=1).view(B, T, C)

		return predictions, (loss / 8) + (aux_loss / 8 if self.moe else 0) if loss else loss
