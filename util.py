from functools import lru_cache
from typing import Union, Optional, Iterable, Any, NoReturn, ClassVar
import numpy as np
import math, os, pathlib, random, argparse
from contextlib import nullcontext
from dataclasses import dataclass
import subprocess
import shutil
from subprocess import CalledProcessError, run

import torch, torchaudio
from torch import Tensor
F = torch.nn.functional
from encodec import EncodecModel

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400 # win length you might call.
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 20 # max length of audio input
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 192000 samples in a 12-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token

exact_div = lambda a,b: a // b
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class MelParams:
	epoch_continuous_mode = 2
	version = '0'
	use_noise_background = False
	use_speed_change = True
	use_freq_mask = False
	use_time_stretch = False
	noise_background_dir = 'noise_dir'
	regularization_on_raw_audio = True
	regularization_on_mel = False
	regularization_on_test_data = False
	regularization_on_train_data = True
	learning_rate = 1e-4
	n_audio_ctx = 600
	n_audio_state = 192
	n_audio_head = 2
	n_audio_layer = 6
	n_text_ctx = 11
	n_text_state = 192
	n_text_head = 2
	n_text_layer = 6
	audio_dropout = 0.1
	text_dropout = 0.1
	attention_dropout = 0.1
	freq_mask = 15
	time_mask = 70
	sample_rate = SAMPLE_RATE
	n_mels = N_MELS
	n_fft = N_FFT
	hop_length = HOP_LENGTH
	chunk_length = CHUNK_LENGTH
	n_samples = N_SAMPLES
	n_samples_per_token = HOP_LENGTH * 2
	n_frames = N_FRAMES
	frames_per_second = FRAMES_PER_SECOND
	tokens_per_second = TOKENS_PER_SECOND
	specaug_rate = 0.5

mel_params = MelParams()


block_size = 128
dim = 512
nheads = 4 if dim <= 512 else 6
params = {
	'block_size': block_size,
	'base_dim': 0,
	'dim': dim,
	'mqa_window_size': 256,
	'mqa_head_dim': (dim // nheads),
	'nheads': nheads,
	'lr': 7e-4, # Learning rate
	'min_lr': 1e-4, # Min learning rate
	'beta1': 0.9,
	'beta2': 0.99, # The less, the more stable
	'decay_lr': True,
	'eval_step': 50, # Every n step, we do an evaluation.
	'iterations': 5001, # Like epochs
	'eval_iterations': 25, # Do n step(s), and calculate loss.
	'batch_size': 2,
	'nlayers': 16,
	'nheads': 6,
	'accumulation_steps': 2,
	'dropout': 0.0,
	'voice_duration': 20,
	'sample_rate': 24000,
	'weight_decay': 0.0,
	'grad_clip': 1.0,
	'vocab_size': 0,
	'device': 'cuda' if torch.cuda.is_available() else 'cpu',
	'variation': 'voice', # When we change something, change this to distinguish different variations.
	'workdir': 'workdir',
	'load': '',
	'train_dirpath': 'train_pairs',
	'test_dirpath': 'test_pairs',
	'action': 'train',
	'mode': 'train',
	'data_load': None,
	'wandb': False,
	'tensorboard': False,
	'save_checkpoint': False,
	'parameters': None,
	'details': '',
	'compile': False,
	'dtype': 'float16',
	'autocast': None,
	'bias': False,
	'topk': -1,
	'token_type': 'token',
	'moe': False, # mixture of experts
	'cross': True,
	'health': 2, # 0 for nothing, 1 for vector values, 2 for weight values of all layers
	'layers_health': [],
	'n_codebooks': 8,
}


class Config:
	def __init__(self, data_dict: dict) -> NoReturn:
		'''
			Given a data_dict, the class treats each key/val as an object.
			Parameters
			----------
			data_dict: dict
				a dict that key is a property and value is its value
		'''
		self.__data_dict__ = data_dict

	def __getattr__(self, k: Union[int, str, bytes]) -> Any:
		'''
			Given a key, it returns its data if it exists, otherwise None.
			Parameters
			----------
			k: str
				key
			Returns
			-------
			v: Union[any type]
				the value of the k
		'''
		if k in self.__data_dict__:
			return self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def __setattr__(self, k: Union[int, str, bytes], v: Any) -> NoReturn:
		if k == '__data_dict__':
			super().__setattr__(k, v)
		else:
			self.__data_dict__[k] = v


	def __delattr__(self, k: Union[int, str, bytes]) -> NoReturn:
		'''
			Given a key, it deletes it from data dict if it exists.
			Parameters
			----------
			k: str
				key that needs to be removed
		'''
		if k in self.__data_dict__:
			del self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def set_args(self, args: argparse.Namespace) -> NoReturn:
		'''
			Given an object of argparse, the method adds all the KVs to the data.
			Parameters
			----------
			args: argparse.Namespace
				parsed args object
		'''
		for kv in args._get_kwargs():
			k, v = kv
			self.__setattr__(k, v)

		after_conf_init()


	def get_model_params(self, abstract: bool = False) -> dict:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			abstract: bool
				True if you want to remove metadata from dictionary.
		'''
		if abstract:
			filters = (
				'data_load', 'action', 'load', 'workdir',
				'wandb', 'tensorboard', 'details', 'data_file',
				'variation', 'device', 'mode', 'autocast',
				'flash_attention', 'compile',
				'init_weight',
			)
		else:
			filters = ('data_load', 'load', 'iterations', 'autocast')
		params = {}
		for k in self.__data_dict__:
			if k not in filters:
				params[k] = self.__data_dict__[k]
		return params


	def set_model_params(self, params: dict) -> NoReturn:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			params: dict
				Key value parameters.
		'''

		filters = (
			'data_load', 'action', 'load', 'workdir', 'mode')
		for k in params:
			if k not in filters:
				self.__data_dict__[k] = params[k]


config = Config(params)
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(config.device)


def after_conf_init():
	'''
		boring
	'''
	if config.device == 'cuda':
		config.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else config.dtype
		# config.dtype = 'float32'
	ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
	config.autocast = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
	config.topk = None if config.topk <= 0 else config.topk

	torch.set_float32_matmul_precision('high')
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True


def get_codec(waveform):
	encoded = model.encode(waveform.pin_memory().to(config.device))[0][0]
	return encoded.squeeze(0)


def load_audio(file: str, sr: int = mel_params.sample_rate):
	cmd = [
		'ffmpeg',
		'-nostdin',
		'-threads', '0',
		'-i', file,
		'-f', 's16le',
		'-ac', '1',
		'-acodec', 'pcm_s16le',
		'-ar', str(sr),
		'-'
	]
	# fmt: on
	try:
		out = run(cmd, capture_output=True, check=True).stdout
	except CalledProcessError as e:
		raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

	return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = mel_params.n_samples, *, axis: int = -1):
	if torch.is_tensor(array):
		if array.shape[axis] > length:
			array = array.index_select(
				dim=axis, index=torch.arange(length, device=array.device)
			)

		if array.shape[axis] < length:
			pad_widths = [(0, 0)] * array.ndim
			pad_widths[axis] = (0, length - array.shape[axis])
			array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
	else:
		if array.shape[axis] > length:
			array = array.take(indices=range(length), axis=axis)

		if array.shape[axis] < length:
			pad_widths = [(0, 0)] * array.ndim
			pad_widths[axis] = (0, length - array.shape[axis])
			array = np.pad(array, pad_widths)

	return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = mel_params.n_mels) -> torch.Tensor:
	assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
	with np.load(
		os.path.join(os.path.dirname(__file__), 'assets', 'mel_filters.npz')
	) as f:
		return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
	audio: Union[str, np.ndarray, torch.Tensor],
	n_mels: int = mel_params.n_mels,
	padding: int = 0,
	device: Optional[Union[str, torch.device]] = None,
):

	if not torch.is_tensor(audio):
		if isinstance(audio, str):
			audio = load_audio(audio)
		audio = torch.from_numpy(audio)


	if device is not None:
		audio = audio.to(device)

	# if padding > 0:
	# 	audio = F.pad(audio, (0, padding))

	window = torch.hann_window(mel_params.n_fft).to(audio.device)
	stft = torch.stft(audio, mel_params.n_fft, mel_params.hop_length, window=window, return_complex=True)
	magnitudes = stft[..., :-1].abs() ** 2

	filters = mel_filters(audio.device, n_mels)
	mel_spec = filters @ magnitudes

	log_spec = torch.clamp(mel_spec, min=1e-10).log10()
	log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
	log_spec = (log_spec + 4.0) / 4.0
	return log_spec


def get_mel_freqs(
	file_path: str,
	no_mel: bool = False,
):
	if no_mel:
		return None, pad(torchaudio.load(file_path)[0].view(1, 1, -1), target=480000)
	mel = log_mel_spectrogram(file_path, padding=mel_params.n_samples, device=config.device)
	mel = pad_or_trim(mel, mel_params.n_frames)
	return mel.unsqueeze(0), pad(torchaudio.load(file_path)[0].view(1, 1, -1), target=480000)


def random_frequency_mask(log_mels, freq_mask_prob=0.6, freq_mask_num_masks=2, freq_mask_max_percentage=0.1):
	if len(log_mels.shape) == 2:
		log_mels = log_mels.unsqueeze(1)
	batch_size, time_dim, freq_dim = log_mels.shape
	masked_log_mels = log_mels.clone()

	for _ in range(freq_mask_num_masks):
		if torch.rand(1) < freq_mask_prob:
			start_freq = torch.randint(0, freq_dim, (1,)).item()
			num_freq_bins = torch.randint(1, int(freq_dim * freq_mask_max_percentage) + 1, (1,)).item()
			end_freq = min(start_freq + num_freq_bins, freq_dim)
			masked_log_mels[:, :, start_freq:end_freq] = 0
	return masked_log_mels


def pad(waves, target=750):
	if waves.size(-1) > target:
		waves = waves[:, :, :target]
	elif waves.size(-1) < target:
		waves = F.pad(waves, (0, target - waves.size(-1)))
	return waves



class Data(torch.utils.data.Dataset):
	'''
		Load the data from the json file.
		The json file should be a list of objects with the following keys:
			key: path to the audio file
			text: the text transcript
	'''

	def __init__(self, dir_path: Tensor, device: Union[str, torch.device], augment: bool = False, mode: str = 'train'):
		self.device = device
		self.mode = mode
		self.batch_size = config.batch_size
		self.dir_path = dir_path
		self.datalen = 19000
		self.offset = (torch.arange(8).view(-1, 1) * 1024)
		self.augment = augment
		self.dir_id = 0
		self.epoch_id = 0
		self.change_dir_after = 50
		self.max_dir_id = 20
		self.test_flag = True
		self.data_points = [os.path.join(self.dir_path, x) for x in os.listdir(self.dir_path) if 'en_' in x]
		self.datalen = len(self.data_points)

	def on_fly_load(self, dir_id):
		if isinstance(dir_id, str):
			dp = self.data_points
		else:
			if dir_id * 1024 >= self.datalen:
				self.dir_id = 0
				return
			dp = self.data_points[dir_id * 1024: (dir_id * 1024) + 1024]
		self.data_x1 = []
		self.data_x2 = []
		self.data_x3 = []
		self.data_y1 = []
		model.to('cuda')
		with torch.no_grad():
			for filename in dp:
				fpath_output_voice = filename.replace('en_', 'pe_')
				imel, iraw = get_mel_freqs(filename)
				_, oraw = get_mel_freqs(fpath_output_voice, no_mel=True)
				self.data_x2.append(iraw)
				self.data_x3.append(imel)
				self.data_y1.append(oraw)
			self.data_x3 = torch.cat(self.data_x3, dim=0)
			bsize_x = math.ceil(len(self.data_x2) / 16)
			bsize_y = math.ceil(len(self.data_y1) / 16)

			self.data_x1 = torch.cat([get_codec(torch.cat(self.data_x2[x*bsize_x: (x*bsize_x)+bsize_x], dim=0)) for x in range(16)]).cpu()
			self.data_y1 = torch.cat([get_codec(torch.cat(self.data_y1[x*bsize_y: (x*bsize_y)+bsize_y], dim=0)) for x in range(16)]).cpu()
			self.data_x2 = torch.cat(self.data_x2, dim=0).cpu()
			self.data_x1.to('cpu')
			self.data_y1.to('cpu')
			self.data_x2.to('cpu')

		model.to('cpu')
		self.datalen = len(self.data_x1)


	def __len__(self):
		return len(self.data_x)


	def get_batch(self,
		step: int,
		batch_size: int = -1,
	) -> tuple[Tensor, Tensor]:
		batch_size = self.batch_size if batch_size == -1 else batch_size
		if step % self.change_dir_after == 0 and self.test_flag and step != -1:
			self.on_fly_load(self.dir_id if self.mode == 'train' else 'test')
			self.dir_id += 1
			if self.dir_id == self.max_dir_id:
				self.dir_id = 0
			if self.mode == 'test':
				self.test_flag = False

		ix = torch.randint(0, self.datalen, (batch_size,))
		with torch.no_grad():
			if self.augment and self.mode == 'train':
				x = [self.data_x1[ix] + self.offset, random_frequency_mask(self.data_x2[ix]), random_frequency_mask(self.data_x3[ix])]
			else:
				x = [self.data_x1[ix] + self.offset, self.data_x2[ix], self.data_x3[ix]]
			x = [i.to(config.device, non_blocking=True) for i in x]
			y = self.data_y1[ix].to(config.device, non_blocking=True)
		return (
			x,
			y,
		)

