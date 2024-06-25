'''
Contains main methods for training a model.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from torch import Tensor, nn
import model
from torch.utils.tensorboard import SummaryWriter
import argparse, wandb, time, random, math, numpy, json
from typing import Union, Optional, Iterable, Any, NoReturn, ClassVar
from util import Data, config


def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ManageModel:
	def __init__(self, model: ClassVar = None) -> NoReturn:
		'''
			Parameters
			----------
			model: Union[ClassVar, None]
				model instance
		'''
		self.model = model
		self.optimizer = None
		self.loss = {}
		self.best_loss = 1e9
		self.elapsed_time = 0
		self.scaler = None
		self.aug_prob = 0.25


	def get_lr(self, epoch, warmup_iters=100, lr_decay_iters=2000):

		if epoch < warmup_iters:
			return config.lr # no warmup
			# return lr * epoch / warmup_iters

		if epoch > lr_decay_iters:
			return config.min_lr

		decay_ratio = (epoch - warmup_iters) / (lr_decay_iters - warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
		return config.min_lr + coeff * (config.lr - config.min_lr)


	def load_model(self, path: str) -> NoReturn:
		'''
			Load a model from path
			Parameters
			----------
			path: str
				Path to the model
		'''
		if not os.path.exists(path):
			print(f"Path '{path}' does not exist.")
			exit()
		checkpoint = torch.load(path)
		config.set_model_params(checkpoint['config'])
		config.train_data_load = Data(config.train_dirpath, config.device)
		config.test_data_load = Data(config.test_dirpath, config.device)
		model.config = config
		self.model = model.S2SModel()
		self.model.load_state_dict(checkpoint['model'])


	def net_health(self, epoch: int, lr: float, test_time: bool) -> NoReturn:
		'''
			Logging more information about the vectors, weights, 
			and one day gradients. Needs to be run after each iter.
			Parameters
			----------
			epoch: int
				current epoch
			lr: float
				current learning rate
			test_time: bool
				true if it's the test time, false otherwise
		'''
		for i, layer in enumerate(config.layers_health):
			for k, v in layer.items():
				if config.tensorboard:
					self.tensorboard_writer.add_scalar(f"layers.{i}.{k}", v, epoch, new_style=True)
				if config.wandb:
					wandb.log({
						f"layers.{i}.{k}": v,
					})


		if config.tensorboard and config.decay_lr:
			self.tensorboard_writer.add_scalar('lr', lr, epoch, new_style=True)
		if config.wandb:
			wandb.log({
				'lr': lr,
			})
		if test_time:
			for name, param in self.model.named_parameters():
				grad_norm = None if param.grad is None else param.grad.data.norm(2).item()
				weight_norm = None if 'weight' not in name else param.norm(2).item()
				if config.tensorboard:
					if grad_norm is not None:
						self.tensorboard_writer.add_scalar(f"{name}.gradient.norm", grad_norm, epoch, new_style=True)
					if weight_norm is not None:
						self.tensorboard_writer.add_scalar(f"{name}.weight.norm", weight_norm, epoch, new_style=True)

				if config.wandb:
					if grad_norm is not None:
						wandb.log({
							f"{name}.gradient.norm": grad_norm,
						})
					if weight_norm is not None:
						wandb.log({
							f"{name}.gradient.norm": weight_norm,
						})


		config.layers_health = []

		if config.tensorboard:
			self.tensorboard_writer.flush()


	def pre_train(self) -> NoReturn:
		'''
			Prepare the model for training.
			Init optimizer, tensorboard, wandb, dirs, model, etc.
		'''
		self.model.train()
		self.model.to(config.device)

		if self.optimizer is None:
			use_fused = config.device == 'cuda'

			self.optimizer = torch.optim.AdamW(
				self.model.parameters(),
				lr=config.lr,
				betas=(config.beta1, config.beta2),
				eps=1e-8,
				fused=use_fused,
			)

		ver = f'{config.variation}'

		variation = f"{ver}_{config.nlayers}nl_\
		{config.dim}d_{config.dropout}\
		do_{config.block_size}bs_{config.lr}lr_{int(config.decay_lr)}\
		dlr".strip().replace('\t', '').replace(' ', '')

		if config.tensorboard:
			self.tensorboard_writer = SummaryWriter(
				comment='_' + variation,
				filename_suffix='',
			)
		if config.wandb:
			self.wandb_init = wandb.init(
				project='S2SModel',
				name=variation,
				config=config.get_model_params(),
			)
		self.path_format = os.path.join(
			config.workdir,
			f"S2SModel_{variation}",
		)

		if config.wandb:
			self.wandb_init.watch(self.model, log='all')

		os.makedirs(config.workdir, exist_ok=True)
		self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))


	def pre_test(self) -> NoReturn:
		'''
			Prepare the language model for testing.
		'''
		self.model.eval()
		self.model.to(config.device)


	def post_train(self) -> NoReturn:
		'''
			Tasks that relate to after training happen here.

		'''
		if config.tensorboard:
			hyparams = config.get_model_params(abstract=True)
			metrics = {}
			hyparams['test_loss'] = self.loss['test'].item()
			hyparams['train_loss'] = self.loss['train'].item()
			hyparams['elapsed_time'] = round(self.elapsed_time / 60, 4)
			hyparams['parameters'] = config.parameters
			for i in hyparams:
				self.tensorboard_writer.add_text(i, str(hyparams[i]))
			self.tensorboard_writer.flush()
			self.tensorboard_writer.close()
		if config.wandb:
			wandb.log({
				'meta/params': config.parameters,
				'meta/elapsed_time': round(self.elapsed_time / 60, 4)
			})


	def post_test(self) -> NoReturn:
		pass


	@torch.no_grad()
	def calculate_loss(self) -> dict[str, int]:
		'''
			We select eval_iterations chunks from both train and test data
			and save their losses. All in all, evaluating the perf
			of the model on train and test data.
			Parameters
			----------

			Returns
			-------
			loss: dict
				testing process loss
		'''

		self.model.eval()

		out = {}
		for split in ('train', 'test'):
			# A tensor to capture the losses
			losses = torch.zeros(config.eval_iterations)
			for k in range(config.eval_iterations):
				if split == 'train':
					X, y = config.train_data_load.get_batch()
				else:
					X, y = config.test_data_load.get_batch()
				with config.autocast:
					_, loss = self.model(X, y)
				losses[k] = loss.item()
			out[split] = losses.mean()

		self.model.train()

		return out


	@torch.no_grad()
	def test(self, epoch: int) -> NoReturn:
		'''
			Generate a sequence, calculate loss, and log
			Parameters
			----------
			epoch: int
				current epoch
		'''
		state = config.mode
		config.mode = 'inference'
		elapsed = self.generator()

		print('-' * 10)
		print(f"[{epoch}] > Elapsed: {elapsed}")

		self.loss = self.calculate_loss()
		test_loss = round(self.loss['test'].item(), 5)
		train_loss = round(self.loss['train'].item(), 5)
		test_pp = round(torch.exp(self.loss['test']).item(), 5)
		train_pp = round(torch.exp(self.loss['train']).item(), 5)
		print(f"[{epoch}] > train: {train_loss}, {train_pp} PP, test: {test_loss}, {test_pp} PP")
		print('-' * 30)

		if config.tensorboard:
			self.tensorboard_writer.add_scalar(f'train_loss', train_loss, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'test_loss', test_loss, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'train_pp', train_pp, epoch, new_style=True)
			self.tensorboard_writer.add_scalar(f'test_pp', test_pp, epoch, new_style=True)
			self.tensorboard_writer.flush()

		if config.wandb:
			wandb.log({
				f'train/loss': train_loss,
				f'test/loss': test_loss,
				f'train/perplexity': train_pp,
				f'test/perplexity': test_pp,
				'iter': epoch,
			})

		config.mode = state


	def train_procedure(self) -> NoReturn:
		'''
			Running one iteration.
			Parameters
			----------
			Returns
			-------
			bool:
				specifies whether the training should continue or not.
		'''

		epoch = 0

		X, Y = config.train_data_load.get_batch()
		while True:
			test_time = epoch % config.eval_step == config.eval_step - 1
			lr = self.get_lr(epoch + 1) if config.decay_lr else config.lr

			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

			start = time.time()
			for accum_step in range(config.accumulation_steps):
				with config.autocast:
					pred, loss = self.model(X, Y)
					loss = loss / config.accumulation_steps
				X, Y = config.train_data_load.get_batch()
				self.scaler.scale(loss).backward()


			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(
				self.model.parameters(),
				config.grad_clip,
			)

			self.scaler.step(self.optimizer)
			self.scaler.update()
			self.optimizer.zero_grad(set_to_none=True)

			stop = time.time()
			self.elapsed_time += stop - start

			# If it's the right time to test the model
			if test_time:
				with torch.inference_mode():
					self.test(epoch)
				if config.save_checkpoint or self.loss['test'] < self.best_loss:
					self.best_loss = self.loss['test']
					# torch.save({
					# 	'model': self.model.state_dict(),
					# 	'optimizer': self.optimizer.state_dict(),
					# 	'config': config.get_model_params(),
					# 	'train_loss': self.loss['train'],
					# 	'test_loss': self.loss['test'],
					# 	'epoch': epoch,
					# 	}, self.path_format + f"_{epoch}.pt")

			epoch += 1
			print(epoch, end='\r')
			if epoch > config.iterations:
				break


	def train(self) -> NoReturn:
		'''
			Training process.
		'''

		self.pre_train()

		try:
			self.train_procedure()
		except KeyboardInterrupt:
			print(f"Keyboard interrupt.")

		self.post_train()


	@torch.no_grad()
	def generator(self) -> tuple[float, float]:
		'''
			Generate a sequence with seq_len length and return it
			along with time elapsed.
			Parameters
			----------
			seq_len: int
				sequence length you want to create
			Returns
			-------
			took: float
				elapsed time to generate the sequence
			took_per_token: float
				elapsed time to generate each token
		'''
		self.pre_test()

		X, _ = config.test_data_load.get_batch(batch_size=1)
		X2, _ = config.train_data_load.get_batch(batch_size=1)

		start = time.time()

		with config.autocast:
			generated, _ = self.model(X)
			generated2, _ = self.model(X2)
		torch.save([x.cpu().numpy() for x in X], 'sample_i.pt')
		torch.save(generated.cpu().numpy(), 'sample_o.pt')
		torch.save([x.cpu().numpy() for x in X2], 'train_sample_i.pt')
		torch.save(generated2.cpu().numpy(), 'train_sample_o.pt')

		end = time.time()
		took = end - start
		self.post_test()

		return took


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--action', '-a', type=str, help='train, and test', required=True)
	parser.add_argument('--device', type=str, default=config.device, help=f"device type, default {config.device}")
	parser.add_argument('--workdir', type=str, default=config.workdir, help=f"directory to save models, default {config.device}")
	parser.add_argument('--load', type=str, default=config.load, help='path to a model to start with')
	parser.add_argument('--variation', '-v', type=str, default=config.variation, help=f"model variation, default {config.variation}")
	parser.add_argument('--iterations', '-i', type=int, default=config.iterations, help=f"number of training iterations, default {config.iterations}")
	parser.add_argument('--lr', '-lr', type=float, default=config.lr, help=f"learning rate, default {config.lr}")
	parser.add_argument('--min-lr', '-ml', type=float, default=config.min_lr, help=f"minimum learning rate, default {config.min_lr}")
	parser.add_argument('--dropout', '-do', type=float, default=config.dropout, help=f"dropout prob, default {config.dropout}")
	parser.add_argument('--nlayers', '-nl', type=int, default=config.nlayers, help=f"number of blocks, default {config.nlayers}")
	parser.add_argument('--dim', '-d', type=int, default=config.dim, help=f"embedding size, default {config.dim}")
	parser.add_argument('--accumulation-steps', '-as', type=int, default=config.accumulation_steps, help=f"accumulation steps, default {config.accumulation_steps}")
	parser.add_argument('--block-size', '-bs', type=int, default=config.block_size, help=f"length input sequence, default {config.block_size}")
	parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help=f"batch size, default {config.batch_size}")
	parser.add_argument('--topk', type=int, default=config.topk, help=f"topk sampling, default {config.topk}")
	parser.add_argument('--wandb', action='store_true', default=config.wandb, help=f"use wandb for visualization, default {config.wandb}")
	parser.add_argument('--tensorboard', action='store_true', default=config.tensorboard, help=f"use tensorboard for visualization, default {config.tensorboard}")
	parser.add_argument('--compile', action='store_true', default=config.compile, help=f"compile the model for faster training, default {config.compile}")
	parser.add_argument('--decay-lr', action='store_true', default=config.decay_lr, help=f"decay learning rate, default {config.decay_lr}")
	parser.add_argument('--moe', action='store_true', default=config.moe, help=f"mixture of experts, default {config.moe}")
	args = parser.parse_args()

	config.set_args(args)
	task = ManageModel()

	match config.action:
		case 'train':
			config.mode = 'train'
			if config.load != '':
				task.load_model(config.load)
			else:
				config.train_data_load = Data(config.train_dirpath, config.device, mode='train')
				config.test_data_load = Data(config.test_dirpath, config.device, mode='test', augment=True)
				model.config = config
				themodel = model.S2SModel()
				task.model = torch.compile(themodel) if config.compile else themodel
			task.train()
		case 'test':
			pass
		case _:
			print('Invalid action.')
