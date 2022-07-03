import os
import torch
import numpy as np
from torch.autograd import Variable
import random
from dataset import AMP_dataset
from model4 import RNN_VAE_attn
from tqdm import trange
import torch.optim as optim
from args import buildParser
import sys
import warnings
import json
from utils import *
from transformers import get_linear_schedule_with_warmup, AdamW

args = buildParser().parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using GPU:', torch.cuda.is_available())
# ignore deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

batch_size = args.batch_size
seq_len = 30
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
fix_length = True if args.fix_length==True else False
print(fix_length)
# make the train-dataset
if fix_length:
	dataset = AMP_dataset(batch_size=batch_size, file_train='amp_train.csv', file_valid='amp_valid.csv', file_test='amp_test.csv', device=device, fixlen=30)
else:
	dataset = AMP_dataset(batch_size=batch_size, file_train='amp_train.csv', file_valid='amp_valid.csv', file_test='amp_test.csv', device=device)

n_vocab = 24
h_dim = args.h_dim
z_dim = args.z_dim
c_dim = args.c_dim
attn_dim = h_dim
attn_prior = True if args.attn_prior else False
anneal_type = args.anneal_type 
epochs = args.epochs
total_steps = (dataset.train_len//batch_size)*epochs
log_interval = int(total_steps//1000) # print loss every ... step
max_weight = args.max_weight # maximum weight for KL term in loss
lr = args.lr
kl_attn_weight = args.kl_attn_weight # attention weight in KL
dropout_prob = args.dropout_prob
args.use_MMD = True if args.use_MMD else False
save_file = args.save_file
save_name = args.save_file
use_hmean = True if args.use_hmean  else False
kld_start_inc = int(total_steps/10)
bidirectional = True if args.bidirectional else False
gpu = True if args.gpu else False

if anneal_type is not None and anneal_type in ['monotonic','cyclic']:
	use_anneal = True 
else:
	use_anneal = False

if args.use_MMD: # if using MMD loss do not perform KL annealing
	use_anneal = False


if anneal_type == 'monotonic':
	beta = frange(0.01, max_weight, (max_weight/total_steps)*2, total_steps, total_steps//4)
	#beta = monoton_sigmoid(0.001, max_weight, total_steps)
elif anneal_type == 'cyclic':
	#beta = frange_cycle_linear(0, max_weight, total_steps, n_cycle=4, ratio=0.5)
	beta = frange_cycle_sigmoid(0.01, max_weight, total_steps, n_cycle=3, ratio=0.5)
	#beta = frange_cycle_cosine(0, max_weight, total_steps, n_cycle=4, ratio=0.5)

model = RNN_VAE_attn(n_vocab, h_dim, z_dim, attn_dim, dropout_prob=args.dropout_prob, 
					p_word_dropout=args.word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
					freeze_embeddings=False, gpu=gpu, bidirectional=bidirectional, fix_length=fix_length)


def count_params(model):
	'''
	count the number of parameters in the model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters in the model: ', count_params(model))


if gpu:
	model.cuda()
	print('using GPU')

optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-9)

#optimizer = AdamW(model.parameters(), lr=lr, eps=1e-9)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps)//8, num_training_steps=total_steps)

if not os.path.exists(args.save_file):
	os.makedirs(args.save_file)


with open(save_file+'/train_args.txt','w') as f:
	json.dump(args.__dict__, f, indent=2)

print(args)
args_file = os.path.join(args.save_file, 'args.txt')
with open(args_file,'w') as f:
	f.write(str(args))


tolstep = 0
print('---Running training ----')
print(' Number of examples = {}'.format(dataset.train_len + dataset.valid_len))
print(' Number of steps = {}'.format(total_steps))


for epoch in range(args.epochs):

	'''------- Training loop ------	'''
	for batch in range(dataset.train_len//batch_size):
		tolstep += 1
		model.train()
		(inputs, lengths), labels = dataset.next_batch(gpu=gpu)

		labels = labels.unsqueeze(0)
		labels = labels.transpose(0,1)
		optimizer.zero_grad()
		if gpu:
			inputs = inputs.cuda()
			labels = labels.cuda()

		recons_loss, kl_loss, kl_loss_attn, attn_wae_mmd_loss, wae_mmd_loss = model(sequence=inputs, labels=labels, lengths=lengths.to('cpu'))

		if use_anneal:
			if tolstep >= total_steps:
				kld_weight = beta[-1]
			else:
				kld_weight = beta[tolstep]
		else:
			kld_weight = max_weight		

		if args.use_MMD:
			loss = recons_loss + kld_weight * ( wae_mmd_loss + kl_attn_weight * attn_wae_mmd_loss)
		else:
			loss = recons_loss + kld_weight * (kl_loss + kl_loss_attn * kl_attn_weight)


		loss.backward()
		optimizer.step()
		grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.)
		#scheduler.step()
		''' ----------  Calculate loss for validation set ---------'''
		if tolstep % log_interval == 1:
			# generate a sequence just for testing
			model.eval()
			z = model.sample_z_prior(1)
			z = z.cuda() if  model.gpu else z
			c = Variable(torch.tensor(1))
			c = c.cuda() if model.gpu else c
			sample_idxs = model.sample_sequence_attn(z, attn_prior=args.attn_prior)
			sample_sent = dataset.idx2sequence(sample_idxs)
			print()

			# calculate the loss for the validation set

			
			if args.use_MMD:
				#out_loss = 'validation: iter-{}, Loss" {:.4f}, Recons_loss: {:.4f}, z-MMD: {:.4f}, attn-MMD: {:.4f}, Grad_norm: {:.4f}, KL_weight: {:.4f}'.format(
				#        tolstep, loss_val, recons_loss_val, wae_mmd_loss_val, attn_wae_mmd_loss_val, grad_norm, kld_weight)
				loss_out_train = 'training: iter-{}, Loss" {:.4f}, Recons_loss: {:.4f}, z-MMD: {:.4f}, attn-MMD: {:.4f}, Grad_norm: {:.4f}, KL_weight: {:.4f}'.format(
				        tolstep, loss, recons_loss, wae_mmd_loss, attn_wae_mmd_loss, grad_norm, kld_weight)
			else:
				#out_loss = 'validation: iter-{}, Loss" {:.4f}, Recons_loss: {:.4f}, KL: {:.4f}, KL_attn: {:.4f}, Grad_norm: {:.4f}, KL_weight: {:.4f}'.format(
				#       tolstep, loss_val, recons_loss_val, kl_loss_val, kl_loss_attn_val, grad_norm, kld_weight)
				loss_out_train = 'train: iter-{}, Loss" {:.4f}, Recons_loss: {:.4f}, KL: {:.4f}, KL_attn: {:.4f}, Grad_norm: {:.4f}, KL_weight: {:.4f}'.format(
				        tolstep, loss, recons_loss, kl_loss, kl_loss_attn, grad_norm, kld_weight)
			
			print(loss_out_train)
			#print(out_loss)

			with open(args.save_file+'/losses.txt','a') as f:
				f.write(loss_out_train)
				f.write('\n')

			print('Sample: "{}"'.format(sample_sent))


	if args.save_checkpoints and epoch % 5 == 0: # save the checkpoint every 5 epochs of training
		torch.save({
			'epoch': epoch,
			'state_dict': model.state_dict()
			}, args.save_file + '/logs_' + str(epoch) + '.pt')


file_name = os.path.join(args.save_file, 'finalmodel.bin')
torch.save(model.state_dict(), file_name)		
