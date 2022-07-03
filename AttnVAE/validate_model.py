import torch
import os
import numpy as np
from torch.autograd import Variable
import random
from dataset import AMP_dataset
from model4 import RNN_VAE_attn
from tqdm import tqdm
import torch.optim as optim
import argparse
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument('--load-file', help='Load setting from json file format.')
parser.add_argument('--num-sequences', type=int, default=5000, help='number of peptides to generate')


args = parser.parse_args()

load_file = args.load_file + '/train_args.txt'

config = json.load(open(load_file))
parser.set_defaults(**config)

args = parser.parse_args()

print(args)

use_gpu = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
seq_len = 30

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

fix_length = True if args.fix_length==True else False

print(fix_length)
# make the train-dataset
if fix_length:
	dataset = AMP_dataset(batch_size=batch_size, file_train='amp_seqs.csv', file_valid='amp_valid.csv', file_test='amp_test.csv', device=device, fixlen=30)
else:
	dataset = AMP_dataset(batch_size=batch_size, file_train='amp_seqs.csv', file_valid='amp_valid.csv', file_test='amp_test.csv', device=device)


n_vocab = 24
h_dim = args.h_dim
z_dim = args.z_dim
c_dim = args.c_dim
attn_dim  = h_dim
attn_prior = True
use_anneal = True
epochs = args.epochs
total_steps = dataset.train_len//args.batch_size
log_interval = int(total_steps//10)
max_weight = args.max_weight
cnew_dim = args.cnew_dim
lr = args.lr
kl_attn_weight = args.kl_attn_weight
dropout_prob = args.dropout_prob
attn_dim = h_dim
anneal_type = args.anneal_type

args.train = False
num_sequences = args.num_sequences

if anneal_type is not None and anneal_type in ['monotonic','cyclic']:
	use_anneal = True 
else:
	use_anneal = False

if args.use_MMD: # if using MMD loss do not perform KL annealing
	use_anneal = False

model = RNN_VAE_attn(n_vocab, h_dim, z_dim, attn_dim, dropout_prob=0, 
					p_word_dropout=0, pretrained_embeddings=dataset.get_vocab_vectors(),
					freeze_embeddings=True, gpu=args.gpu, bidirectional=True, fix_length=fix_length)

load_model_name = os.path.join(args.load_file,'finalmodel.bin')
#checkpoint = torch.load(load_model_name)
#epoch_num = checkpoint['epoch']
model.load_state_dict(torch.load(load_model_name, map_location=lambda storage,loc:storage), strict=False)
#model.load_state_dict(checkpoint['state_dict'])

model.eval()
recons_losses_val = []
kl_losses_val = []
kl_losses_attn_val = []
for batch in range(dataset.valid_len//batch_size):
	(inputs, lengths), labels = dataset.next_validation_batch(gpu=args.gpu)
	labels = labels.unsqueeze(0)
	labels = labels.transpose(0,1)

	if args.gpu:
		inputs = inputs.cuda()
		labels = labels.cuda()

	recons_loss, kl_loss , kl_loss_attn, _, _ = model(sequence=inputs, labels=labels, lengths=lengths.to('cpu'))
	recons_losses_val.append(recons_loss.item())
	kl_losses_val.append(kl_loss.item())
	kl_losses_attn_val.append(kl_loss_attn.item())

recons_loss_val = np.mean(recons_losses_val)
kl_loss_val = np.mean(kl_losses_val)
kl_loss_attn_val = np.mean(kl_losses_attn_val)

print('recons_loss:', recons_loss_val)
print('kl_loss:', kl_loss_val)
print('kl_loss_attn_val:', kl_loss_attn_val)

validation_loss = 'recons_loss {:.4f}, kl_loss: {:.4f}, kl_loss_attn: {:.4f}'.format(recons_loss_val, kl_loss_val, kl_loss_attn_val)

with open(args.load_file+'/validation_losses.txt', 'a') as f:
	f.write(validation_loss)


