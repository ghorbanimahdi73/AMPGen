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
parser.add_argument('--top-k', type=int, default=0, help='top-k sampling in decoder.')
parser.add_argument('--top-p', type=float, default=0., help='top-p sampling in decoder.')


args = parser.parse_args()

load_file = args.load_file + '/train_args.txt'

if args.top_p > 0:
	save_csv = args.load_file + '/seqs_top_p_' + str(args.top_p) + '.csv'
elif args.top_k > 0:
	save_csv = args.load_file + '/seqs_top_k_' + str(args.top_k) + '.csv'
else:
	save_csv = args.load_file + '/seqs.csv'

config = json.load(open(load_file))
parser.set_defaults(**config)

args = parser.parse_args()

print(args)


use_gpu = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
seq_len = 30

#seed = 7
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)


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
bidirectional = True if args.bidirectional else False
args.train = False
num_sequences = args.num_sequences

model = RNN_VAE_attn(n_vocab, h_dim, z_dim, attn_dim, dropout_prob=0, 
					p_word_dropout=0, pretrained_embeddings=dataset.get_vocab_vectors(),
					freeze_embeddings=True, gpu=args.gpu, bidirectional=True, fix_length=fix_length)

load_model_name = os.path.join(args.load_file,'finalmodel.bin')
#checkpoint = torch.load(load_model_name)
#epoch_num = checkpoint['epoch']
model.load_state_dict(torch.load(load_model_name, map_location=lambda storage,loc:storage), strict=False)
#model.load_state_dict(checkpoint['state_dict'])

sequences = []
print('Begin generating ...')
i = 0
for i in tqdm(range(num_sequences)):
	model.eval()
	z = model.sample_z_prior(1)
	
	c = torch.Tensor([1]) # AMP
	c = c.view(1,-1)
	c = c.cuda() if model.gpu else c
	sample_idxs = model.sample_sequence_attn(z, temp=1, attn_prior=True, usesam=True, top_p=args.top_p, top_k=args.top_k, min_length=5)
	pep_seqs = dataset.idx2sequence(sample_idxs)
	pep_seqs.replace(' ','')
	sequences.append(pep_seqs)

sequences_final = [i.replace(' ','') for i in sequences]
df = pd.DataFrame({'sequence':sequences_final, 'label':'pos'})
print(df.head(10))
df.to_csv(save_csv, index=None)