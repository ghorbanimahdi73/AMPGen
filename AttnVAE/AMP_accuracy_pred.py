import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import json
import argparse
from AMP_dataset import AMP_dataset
from AMP_model import Conv_LSTM_selfattn

parser = argparse.ArgumentParser()
parser.add_argument('--load-file', type=str, help='csv file to load')
parser.add_argument('--model-file', type=str, default='model_1', help='model file name for accuracy prediction')

args = parser.parse_args()
model_file =  args.model_file + '/finalmodel.pt'

config = json.load(open(args.model_file+'/train_args.txt'))
parser.set_defaults(**config)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 21
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
n_filters = args.n_filters
filter_size = args.filter_size
bidirectional = True if args.bidirectional else False
dropout = args.dropout
batch_size = args.batch_size
n_epochs = args.n_epochs
patience = args.patience
early_stop = True if args.early_stop else False
cross_val = True if args.cross_val else False
k_folds = args.k_folds

model = Conv_LSTM_selfattn(input_dim, embedding_dim, n_filters, filter_size,
		                  bidirectional, hidden_dim, dropout)
model.load_state_dict(torch.load(model_file))


generated_data = AMP_dataset(file_path=args.load_file, split_ratio=[0.9999,0.0001])
print('loaded dataset')
print('number of examples: ', generated_data.train_len + generated_data.valid_len)
all_preds = []

with torch.no_grad():
	for batch in generated_data.train_iter:
		(inputs, lengths), labels = batch

		predictions = model(inputs)
		rounded_preds = torch.round(torch.sigmoid(predictions)).flatten().numpy()
		all_preds.append(rounded_preds)

all_preds = np.concatenate(all_preds)
acc = len(all_preds[all_preds==1])/len(all_preds) * 100
print('accuracy: ', acc)
acc_file = args.load_file[:-3] + 'acc.txt'
with open(acc_file,'w') as f:
	f.write('model accuracy: {}'.format(acc))
