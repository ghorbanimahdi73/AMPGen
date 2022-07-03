import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy import data


class AMP_dataset:
	def __init__(self, batch_size=32, file_path='final_dataset_labeled.csv', split_ratio=[0.7, 0.3]):
		self.SEQ = data.Field(sequential=True, init_token=None, eos_token=None,
							  tokenize=self.tokenizer, unk_token=None,
							  include_lengths=True)
		self.LABEL = data.LabelField(dtype=torch.float)
		self.Amino_vocab = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9,
						'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18,
						'W':19, 'Y':20}
		self.fields = {'sequence': ('seq', self.SEQ), 'label':('label', self.LABEL)}
		self.SEQ.build_vocab(self.Amino_vocab)
		label_dict = {'pos':1.0, 'neg':0.0}
		self.LABEL.build_vocab(label_dict)
		self.dataset = data.TabularDataset(path=file_path, format='csv', fields=self.fields)
		self.train_data, self.valid_data = self.dataset.split(split_ratio=split_ratio)

		self.train_iter, self.valid_iter = data.BucketIterator.splits(
			(self.train_data, self.valid_data), batch_size=batch_size, device='cpu',
			shuffle=True, repeat=False, sort_key=lambda x: len(x.seq), sort_within_batch=True)


		self.train_len = len(self.train_data)
		self.valid_len = len(self.valid_data)

	def tokenizer(self, seq):
		return list(seq)

	def get_vocab_vectors(self):
		return self.SEQ.vocab.vectors

	def idx2sequence(self, idxs):
		return ' '.join([self.SEQ.vocab.itos[i] for i in idxs])

	def idx2label(self, idx):
		return self.LABEL.vocab.itos[idx]

