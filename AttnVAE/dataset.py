import numpy as np
import pandas as pd
from torchtext.legacy import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AMP_dataset:
	def __init__(self, batch_size=32, file_train='amp_train.csv', file_valid='amp_valid.csv', file_test='amp_test.csv', fixlen=None, device='cuda'):

		if fixlen is not None:
			self.SEQ = data.Field(sequential=True, init_token='<start>', eos_token='<eos>', tokenize=self.tokenizer,
							pad_token='<pad>', unk_token='<unk>', include_lengths=True, fix_length=fixlen+2)
		else:
			self.SEQ = data.Field(sequential=True, init_token='<start>', eos_token='<eos>', tokenize=self.tokenizer,
							pad_token='<pad>', unk_token='<unk>', include_lengths=True)
		
		self.LABEL = data.LabelField(dtype=torch.float)

		self.Amino_vocab = {'A':4, 'C':5,'D':6, 'E':7, 'F':8, 'G':9, 'H':10,
						    'I':11, 'K':12, 'L':13, 'M':14, 'N':15, 'P':16, 'Q':17, 
						    'R':18, 'S':19, 'T':20, 'V':21, 'W':22, 'Y':23}

		self.fields = {'sequence': ('seq', self.SEQ), 'label':('label', self.LABEL)}

		self.SEQ.build_vocab(self.Amino_vocab)
		label_dict = {'pos':1.0, 'neg':0.0}
		self.LABEL.build_vocab(label_dict)
		self.train_dataset = data.TabularDataset(path=file_train, format='csv', fields=self.fields)
		self.valid_dataset = data.TabularDataset(path=file_valid, format='csv', fields=self.fields)
		self.test_dataset = data.TabularDataset(path=file_test, format='csv', fields=self.fields)

		self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
				(self.train_dataset, self.valid_dataset, self.test_dataset), batch_size=batch_size, device=device,
				sort_within_batch=True, sort_key=lambda x:len(x.seq), shuffle=True, repeat=True)

		self.train_iter = iter(self.train_iter)
		self.valid_iter = iter(self.valid_iter)
		self.test_iter = iter(self.test_iter)

		self.train_len = len(self.train_dataset)
		self.valid_len = len(self.valid_dataset)
		self.test_len = len(self.test_dataset)

	def tokenizer(self, seq):
		return list(seq)

	def get_vocab_vectors(self):
		return self.SEQ.vocab.vectors

	def next_batch(self, gpu=False):
		batch = next(self.train_iter)

		if gpu:
			return (batch.seq[0].cuda(), batch.seq[1].cuda()), batch.label.cuda()

		return batch.seq, batch.label

	def next_validation_batch(self, gpu=False):
		batch = next(self.valid_iter)

		if gpu:
			return (batch.seq[0].cuda(), batch.seq[1].cuda()), batch.label.cuda()
		return batch.seq, batch.label

	def next_test_batch(self, gpu=False):
		batch = next(self.test_iter)

		if gpu:
			return (batch.seq[0].cuda(), batch.seq[1].cuda()), batch.label.cuda()
		return batch.seq, batch.label

	def idx2sequence(self, idxs):
		return ' '.join([self.SEQ.vocab.itos[i] for i in idxs])

	def idx2label(self, idx):
		return self.LABEL.vocab.itos[idx]
