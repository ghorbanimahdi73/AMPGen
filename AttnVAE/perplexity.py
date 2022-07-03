import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
import torch
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-forward', action='store_true', default=False, help='train LSTM model for perplexity')
parser.add_argument('--train-backward', action='store_true', default=False, help='train LSTM on the generated data to compute perplexity for real data')
parser.add_argument('--f-perplexity', action='store_true', default=False, help='evaluate the trained model on the test data')
parser.add_argument('--b-perplexity', action='store_true', default=False, help='evaluate the backward perplexity')
parser.add_argument('--test-file', type=str, default='seqs.csv', help='real amps dataset CSV file')
parser.add_argument('--csv-name',type=str, default='seqs_top_k_5.csv', help='csv file name')

args = parser.parse_args()

class AMP_dataset:
	def __init__(self, batch_size=32, file_train='amp_train.csv', file_valid='amp_valid.csv', fixlen=None, device='cuda'):

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

		self.train_iter, self.valid_iter = data.BucketIterator.splits(
				(self.train_dataset, self.valid_dataset), batch_size=batch_size, device=device,
				sort_within_batch=True, sort_key=lambda x:len(x.seq), shuffle=True, repeat=True)

		self.train_iter = iter(self.train_iter)
		self.valid_iter = iter(self.valid_iter)

		self.train_len = len(self.train_dataset)
		self.valid_len = len(self.valid_dataset)

	def tokenizer(self, seq):
		return list(seq)

	def get_vocab_vectors(self):
		return self.SEQ.vocab.vectors

	def next_batch(self, gpu=False):
		batch = next(self.train_iter)

		if gpu:
			return (batch.seq[0].cuda(),batch.seq[1].cuda()), batch.label.cuda()

		return batch.seq, batch.label

	def next_validation_batch(self, gpu=False):
		batch = next(self.valid_iter)

		if gpu:
			return (batch.seq[0].cuda(), batch.seq[1].cuda()), batch.label.cuda()
		return batch.seq, batch.label

	def idx2sequence(self, idxs):
		return ' '.join([self.SEQ.vocab.itos[i] for i in idxs])

	def idx2label(self, idx):
		return self.LABEL.vocab.itos[idx]




class RNNLM(nn.Module):
	def __init__(self, vocab_size=24, word_dim=100, h_dim=100, num_layers=1, dropout=0.3):
		super(RNNLM, self).__init__()

		self.h_dim = h_dim
		self.num_layers = num_layers
		self.word_vecs = nn.Embedding(vocab_size, word_dim)
		self.dropout = nn.Dropout(dropout)
		self.rnn = nn.LSTM(word_dim, h_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
		self.vocab_linear = nn.Sequential(nn.Dropout(dropout),
										  nn.Linear(h_dim, vocab_size),
										  nn.LogSoftmax(dim=-1))

	def forward(self, sent):
		word_vecs = self.dropout(self.word_vecs(sent[:,:-1]))
		h, _ = self.rnn(word_vecs)
		preds = self.vocab_linear(h)
		return preds


# parameters for the LSTM language model
word_dim = 128
h_dim = 128
dropout=0.3
batch_size = 32

if args.train_forward or args.train_backward:
	train = True
else:
	train = False


if train:
	if args.train_forward:
		dataset = AMP_dataset(batch_size=batch_size, file_train='amp_train.csv', file_valid='amp_valid.csv')

	if args.train_backward:
		dataset = AMP_dataset(batch_size=batch_size, file_train=args.test_file+ '/' +args.csv_name, file_valid='amp_valid.csv')

	model = RNNLM(word_dim=word_dim, h_dim=h_dim, dropout=dropout)
	for param in model.parameters():
		param.data.uniform_(-0.1, 0.1)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.NLLLoss()
	model.train()

	model.cuda()
	best_val_ppl = 1e5
	epoch = 0

	validation_ppl = []
	training_ppl = []
	while epoch < 100:
		epoch += 1
		train_nll = 0
		num_sents = 0
		num_words = 0
		b = 0

		for batch in range(dataset.train_len//batch_size):
			(inputs, lengths), labels = dataset.next_batch()
			inputs = inputs.t()

			b += 1
			optimizer.zero_grad()
			preds = model(inputs)

			nll = sum([criterion(preds[:,l], inputs[:,l+1]) for l in range(lengths[0]-1)])
			train_nll += nll.item() * batch_size
			nll.backward()

			torch.nn.utils.clip_grad_norm(model.parameters(), 10.)
			optimizer.step()

			num_sents += batch_size
			num_words += batch_size * lengths[0]

			if b % 100 == 0:
				print('Epoch: %d, TrainPPL: %.2f' % 
					(epoch, np.exp(train_nll/num_words.cpu())))

		training_ppl.append(np.exp(train_nll/num_words.cpu()))

		model.eval()
		num_sents = 0
		num_words = 0
		total_nll = 0.
		for valid_data in range(dataset.valid_len//batch_size):
			(inputs, lengths), labels = dataset.next_validation_batch(gpu=True)
			inputs = inputs.t()

			num_words += batch_size*lengths[0]
			num_sents += batch_size

			preds = model(inputs)

			nll = sum([criterion(preds[:,l], inputs[:,l+1]) for l in range(lengths[0]-1)])
			total_nll += nll.item() * batch_size
		ppl = np.exp(total_nll/num_words.cpu())
		print('validation perplexity: ', ppl)
		validation_ppl.append(ppl)
		model.train()

	training_ppl = np.array(training_ppl)
	validation_ppl = np.array(validation_ppl)

	if args.train_forward:
		np.save('training_ppl_forward.npy', training_ppl)
		np.save('validation_ppl_forward.npy', validation_ppl)
		file_name = 'forward_ppl.bin'
	elif args.train_backward:
		np.save(args.test_file+'/training_ppl_backward.npy', training_ppl)
		np.save(args.test_file+'/validation_ppl_backward.npy', validation_ppl)
		file_name = args.test_file+'/backward_ppl.bin'

	torch.save(model.state_dict(), file_name)


if args.f_perplexity:
	# calculate forward perplexity
	
	dataset = AMP_dataset(batch_size=batch_size, file_train=args.test_file + '/' + args.csv_name, file_valid=args.test_file+'/'+args.csv_name)
	model = RNNLM(word_dim=128, h_dim=128, dropout=0)
	model.load_state_dict(torch.load('forward_ppl.bin', map_location=lambda storage, loc:storage), strict=False)
	criterion = nn.NLLLoss()
	model.eval()
	model.cuda()
	with torch.no_grad():
		num_sents = 0
		num_words = 0
		total_nll = 0.

		for valid_data in range(dataset.train_len//batch_size):
			(inputs, lengths), labels = dataset.next_batch(gpu=True)
			inputs = inputs.t()

			num_words += batch_size*lengths[0]
			num_sents += batch_size

			preds = model(inputs)

			nll = sum([criterion(preds[:,l], inputs[:,l+1]) for l in range(lengths[0]-1)])
			total_nll += nll.item() * batch_size

	ppl = np.exp(total_nll/num_words.cpu())
	print('validation perplexity: ', ppl.item())

	with open(args.test_file+'/f_ppl.txt', 'w') as f:
		f.write('model perplexity: {}'.format(ppl.item()))

if args.b_perplexity:
	# calculating backward perplexity

	dataset = AMP_dataset(batch_size=batch_size, file_train=args.test_file+'/'+args.csv_name, file_valid=args.test_file+'/'+args.csv_name)
	model = RNNLM(word_dim=128, h_dim=128, dropout=0)
	model.load_state_dict(torch.load('backward_ppl.bin', map_location=lambda storage, loc:storage), strict=False)
	criterion = nn.NLLLoss()
	model.eval()
	model.cuda()
	with torch.no_grad():
		num_sents = 0
		num_words = 0
		total_nll = 0.

		for valid_data in range(dataset.train_len//batch_size):
			(inputs, lengths), labels = dataset.next_batch(gpu=True)
			inputs = inputs.t()

			num_words += batch_size*lengths[0]
			num_sents += batch_size

			preds = model(inputs)

			nll = sum([criterion(preds[:,l], inputs[:,l+1]) for l in range(lengths[0]-1)])
			total_nll += nll.item() * batch_size

	ppl = np.exp(total_nll/num_words.cpu())
	print('validation perplexity', ppl.item())
	with open(args.test_file+'/b_ppl.txt', 'w') as f:
		f.write('model perplexity: {}'.format(ppl.item()))
