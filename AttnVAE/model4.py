import numpy as np
from torchtext.legacy import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention import LuongAttention, BahdanauAttention
from losses import *


class RNN_VAE_attn(nn.Module):
	def __init__(self, n_vocab, h_dim, z_dim, attn_dim, dropout_prob=0.1, p_word_dropout=0.3,
				unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_seq_len=30, pretrained_embeddings=None,
				freeze_embeddings=False, gpu=False, bidirectional=True, fix_length=False):

		super(RNN_VAE_attn, self).__init__()

		self.UNK_IDX = unk_idx
		self.PAD_IDX = pad_idx
		self.START_IDX = start_idx
		self.EOS_IDX = eos_idx
		self.MAX_SEQ_LEN = max_seq_len

		self.n_vocab = n_vocab
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.p_word_dropout = p_word_dropout
		self.dropout_prob = dropout_prob
		self.gpu = gpu
		self.attn_dim = attn_dim 
		self.bidirectional = bidirectional
		self.factor = 2 if bidirectional else 1
		self.fix_length = fix_length

		if pretrained_embeddings is None:
			self.emb_dim = h_dim
			self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX) # ignore the padding for embedding
		else:
			self.emb_dim = pretrained_embeddings.size(1)
			self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

			# set pretrained embeddings
			self.word_emb.weight.data.copy_(pretrained_embeddings)

			if freeze_embeddings:
				self.word_emb.weight.requires_grad = False


		self.encoder = nn.GRU(self.emb_dim, self.h_dim, bidirectional=self.bidirectional) # emb_dim -> h_dim
		self.q_mu  = nn.Linear(self.factor * h_dim, z_dim ) # 2 * h_dim ->  z_dim	
		self.q_logvar = nn.Linear(self.factor * h_dim, z_dim ) # 2 * h_dim -> z_dim
		self.z_to_h = nn.Linear(self.z_dim, self.h_dim) # z_dim -> h_dim

		self.attn = LuongAttention(self.h_dim * self.factor, self.h_dim)
		# outputs 1 ) attention_vector [1, batch_size, enc_hid_dim]
		# 		  2 ) weights [batch_size, seq_len]
		self.attn_q_logvar = nn.Linear(self.h_dim*self.factor, self.factor*self.attn_dim) # 2*h_dim -> 2*attn_dim
		# h_dim * 2 -> z_dim * 2
		self.decoder_fc = nn.Linear(self.h_dim + self.z_dim, n_vocab)
		# emb_dim + h_dim + z_dim -> n_vocab
		self.decoder_attn = nn.GRU(self.emb_dim + self.factor*self.attn_dim + self.z_dim, self.h_dim)
		# (emb_dim) + (2*z_dim+c_dim) + (2*attn_dim) -> 2*z_dim + c_dim
		self.dropout = nn.Dropout(self.p_word_dropout)

		if self.gpu:
			self.cuda()

	def forward_encoder(self, inputs, lengths):
		# inputs -> [seq_len, batch_size]
		mask = inputs != self.PAD_IDX # padded tokens will be zero and non-padded will be 1
		inputs = self.word_emb(inputs) # [seq_len, batch_size, emb_dim]
		return self.forward_encoder_embed(inputs, mask, lengths)

	def forward_encoder_embed(self, inputs, mask, lengths):
		# inputs [seq_len, batch_size, emb_dim]

		packed_embedded = nn.utils.rnn.pack_padded_sequence(inputs, lengths.to('cpu'))
		packed_outputs, hidden = self.encoder(packed_embedded, None)

		# packed_outputs if packed sequence contains all hidden states
		# hidden is from the final non-padded elements in the batch

		if self.fix_length:
			output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=self.PAD_IDX, total_length=self.MAX_SEQ_LEN+2)
		else:	
			output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=self.PAD_IDX)

		# output is not non-packed sequence
		# outputs [seq_len, batch_size, num_dirs*h_dim]
		# hidden [2, batch_size, h_dim]

		if self.bidirectional:
			h = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=1)
		else:
			h = hidden
			h = h.view(-1, self.h_dim)

		# h [batch_size, h_dim * num_dirs]
		mu = self.q_mu(h) # [batch_size,  z_dim]
		logvar = self.q_logvar(h) # [batch_size, z_dim]
		return mu, logvar, output, mask

	def sample_z(self, mu, logvar): # sample from mu and logvar
		eps = Variable(torch.randn(self.z_dim))
		eps = eps.cuda() if self.gpu else eps
		return mu + torch.exp(logvar/2) * eps

	def sample_z_prior(self, batch_size): # sample from a normal distribution
		z = Variable(torch.randn(batch_size, self.z_dim))
		z = z.cuda() if self.gpu else z
		return z

	def sample_attn(self, attn_mu, attn_logvar):
		eps = Variable(torch.randn(self.factor * self.attn_dim))
		eps = eps.cuda() if self.gpu else eps
		return attn_mu + torch.exp(attn_logvar/2) * eps

	def sample_attn_prior(self, batch_size):
		attn_vec = Variable(torch.randn(batch_size, self.factor * self.attn_dim))
		attn_vec = attn_vec.cuda() if self.gpu else attn_vec
		return attn_vec

	def forward_decoder_attn(self, inputs, z, enc_h, mask=None, use_hmean=False, lengths=None):
		# inputs [seq_len, batch_size]
		# z -> [batch_size, z_dim ]
		# c -> [batch_size,1]
		# enc_h -> [seq_len, batch_size, h_dim * factor]
		# mask [seq_len, batch_size]
		# lengths -> [batch_size]

		dec_inputs = self.word_dropout(inputs)
		inputs_emb = self.word_emb(dec_inputs)
		inputs_emb = self.dropout(inputs_emb) # [seq_len, batch_size, emb_dim]

		#max_length = torch.max(lengths)
		n_vocab = 24
		batch_size = inputs.shape[1]

		z = z.unsqueeze(0) # [1, batch_size, z_dim]
		z_h = self.z_to_h(z) # [1,batch_size, h_dim]
		output_all = []

		input_len = inputs.shape[0]
		for i in range(inputs_emb.shape[0]):

			attn_vec, weights = self.attn(enc_h, z_h, mask)
			# attn_vec -> [1, batch_size, h_dim*2]
			# weights -> [batch_size, seq_len]
			#h_source_mu = torch.mean(enc_h, dim=0)  # [batch_size, h_dim*factor]

			attn_mu = attn_vec
			attn_logvar = self.attn_q_logvar(attn_vec) # [batch_size, attn_dim *2]
			attn_logvar = F.tanh(attn_logvar)
			#attn_logvar = F.tanh(attn_logvar)  # tanh activation

			sample_attn_vec = self.sample_attn(attn_mu, attn_logvar) # [1, batch_size, 2*attn_dim]
			new_attn_vec = sample_attn_vec

			new_inputs_emb = torch.cat([inputs_emb[i].unsqueeze(0), z, new_attn_vec], dim=2)
			# [1, batch_size, emb_dim + z_dim + 2*attn_dim ]

			outputs, hidden = self.decoder_attn(new_inputs_emb, z_h)
			# [1, batch_size, h_dim]
			seq_len, batch_size, _ = outputs.size()
			outputs = outputs.view(seq_len*batch_size, -1) # [batch_size, z_dim]

			outputs_cat = torch.cat([outputs, z.squeeze()], dim=1) # skip connection from z to word prediction
			y = self.decoder_fc(outputs_cat) # [batch_size, n_vocab]
			y = y.view(seq_len, batch_size, self.n_vocab) # [1, batch_size, n_vocab]

			output_all.append(y)

			z_h = hidden # renew the hidden state
			if i == 0:
				if use_hmean:
					attn_kl_loss = calc_kl_loss(attn_mu - h_source_mu, attn_logvar)
					attn_wae_mmd_loss = wae_mmd_gaussianprior(sample_attn_vec.squeeze(), method='full_kernel')
				else:
					attn_kl_loss = calc_kl_loss(attn_mu, attn_logvar)
					attn_wae_mmd_loss = wae_mmd_gaussianprior(sample_attn_vec.squeeze(), method='full_kernel')
			else:
				if use_hmean:
					attn_kl_loss += calc_kl_loss(attn_mu - h_source_mu, attn_logvar)
					attn_wae_mmd_loss += wae_mmd_gaussianprior(sample_attn_vec.squeeze(), method='full_kernel')
				else:
					attn_kl_loss += calc_kl_loss(attn_mu, attn_logvar)
					attn_wae_mmd_loss += wae_mmd_gaussianprior(sample_attn_vec.squeeze(), method='full_kernel')

		outputs = torch.cat(output_all, 0) # [seq_len, batch_size, n_vocab]
		return outputs, attn_kl_loss, attn_wae_mmd_loss

	def forward(self, sequence, labels, lengths):
		'''
		inputs:
			sequence: torch.Tensor, batch of sequences
					[seq_len, batch_size]
			labels: torch.Tensor, batch of labels
					[batch_size]
			lengths: torch.Tensor, lengths of sequences in the batch
					[batch_size]

		outputs:
		'''
		self.train()

		batch_size = sequence.size(1)
		pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, batch_size)
		pad_words = pad_words.cuda() if self.gpu else pad_words

		enc_inputs = sequence
		dec_inputs = sequence # [seq_len, batch_size]
		dec_targets = torch.cat([sequence[1:], pad_words], dim=0)
		# append a <pad> to every sequence 
		mu, logvar, output_h, mask = self.forward_encoder(enc_inputs, lengths)
		# mu, logvar -> [batch_size, z_dim ]
		# output_h -> [seq_len, batch_size, h_dim * factor]
		# mask -> [seq_len, batch_size]

		# sample from encoder
		z = self.sample_z(mu, logvar) # [batch_size, z_dim]

		use_hmean = False
		# forward the decoder / using attention
		y, kl_loss_attn, attn_wae_mmd_loss = self.forward_decoder_attn(dec_inputs, z, output_h, mask, use_hmean, lengths)

		recons_loss = calc_cross_ent(dec_targets, y)
		kl_loss = calc_kl_loss(mu, logvar)
		wae_mmd_loss = wae_mmd_gaussianprior(z, method='full_kernel')

		return recons_loss, kl_loss, kl_loss_attn, attn_wae_mmd_loss, wae_mmd_loss

	def sample_sequence_attn(self, z, attn_mu=None, attn_logvar=None, temp=1, attn_prior=True, usesam=True, top_p=0., top_k=0, min_length=5):

		self.eval()
		seq = torch.LongTensor([self.START_IDX])
		seq = seq.cuda() if self.gpu else seq
		seq = Variable(seq)

		z = z.view(1, -1)
		z = z.view(1, 1, -1)
		z_h = self.z_to_h(z)

		if not isinstance(z, Variable):
			z = Variable(z)
			z_h = Variable(z_h) 

		outputs = []

		for i in range(self.MAX_SEQ_LEN):
			if attn_prior:
				sam_attn = self.sample_attn_prior(1)
				sam_attn = sam_attn.unsqueeze(0)
			else:
				sam_attn = self.sample_attn(attn_mu, attn_logvar) # only done during training
				sam_attn = sam_attn.unsqueeze(0) # only done during training

			emb = self.word_emb(seq).view(1,1,-1)

			output, hidden = self.decoder_attn(torch.cat([emb, z, sam_attn], dim=2), z_h)

			# linear transform to n_vocab
			seq_len, batch_size, _ = output.size()
			output = output.view(seq_len*batch_size, -1) # [batch_size, h_dim]

			output_cat = torch.cat([output, z.squeeze(0)], dim=1)

			y = self.decoder_fc(output_cat).view(-1)

			if len(outputs) < min_length:
				y[self.EOS_IDX] = -1e12

			y[self.PAD_IDX]   =  -1e12
			y[self.UNK_IDX]   =  -1e12
			y[self.START_IDX] =  -1e12

			if top_k>0 or top_p > 0:
				y = self.top_k_top_p_filtering(y, top_k=top_k, top_p=top_p, filter_value=-1e12)

			z_h = hidden

			y = F.softmax(y/temp, dim=0) 

			if usesam:
				idx = torch.multinomial(y,1)
			else:
				idx = torch.max(y,0)[1]

			seq = Variable(torch.LongTensor([int(idx)]))
			seq = seq.cuda() if self.gpu else seq

			idx = int(idx)

			if idx == self.EOS_IDX:
				break

			outputs.append(idx)

		self.train()
		return outputs

	@staticmethod
	def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-1e12):
		'''
		Filter a distribution of logits using top-k and/or top-p filtering
		args:
			logits: logits distributions shape [vocabulary size]
			top_k > 0 : keep only top k tokens with highest probability (top-p filtering)
			top_p > 0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		'''
		assert logits.dim() == 1, 'batch size of 1 must be used here'
		top_k = min(top_k, logits.size(-1)) # saftey check
		if top_k > 0:
			# Remove all tokens with a probability less than the last token of the top-k
			indices_to_remove = logits < torch.topk(logits, top_k)[0][...,-1,None]
			logits[indices_to_remove] = filter_value

		if top_p > 0.0:
			sorted_logits, sorted_indices = torch.sort(logits, descending=True)
			cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

			# Remove tokens with cumulative prob above a threshold
			sorted_indices_to_remove = cumulative_probs > top_p
			# Shift the indices to the right to keep also the first token above the threshold
			sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
			sorted_indices_to_remove[...,0] = 0

			indices_to_remove = sorted_indices[sorted_indices_to_remove]
			logits[indices_to_remove] = filter_value
		return logits

	def word_dropout(self, inputs):
		'''
		Do word dropout with prob: p_word_dropout, set word to <UNK>
		'''

		if isinstance(inputs, Variable):
			data = inputs.data.clone()
		else:
			data = inputs.clone()

		# sample masks: elems with val 1 will be set to <UNK>
		mask = torch.from_numpy(
			np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size())).astype('uint8'))

		if self.gpu:
			mask = mask.cuda()

		# set to UNK
		data[mask] = self.UNK_IDX
		return Variable(data)
