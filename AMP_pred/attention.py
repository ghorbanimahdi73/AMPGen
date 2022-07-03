# attention models
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
	'''
	Scaled dot product attention from "Attention is all you need"
	Comput ethe dot product of the query with all keys, divide by sqrt(dim)
	and apply softmax function to obtain the weights of on the values

	Args: dim, mask
		dim: dimension of attention
		mask: (torch.Tensor) tensor obtaining indices to be masked

	inputs:	query, key, value, mask
		query: [batch, q_len, d_model]: tensor containing projection vectors for decoder
		key: [batch, k_len, d_model]: tesnor containing proejction vector for encoder
		value: [batch, v_len, d_model]: tesnor containing feature of encoded input sequence.
		mask: tensor containing indices to be masked

	Returns: context, attn
		context: context vector from attention mechanism
		attn: tensor containing attention (alignment) from the encoder outputs.
	'''
	def __init__(self, dim:int):
		super(ScaledDotProductAttention, self).__init__()
		self.sqrt_dim = np.sqrt(dim)

	def forward(self, query, key, value, mask=None):
		score = torch.bmm(query, key.transpose(1,2)) / self.sqrt_dim

		if mask is not None:
			score.masked_fill_(mask.view(score.size()), -float('Inf'))

		attn = F.softmax(score, -1)
		context = torch.bmm(attn, value)
		return context, attn


class DotProductAttention(nn.Module):
	'''
	Compute the dot products of query with all values and apply a softmax function to obtain weights on the values.
	'''
	def __init__(self, hidden_dim):
		super(DotProductAttention, self).__init__()

	def forward(self, query, value):
		batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

		score = torch.bmm(query, value.transpose(1,2))
		attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
		context = torch.bmm(attn, value)

		return context, attn


class BahdanauAttention(nn.Module):
	'''
	Addition (Bahdanau) attention on the outupt efatures from decoder.
	Additive attention proposed in "neural machine translation by jointly learning to align and Translate"

	args:
		hidden_dim: dimension of hidden state vector

	Inputs:
		query: [batch_size, q_len, hid_dim] output features from decoder
		value: [batch_size, v_len, hid_dim] features of the encoded input sequence

	Returns: 
		context: tensor containing the context vector from attention 
		attn: tensor containing the alignment from the encoder outputs.
	'''
	def __init__(self, hidden_dim):
		super(BahdanauAttention, self).__init__()
		self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
		self.score_proj = nn.Linear(hidden_dim, 1)

	def forward(self, query, key, value):
		score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
		attn = F.softmax(score, dim=-1)
		context = torch.bmm(attn.unsqueeze(1), value)
		return context, attn



class LuongAttention(nn.Module):
	def __init__(self, hidden_size_enc, hidden_size_dec, use_cuda=True, method='general'):
		super().__init__()
		# hidden_size_enc -> h_dim
		# hidden_size_dec -> z_dim

		self.hidden_size_enc = hidden_size_enc
		self.hidden_size_dec = hidden_size_dec
		self.use_cuda = use_cuda
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, 'is not appropriate!')
		if self.method == 'general': # W vector
			self.general_weights = torch.nn.Parameter(torch.randn(hidden_size_dec, hidden_size_enc))
		elif self.method == 'concat':
			self.general_weights = torch.nn.Parameter(torch.randn(hidden_size_dec, hidden_size_enc))
			self.v = torch.nn.Parameter(torch.randn(hidden_size_dec, hidden_size_enc))

		# general_weights -> [hidden_size_dec, hidden_size_enc]

	def forward(self, 
				encoder_outputs,
				decoder_outputs,
				enc_mask=None):

		# general_weights -> [dec_hid_dim, enc_hid_dim]
		# encoder_outputs -> [seq_len, batch_size, enc_hid_dim]
		# decoder_outputs -> [1, batch_size, dec_hid_dim]
		dec_len = decoder_outputs.size(0) # 1
		enc_len = encoder_outputs.size(0) # seq_len

		decoder_outputs = torch.transpose(decoder_outputs, 0, 1)
		# -> [batch_size, 1, dec_hid_dim]
		encoder_outputs = encoder_outputs.permute(1,2,0)
		# -> [batch_size, enc_hid_dim, seq_len]

		score = torch.bmm(decoder_outputs @ self.general_weights, encoder_outputs)
		# -> [batch_size, 1, seq_len]

		if enc_mask is not None:
			enc_mask = enc_mask.unsqueeze(1)
			enc_mask = torch.transpose(enc_mask, 0, 2)
			score = score.masked_fill(enc_mask==0, -1e12)

		weights_flat = F.softmax(score.view(-1, enc_len), dim=1)
		# [batch_size, seq_len]
		weights = weights_flat.view(-1, dec_len, enc_len)
		# [batch_size, 1, seq_len]
		attention_vector = torch.bmm(weights, encoder_outputs.permute(0,2,1))
		# [batch_size, 1, enc_hid_dim]
		attention_vector = attention_vector.permute(1,0,2)
		# [1, batch_size, enc_hid_dim]
		return attention_vector, weights.view(-1, enc_len)
		# attention_vector -> [1, batch_size, enc_hid_dim]
		# weights -> [batch_size, seq_len]





class MultiHeadAttention(nn.Module):
	'''
	multi-head attention proposed in 'attention is all you need'
	Instead of performing a single attention function with d_model-dim keys, values and queries,
	projects the queries, keys and values h times with different, learned linear projections to d_head dims.
	These are concatenated and once again projected, resulting in the final values.
	Multi-head attention allows the mode lto joinly attend to information from different representation subspaces
	at different positions.

	MultiHead(Q,K,V) = Concat(head_1, ..., head_h) . W_o
		where head_i = Attention(Q . W_q, K . W_k, V . W_v)

	Args:
		d_model (int): dimension of keys/values/queries
		num_heads (int): number of attention heads

	Inputs: query, key, value, mask
		query: [batch, q_len, d_model]
		key: [batch, k_len, d_model]
		value: [batch, v_len, d_model]
		mask: [batch, v_len, d_model]

	Returns:
		output: [batch, output_len, dimensions] attended output features.
		attn: [batch * num_heads, v_len] tensor containing the attention from encoder outputs.
	'''
	def __init__(self, d_model=64, num_heads=4):
		super(MultiHeadAttention, self).__init__()
		assert d_model % num_heads ==0, 'd_model % num_heads should be zero.'

		self.d_head = int(d_model/num_heads)
		self.num_heads = num_heads
		self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
		self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
		self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
		self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

	def forward(self, 
				query,
				key,
				value,
				mask):

		batch_size = value.size(0)
		query = self.key_proj(query).view(batch_size, -1, self.num_heads, self.d_head) # B*Q_LEN*N*D
		key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head) # B*K_LEN*N*D
		value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head) # B*V_LEN*N*D

		query = query.permute(2,0,1,3).contiguous().view(batch_size * self.num_heads, -1, self.d_head) # BN*Q_LEN*D
		key = key.permute(2,0,1,3).contiguous().view(batch_size * self.num_heads, -1, self.d_head) # BN*K_LEN*D
		value = value.permute(2,0,1,3).contiguous().view(batch_size*self.num_heads, -1, self.d_head) # BN*V_LEN*D

		if mask is not None:
			mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # B*N*Q_LEN*K_LEN

		context, attn = self.scaled_dot_attn(query, key, value, mask)
		
		context = context.view(self.num_heads, batch_size, -1, self.d_head)
		context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head) # B*T*ND
		return context, attn


class SelfAttention(nn.Module):
	'''
	self attention mechanism

	inputs:


	Outputs:

		
	'''
	def __init__(self, hidden_dim):
		super().__init__()
		self.projection = nn.Sequential(
				nn.Linear(hidden_dim, 64),
				nn.ReLU(True),
				nn.Linear(64,1))

	def forward(self, encoder_outputs):
		# encoder outputs: [batch_size, seq_len, hid_dim]
		energy = self.projection(encoder_outputs)
		# energy [batch_size, seq_len, 1]
		weights = F.softmax(energy.squeeze(-1), dim=1)
		# weights [batch_size, seq_len]
		outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
		return outputs, weights









