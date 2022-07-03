import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from attention import SelfAttention


class Conv_LSTM_selfattn(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, bidirectional, hidden_dim, dropout):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embedding_dim))
		self.lstm = nn.LSTM(n_filters, hidden_dim, bidirectional=bidirectional, dropout=dropout)
		self.attention = SelfAttention(hidden_dim)
		self.fc = nn.Linear(hidden_dim, 1)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# x [seq_len, batch_size]
		x = x.permute(1,0)
		# x [batch_size, seq_len]
		embedded = self.embedding(x)
		# [batch_size, seq_len, emb_dim]
		embedded = embedded.unsqueeze(1)
		# [batch_size, 1, seq_len, emb_dim]
		conved = F.relu(self.conv(embedded)).squeeze(3)
		# conved = [batch_size, n_filters, seq_len - filter_size]
		x = conved.permute(2, 0, 1)
		# [seq_len - filter_size, batch_size, n_filters]
		output, (hidden, cell) = self.lstm(x)
		# output = [seq_len - filter_size, batch_size, hid_dim * num_dirs]
		output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
		# output [seq_len, batch_size, hid_dim]
		output = output.permute(1, 0, 2)
		# output [batch_size, seq_len, hid_dim]
		new_embedded, weights = self.attention(output)
		# new_embed = [batch_size, hid-dim]
		# weights = [batch_size, seq_len]
		new_embedded = self.dropout(new_embedded)
		return self.fc(new_embedded)