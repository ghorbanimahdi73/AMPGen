import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from AMP_dataset import AMP_dataset
from model import Conv_LSTM_selfattn
from args_classification import buildParser
import warnings
import os
import json


args = buildParser().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using GPU: ', torch.cuda.is_available())
# ignore deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

dataset = AMP_dataset(batch_size=args.batch_size)

input_dim = len(dataset.SEQ.vocab)
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
save_path =args.save_path

if not os.path.exists(args.save_path):
	os.makedirs(args.save_path)

model = Conv_LSTM_selfattn(input_dim, embedding_dim, n_filters, filter_size,
		                  bidirectional, hidden_dim, dropout)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

if not os.path.exists(args.save_path):
	os.makedirs(args.save_path)

with open(args.save_path+'/train_args.txt','w') as f:
	json.dump(args.__dict__, f, indent=2)

print(args)


def count_params(model):
	'''
	count the number of parameters in the model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('number of parameters in the model: ', count_params(model))


def categorical_accuracy(preds, y):
	'''
	Returns accuracy per batch
	'''

	rounded_preds = torch.round(torch.sigmoid(preds))
	correct = (rounded_preds == y).float()
	acc = correct.sum() / len(correct)
	return acc

def train(model, iterator, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	model.to(device)	
	for batch in iterator:
		(inputs, lengths), labels = batch
		labels = labels.unsqueeze(0).t().to(device)
		optimizer.zero_grad()
		predictions = model(inputs.to(device))

		loss = criterion(predictions, labels)
		acc = categorical_accuracy(predictions, labels)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
	epoch_loss = 0
	epoch_acc = 0
	model.to(device)
	model.eval()

	with torch.no_grad():
		for batch in iterator:

			(inputs, lengths), labels = batch
			predictions = model(inputs.to(device))
			labels = labels.unsqueeze(1).to(device)

			loss = criterion(predictions, labels)
			acc = categorical_accuracy(predictions, labels)
			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def cross_val(num_folds=args.k_folds):

	folds_train_loss = []
	folds_train_acc = []
	folds_valid_acc = []
	folds_valid_loss = []
	for fold in range(num_folds):

		best_acc = 0
		early_stop = False
		iters_not_improved = 0
		dataset = AMP_dataset(batch_size=batch_size)

		model = Conv_LSTM_selfattn(input_dim, embedding_dim, n_filters, filter_size,
		                  bidirectional, hidden_dim, dropout)

		model.to(device)

		optimizer = optim.Adam(model.parameters())
		criterion = nn.BCEWithLogitsLoss()


		for epoch in range(n_epochs):
			fold_train_loss, fold_train_acc = train(model, dataset.train_iter, optimizer, criterion)
			fold_valid_loss, fold_valid_acc = evaluate(model, dataset.valid_iter, criterion)
			fold_loss_out = f'| Epoch: {epoch+1:02} | Train Loss: {fold_train_loss:.3f} | Train Acc: {fold_train_acc*100:.2f}% | Val. Loss: {fold_valid_loss:.3f} | Val. Acc: {fold_valid_acc*100:.2f}% |'
			print(fold_loss_out)



			if fold_valid_acc > best_acc:
				iters_not_improved = 0
				best_acc = fold_valid_loss
				snapshot_path = os.path.join(args.save_path, 'fold_'+str(fold)+'.pt')
				torch.save(model.state_dict(), snapshot_path)
			else:
				iters_not_improved += 1
				if iters_not_improved >= args.patience:
					early_stop = True
					print('early stopping patience reached ...')
					break

		folds_train_loss.append(fold_train_loss)
		folds_train_acc.append(folds_train_acc)
		folds_valid_loss.append(folds_valid_loss)
		folds_valid_acc.append(folds_valid_acc)

	return np.array(folds_train_acc), np.array(folds_valid_acc), np.array(folds_train_loss), np.array(folds_train_acc)


def train_model(n_epochs):
	best_acc = 0
	early_stop = False
	iters_not_improved = 0


	model = Conv_LSTM_selfattn(input_dim, embedding_dim, n_filters, filter_size,
		                  bidirectional, hidden_dim, dropout)
	

	optimizer = optim.Adam(model.parameters())
	criterion = nn.BCEWithLogitsLoss()
	dataset = AMP_dataset(batch_size=batch_size)

	for epoch in range(n_epochs):
		train_loss, train_acc = train(model, dataset.train_iter, optimizer, criterion)
		valid_loss, valid_acc = evaluate(model, dataset.valid_iter, criterion)
		loss_out = f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |'
		print(loss_out)
		with open(args.save_path+'/losses.txt','a') as f:
			f.write(loss_out)
			f.write('\n')

		if valid_acc > best_acc:
			
			iters_not_improved = 0
			best_acc = valid_acc
			snapshot_path = os.path.join(args.save_path, 'model_' + str(epoch)+ '.pt')
			torch.save(model.state_dict(), snapshot_path)
		else:
			iters_not_improved += 1
			if iters_not_improved >= args.patience:
				early_stop = True
				print('Early stopping patience reached ...')
				break

	torch.save(model.state_dict(), args.save_path+'/finalmodel.pt')

if args.cross_val:
	folds_train_acc, folds_valid_acc, folds_train_loss, folds_train_acc = cross_val(num_folds=args.k_folds)
	print('k-fold accuracy:',np.mean(folds_valid_acc))
	np.save(args.save_path+'/fold_valid_acc',folds_valid_acc)
	np.save(args.save_path+'/fold_train_acc',folds_train_acc)
	np.save(args.save_path+'/fold_valid_loss',folds_valid_loss)
	np.save(args.save_path+'/folds_train_loss',folds_train_loss)

if args.train:
	print('starting training ...')
	train_model(n_epochs=args.n_epochs)
