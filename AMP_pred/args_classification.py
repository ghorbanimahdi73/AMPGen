import argparse

def buildParser():

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
	parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dimension')
	parser.add_argument('--n-filters', type=int, default=64, help='number of filters')
	parser.add_argument('--filter-size', type=int, default=3, help='Filter size for convolution')
	parser.add_argument('--bidirectional', action='store_true', default=False, help='wheter to use bidirectional RNN')
	parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
	parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
	parser.add_argument('--early-stop', action='store_true', default=False, help='Whether to use early stopping')
	parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs of training')
	parser.add_argument('--attn-type', type=str, default='SelfAttention', help='Type of attention to use')
	parser.add_argument('--train', action='store_true', help='Whether training or evaulation mode')
	parser.add_argument('--cross-val', action='store_true', help='Whether to perform cross-validation')
	parser.add_argument('--k-folds', type=int, default=10, help='number of folds for cross-validation')
	parser.add_argument('--save-path', type=str, default='model_1', help='directory to save files')
	return parser
