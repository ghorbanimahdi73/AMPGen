import argparse

def buildParser():

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
	parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
	parser.add_argument('--seed', type=int, default=7, help='random seed')
	parser.add_argument('--epochs', type=int, default=50, help='number of epochs of training')
	parser.add_argument('--h-dim', type=int, default=200, help='hidden dimension of the model')
	parser.add_argument('--c-dim', type=int, default=1, help='number of classes - 1')
	parser.add_argument('--z-dim', type=int, default=16, help='dimension of latent space')
	parser.add_argument('--max-weight', type=float, default=0.03, help='maximum weight')
	parser.add_argument('--kl-attn-weight', type=float, default=1.0, help='weight for KL-deivergence of attention')
	parser.add_argument('--attn-prior', action='store_true', default=False, help='Whether to use attention prior')
	parser.add_argument('--cnew-dim', type=int, default=200, help='dimension of class variable in the decoding part')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--save-file', type=str, default='model_1', help='Name of the final model to save')
	parser.add_argument('--gpu', action='store_true', default=False, help='Whether to use gpu or not')
	parser.add_argument('--dropout-prob', type=float, default=0.1, help='Dropout rate')
	parser.add_argument('--word-dropout', type=float, default=0.3, help='Word dropout')
	parser.add_argument('--use-MMD', action='store_true', default=False, help='Whether to use WAE with MMD loss')
	parser.add_argument('--save-checkpoints', action='store_true', default=False, help='Whether to save checkpionts during training')
	parser.add_argument('--anneal-type', type=str, default='monotonic', help='type of the annealing schedule [monotonic, cyclic]')
	parser.add_argument('--use-hmean', action='store_true', default=False, help='whether to subtract the mean from the attention vector in decoder')
	parser.add_argument('--bidirectional', action='store_true', default=False, help='Whether to use bidirectional Encoder GRU')
	parser.add_argument('--fix-length', action='store_true', default=False, help='Wheter to feed fixed length during training')
	return parser