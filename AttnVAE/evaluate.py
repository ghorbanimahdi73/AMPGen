import numpy as np
from Bleu import Bleu
from Bleu import SelfBleu
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--get-blue', action='store_true', default=False, help='calculate bleu score')
parser.add_argument('--self-bleu', action='store_true', default=False, help='calculate self bleu score over the generated sequenes')
parser.add_argument('--real-amps', type=str, default='amp_seqs.csv', help='real amps dataset CSV file')
parser.add_argument('--test-amps', type=str, default='seqs.csv', help='test amps dataset CSV file')
parser.add_argument('--num-samples', type=int, default=5000, help='number of examples in test')
#parser.add_argument('--ngram', type=int, default=2, help='the n-gram to compue')
parser.add_argument('--save-file', type=str, default='bleu', help='save file name')


args = parser.parse_args()

if not os.path.exists(args.save_file):
	os.makedirs(args.save_file)


if __name__ == '__main__':

	print('number of cpus', os.cpu_count())
	BLEU = Bleu(real_data=args.real_amps, test_data=args.test_amps, sample_size=args.num_samples, gram=2)
	bleu2 = BLEU.get_bleu_fast()

	print('Bleu-2-gram: {}'.format(bleu2))
	
	BLEU = Bleu(real_data=args.real_amps, test_data=args.test_amps, sample_size=args.num_samples, gram=3)
	bleu3 = BLEU.get_bleu_fast()
	print('Bleu-3-gram: {}'.format(bleu3))


	BLEU = Bleu(real_data=args.real_amps, test_data=args.test_amps, sample_size=args.num_samples, gram=4)
	bleu4 = BLEU.get_bleu_fast()
	print('Bleu-4-gram: {}'.format(bleu4))


	BLEU = Bleu(real_data=args.real_amps, test_data=args.test_amps, sample_size=args.num_samples, gram=5)
	bleu5 = BLEU.get_bleu_fast()
	print('Bleu-5-gram: {}'.format(bleu5))
	
	bleu = np.array([bleu2, bleu3, bleu4, bleu5])
	np.save(args.save_file +'/bleu.npy', bleu)


	s_Bleu = SelfBleu(test_data=args.test_amps, sample_size=args.num_samples, gram=2)
	s_bleu2 = s_Bleu.get_bleu_fast()
	print('self-Bleu-2: {}'.format(s_bleu2))

	s_Bleu = SelfBleu(test_data=args.test_amps, sample_size=args.num_samples, gram=3)
	s_bleu3 = s_Bleu.get_bleu_fast()
	print('self-Bleu-3: {}'.format(s_bleu3))

	s_Bleu = SelfBleu(test_data=args.test_amps, sample_size=args.num_samples, gram=4)
	s_bleu4 = s_Bleu.get_bleu_fast()
	print('self-Bleu-4: {}'.format(s_bleu4))

	s_Bleu = SelfBleu(test_data=args.test_amps, sample_size=args.num_samples, gram=5)
	s_bleu5 = s_Bleu.get_bleu_fast()
	print('self-Bleu-5: {}'.format(s_bleu5))

	self_bleu = np.array([s_bleu2, s_bleu3, s_bleu4, s_bleu5])
	np.save(args.save_file + '/self_bleu.npy', self_bleu)





