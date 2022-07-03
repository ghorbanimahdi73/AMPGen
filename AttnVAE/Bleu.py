import os
from multiprocessing import Pool

import nltk
from  nltk.translate.bleu_score import SmoothingFunction
from abc import abstractmethod
import pandas as pd
from tqdm import tqdm

class Metrics:
	def __init__(self):
		self.name = 'Metrics'

	def get_name(self):
		return self.name

	def set_name(self, name):
		self.name = name

	@abstractmethod
	def get_score(self):
		pass


class Bleu(Metrics):
	def __init__(self, test_data='', real_data='', gram=3, sample_size=5000):
		super().__init__()
		self.name = 'Bleu'
		self.test_data = test_data
		self.real_data = real_data
		self.gram = gram
		self.sample_size = sample_size
		self.reference = None
		self.is_first = True

	def get_name(self):
		return self.name

	def get_score(self, is_fast=True, ignore=False):
		if ignore:
			return 0

		if self.is_first:
			self.get_reference()
			self.is_first = False
		if is_fast:
			return self.get_bleu_fast()
		return self.get_bleu_parallel()

	def get_reference(self):
		if self.reference is None:
			reference = list()
			real_data_df = pd.read_csv(self.real_data)
			for i in range(len(real_data_df)):
				reference.append(' '.join(real_data_df['sequence'][i]).split())

			self.reference = reference
			return reference
		else:
			return self.reference

	def get_bleu(self):
		ngram = self.gram
		bleu = list()
		reference = self.get_reference()
		weight = tuple((1./ngram for _ in range(ngram)))
		test_df = pd.read_csv(self.test_data)

		for i in range(self.sample_size):
			hypothesis = ' '.join(test_df['sequence'][i].split())
			bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
					    smoothing_function=SmoothingFunction().method1))

		return sum(bleu) / len(bleu)

	def calc_bleu(self, reference, hypothesis, weight):
		return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, 
			              smoothing_function=SmoothingFunction().method1)


	def get_bleu_fast(self):
		reference = self.get_reference()

		return self.get_bleu_parallel(reference=reference)

	def get_bleu_parallel(self, reference=None):
		ngram = self.gram
		if reference is None:
			reference = self.get_reference()
		weight = tuple((1./ngram for _ in range(ngram)))
		pool = Pool(os.cpu_count())
		result = list()

		test_df = pd.read_csv(self.test_data)
		for i in tqdm(range(self.sample_size)):
			hypothesis =  ' '.join(test_df['sequence'][i].split())

			result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis, weight)))

		score = 0.0
		cnt = 0
		for i in result:
			score += i.get()
			cnt += 1
		pool.close()
		pool.join()
		return score / cnt


class SelfBleu(Metrics):
	def __init__(self, test_data='', gram=3, sample_size=5000):
		super().__init__()
		self.name = 'Self-Bleu'
		self.test_data = test_data
		self.gram = gram
		self.sample_size = sample_size
		self.reference = None
		self.is_first = True

	def get_name(self):
		return self.name 

	def get_score(self, is_fast=True, ignore=False):
		if ignore:
			return 0
		if self.is_first:
			self.get_reference()
			self.is_first = False
		if is_fast:
			return self.get_bleu_fast()
		return self.get_bleu_parallel()

	def get_reference(self):
		if self.reference is None:
			reference = list()
			real_data_df = pd.read_csv(self.test_data)
			for i in range(len(real_data_df)):
				reference.append(' '.join(real_data_df['sequence'][i]).split())

			return reference
		else:
			return self.reference

	def get_bleu(self):
		ngram = self.gram
		bleu = list()
		reference = self.get_reference()
		weight = tuple((1./ngram for _ in range(ngram)))	
		test_df = pd.read_csv(self.test_data)
		for i in range(self.sample_size):

			hypothesis = ' '.join(test_df['sequence'][i].split())
			bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
					  smoothing_function=SmoothingFunction().method1))

		return sum(bleu) / len(bleu)

	def calc_bleu(self, reference, hypothesis, weight):
		return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
			            smoothing_function=SmoothingFunction().method1)

	def get_bleu_fast(self):
		reference = self.get_reference()

		reference = reference[0:self.sample_size]
		return self.get_bleu_parallel(reference=reference)

	def get_bleu_parallel(self, reference=None):
		ngram = self.gram
		if reference is None:
			reference = self.get_reference()

		weight = tuple((1./ngram for _ in range(ngram)))
		pool = Pool(os.cpu_count())	
		result = list()
		sentence_sum = len(reference)
		for index in tqdm(range(self.sample_size)):
			hypothesis = reference[index]
			other = reference[:index] + reference[index+1:]
			result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

		score = 0.0
		cnt = 0
		for i in result:
			score += i.get()
			cnt += 1
		pool.close()
		pool.join()
		return score / cnt


class UniqueGram(Metrics):
	def __init__(self, test_text='', gram=3):
		super().__init__()
		self.name = 'UniqueGram'
		self.test_data = test_text
		self.gram = gram
		self.sample_size = 500
		self.reference = None
		self.is_first = True

	def get_name(self):
		return self.name

	def get_score(self, ignore=False):
		if ignore:
			return 0
		if self.is_first:
			self.get_reference()
			self.is_first = False
		return self.get_ng()

	def get_ng(self):
		document = self.get_reference()
		length = len(document)
		grams = list()
		for sentence in document:
			grams += self.get_gram(sentence)
		return len(set(grams))/length


#if __name__ == '__main__':
#	bleu = Bleu(test_data='seqs.csv', real_data='amp_seqs.csv', sample_size=1000)
#	score = bleu.get_bleu_fast()
#	print(score)

		