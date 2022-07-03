import sys, os, types
import json

class Bunch(dict):
	def __init__(self, *args, **kwds):
		super(Bunch, self).__init__(*args, **kwds)
		self.__dict__ = self


PAD_IDX = 1
n_vocab = 24

WAE_MMD = Bunch(
		sigma=7.0,
		kernel='gaussian',
		rf_dim=500,
		rf_resample=False)



