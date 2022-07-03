import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cfg

PAD_IDX = 1
n_vocab = 24

def calc_kl_loss(mu, logvar):
	kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))
	return kl_loss

def calc_cross_ent(dec_targets, y):
	recons_loss = F.cross_entropy(y.view(-1, n_vocab), dec_targets.view(-1), size_average=True, ignore_index=PAD_IDX)
	return recons_loss


def wae_mmd_gaussianprior(z, method='full_kernel'):
	''' taken from https://github.com/IBM/controlled-peptide-generation
	Compute MMD with samples from unit Gaussian
	'''
	z_prior = torch.rand_like(z) # shape and device [batch_size, emb_dim]
	MMD_params = cfg.WAE_MMD # parameters for MMD
	if method == 'full_kernel':
		mmd_kwargs = {'sigma': MMD_params.sigma, 'kernel': MMD_params.kernel}
		return mmd_full_kernel(z, z_prior, **mmd_kwargs)
	else:
		mmd_kwargs = {**MMD_params}
		return mmd_rf(z, z_prior, **mmd_kwargs)

def mmd_full_kernel(z1, z2, **mmd_kwargs):
	# z1, z2 [batch_size, emb_dim]
	K11 = compute_mmd_kernel(z1, z1, **mmd_kwargs)
	K22 = compute_mmd_kernel(z2, z2, **mmd_kwargs)
	K12 = compute_mmd_kernel(z1, z2, **mmd_kwargs)
	N = z1.size(0)
	assert N == z2.size(0), 'expected matching sizes z1 and z2'
	H = K11 + K22 - K12 * 2
	loss = 1. / (N * (N-1)) * H.sum()
	return loss

def mmd_rf(z1, z2, **mmd_kwargs):
	mu1 = compute_mmd_mean_rf(z1, **mmd_kwargs)
	mu2 = compute_mmd_mean_rf(z2, **mmd_kwargs)
	loss = ((mu1 - mu2) ** 2).sum()
	return loss

rf = {}

def compute_mmd_mean_rf(z, sigma, kernel, rf_dim, rf_resample=False):
	# random features approx of gaussian kernel mmd
	global rf
	if kernel == 'gaussian':
		if not kernel in rf or rf_resample:

			rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
			rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)
			rf['gaussia'] (rf_w, rf_b)
		else:
			rf_w, rf_b = rf['gaussian']
			assert rf_w.shape == (z.shape[1], rf_dim), 'not expecting z dim or rf_dim to change'
		z_rf = compute_mmd_mean_rf(z, rf_w, rf_b, sigma, rf_dim)
	else:
		raise ValueError('todo implement rf for kernel ' + kernel)
	mu_rf = z_rf.mean(0, keepdim=False)
	return mu_rf

def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
	z_emb = (z @ rf_w) / sigma + rf_b
	z_emb = torch.cos(z_emb) * (2./rf_dim) ** 0.5
	return z_emb

def compute_mmd_kernel(x, y, sigma, kernel):
	x_i = x.unsqueeze(1) # [batch_size, 1, emb_dim]
	y_j = y.unsqueeze(0) # [1, batch_size, emb_dim]
	xmy = ((x_i - y_j)**2).sum(2)
	if kernel == 'gaussian':
		K = torch.exp(-xmy / sigma ** 2)
	elif kernel == 'laplace':
		K = torch.exp(-torch.sqrt(xmy + (sigma ** 2)))
	elif kernel == 'energy':
		K = torch.pow(xmy + (sigma ** 2), -.25)
	return K
