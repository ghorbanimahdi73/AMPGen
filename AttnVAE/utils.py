import os
import numpy as np



def interpolate(start_val, end_val, start_iter, end_iter, current_iter):
	if current_iter < start_iter:
		return start_val
	elif current_iter >= end_iter:
		return end_val
	else:
		return start_val + (end_val - start_val) * (current_iter - start_iter) / (end_iter - start_iter)

def anneal(start_val, end_val, start_iter, end_iter,  it):
	return interpolate(start_val, end_val, start_iter, end_iter, it)


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
	L = np.ones(n_epoch)*stop
	period = n_epoch/n_cycle
	step = (stop - start)/(period*ratio) # linear schedule

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i + c*period) < n_epoch):
			L[int(i+c*period)] = v
			v += step
			i += 1
	return L

def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
	L = np.ones(n_epoch)*stop
	period = n_epoch / n_cycle
	step = (stop - start) / (period*ratio)

	for c in range(n_cycle):

		v, i = start, 0
		while v <= stop:
			L[int(i+c*period)] = 1.0/(1.0+np.exp(-(v*12-6.)))
			v += step
			i += 1
	return L

def frange_cycle_cosine(start, stop, n_epoch, n_cycle, ratio=0.5):
	L = np.ones(n_epoch)*stop
	period = n_epoch/n_cycle
	step = (stop - start)/(period*ratio)

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop:
			L[int(i+c*period)] = 0.5 - 0.5 * math.cos(v*math.pi)
			v += step
			i += 1
	return L

def frange(start, stop, step, n_epoch, start_inc):
	L = np.ones(n_epoch) * start
	for i in range(start_inc, n_epoch):
		if L[i-1] < stop:
			L[i] = L[i-1] + step
		else:
			L[i] = stop
	return L

def monoton_sigmoid(start, stop, n_epoch):
	L = np.arange(n_epoch)
	halfway = n_epoch//2
	divi = halfway//4
	a = ((np.tanh((L-halfway)/divi)+1)/2)*stop
	return a


