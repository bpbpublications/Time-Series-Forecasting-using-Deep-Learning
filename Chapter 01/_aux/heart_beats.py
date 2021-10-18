import scipy
import scipy.signal as sig
from matplotlib import pylab

rr = [1.0, 1.0, 0.5, 1.5, 1.0, 1.0, 1.5, 1.0, 0.5, 1]  # rr time in seconds
fs = 8000.0  # sampling rate
pqrst = sig.wavelets.daub(10)  # just to simulate a signal, whatever
ecg = scipy.concatenate([sig.resample(pqrst, int(r * fs)) for r in rr])
t = scipy.arange(len(ecg)) / fs
pylab.xticks([])
pylab.yticks([])
pylab.plot(t, ecg)
pylab.show()
