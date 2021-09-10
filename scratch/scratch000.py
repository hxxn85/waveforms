from numpy import *
from scipy.signal import *
from scipy.linalg import hadamard
from matplotlib import pyplot as plt

def xcorr(x, y):
    return correlate(x, y, mode='full') / len(x)

def attenuate(x, d, a):
    return x / d ** a if d > 1 else x

h = hadamard(32)[3:6]
d = linspace(0, 10, 101)

x = attenuate(h[0], 100, 4)
y = [attenuate(h[0], k, 2) for k in d]

r = [max(abs(xcorr(x, k))) for k in y]
plt.plot(d, r)
plt.grid(alpha=.5, ls='--')
plt.show()














