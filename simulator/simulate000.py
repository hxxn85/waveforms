from optimizer.simulated_annealing import *
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

def pertfun(s, m):
    _s = s.copy()
    l, n = s.shape
    row, col = np.random.randint(l), np.random.randint(n)
    phase = np.exp(1j*np.random.randint(m) * 2 * np.pi / m)
    _s[row][col] = phase
    return _s, row

def mixfun(x, s):
    _s = np.repeat(s, x.shape[-1]/s.shape[-1], axis=1)
    return x * _s

def costfun(y, l, row, mat):
    def _xcorr(x, y):
        rxx0 = np.max(signal.correlate(x, x))
        ryy0 = np.max(signal.correlate(y, y))
        return signal.correlate(x, y, mode='full')/np.sqrt(rxx0 * ryy0)

    _mat = mat.copy()
    # autocorrelation sidelobe peak
    rxx = _xcorr(y[row], y[row])
    tmp = np.abs(rxx)[len(y[row])-1:]
    ml = np.where(tmp < 1/np.sqrt(len(y[row])))[0][0]
    _mat[row][row] = np.max(tmp[ml:])
    asp = np.max(np.diag(_mat))

    # cross-correlation
    for i in range(y.shape[0]):
        if i == row:
            continue

        rxy = _xcorr(y[row], y[i])
        tmp = np.max(np.abs(rxy))
        _mat[row][i], _mat[i][row] = tmp, tmp

    cp = np.max(np.triu(_mat, 1))

    return (1-l) * asp + l * cp, _mat

x = CodedLFM()
x.l, x.m, x.n, x.a = 4, 8, 32, 0.5

tp, bw = 1e-6, 16e6
fs = 16 * bw
n = int(fs*tp)
t = np.arange(0, n) / fs
a = bw/tp
x.x = np.tile(np.exp(1j*2*np.pi*a*t**2), (x.l, 1))
x.s = np.exp(1j*np.zeros((x.l, x.n)))

opt = SimpleOptimizer(x, pertfun, mixfun, costfun)
opt.optimize()
print(20*np.log10(opt.mat))

#%%
y = opt.x.y[0]
f, pyy = signal.periodogram(y, fs, return_onesided=False, detrend=None)
plt.semilogy(np.fft.fftshift(f), np.fft.fftshift(pyy.T))
plt.show()
