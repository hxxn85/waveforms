from scipy.spatial import distance
from scipy import signal
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

def xcorr(x, y):
    rxx = np.max(np.abs(signal.correlate(x, x)))
    ryy = np.max(np.abs(signal.correlate(y, y)))
    num = signal.correlate(x, y)
    den = np.sqrt(rxx * ryy)
    return num / den if den != 0 else num

def costfun(S, M, locations):
    L, N = S.shape
    comb = list(combinations(range(L), 2))
    r = np.array([1 / (distance.euclidean(locations[x], locations[y])**2) * np.abs(signal.correlate(S[x], S[y])) for x, y in comb])
    r[np.isnan(r)] = 1

    return np.sum(r)

def perturbation(S, M):
    _S = S.copy()
    L, N = _S.shape
    row, col = np.random.randint(L), np.random.randint(N)
    phz = np.exp(1j * 2 * np.pi * np.random.randint(M) / M)
    _S[row][col] = phz
    return _S

#%% ---------------------------
# L, M, N = 4, 4, 32
# S = np.zeros((L, N), dtype='complex128')
#
# # initialize temperature
# costs = []
# for i in range(L*N):
#     S = perturbation(S, M)
#     cost = costfun(S, M, [0,0,0,0])
#     costs.append(cost)
#
# cost = costs[-1]
# t = 20 * np.nanstd(costs)
# a = 0.95
#
# niter = 20000
# _S = S.copy()
#
# for i in range(niter):
#     _S = perturbation(_S, M)
#     _cost = costfun(_S, M, [0,0,0,0])
#     de = _cost - cost
#
#     p = 1 if de <= 0 else np.exp(-de/t)
#
#     if np.random.rand() <= p:   # transition occurs
#         S = _S
#         cost = _cost
#         costs.append(cost)
#
#     t = a * t
#
# plt.plot(costs)
# plt.grid(alpha=.5, ls='-.')
# plt.show()
#
# #%%
# comb = list(combinations(range(4), 2))
# r = [np.abs(np.max(xcorr(S[x], S[y]))) for x, y in comb]
# r = 20*np.log10(r)
# print(comb)
# print(r)

