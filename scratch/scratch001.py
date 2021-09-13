from scipy.signal import waveforms

from optimizer.adaptive_polyphase import *
from optimizer.poly_optimize import *
from scipy.linalg import norm

#%% adaptive polyphase code set
L, M, N = 4, 4, 32
S = np.zeros((L, N), dtype='complex128')

# initialize temperature
costs = []
for i in range(L*N):
    S = perturbation(S, M)
    cost = costfun(S, M, [[0,0,0], [1000,0,0], [2000,0,0], [3000,0,0]])
    costs.append(cost)

cost = costs[-1]
t = 20 * np.nanstd(costs)
a = 0.95

niter = 250
_S = S.copy()

for i in range(niter):
    _S = perturbation(_S, M)
    _cost = costfun(_S, M, [[0,0,0], [1000,0,0], [2000,0,0], [3000,0,0]])
    de = _cost - cost

    p = 1 if de <= 0 else np.exp(-de/t)

    if np.random.rand() <= p:   # transition occurs
        S = _S
        cost = _cost
        costs.append(cost)

    t = a * t

# plt.plot(costs)
# plt.grid(alpha=.5, ls='-.')
# plt.show()

comb = list(combinations(range(4), 2))
r = [np.abs(np.max(xcorr(S[x], S[y]))) for x, y in comb]
r = 20*np.log10(r)
print(comb)
print(r)

#%%
tp, prf, fc = 1e-6, 1/20e-6, 1e9
fs = 256e6
lamd = constants.c / fc

rcs = 0.1
gain = 20
snr = 10.5298   # neyman-pearson white gaussian noise threshold for Pfa = 1e-6, 1 pulse and coherent radar

maxrange = constants.c / 2 / prf
dbterm = db2pow(snr - 2*gain)
loss = constants.k * 290 * 1/tp
pt = 10*(4*np.pi)**3 * maxrange**4 * dbterm * loss / lamd**2 / rcs * 10

tgtloc = [[500,500,0], [1500,1500,0], [2500,1000,0]]
# radloc = [[0,0,0], [1000,0,0]]
radloc = [[0,0,0], [1000,0,0], [2000,0,0], [3000,0,0]]
# radloc = [[3000,0,0]]

wavs = [Waveform(x, tp, fs, prf) for x in np.repeat(S, 8, 1)]
radars = [Radar(wavs, idx, pt, fc, loc, Radiator(gain, fc, [0, 3e9]), Collector(gain, fc, [0, 3e9])) \
          for loc, idx in zip(radloc, range(len(radloc)))]
tg = Target(rcs, tgtloc)
ch = Channel(fs)

def process(tx: Radar, rx: Radar, ch: Channel, tg: Target):
    sig, state = tx.transmit(return_state=True)
    sig = ch.propagate(sig, tx.location, tg.location)
    sig = tg.reflect(sig)
    sig = ch.propagate(sig, tg.location, rx.location)
    sig = rx.receive(sig, state ^ 1)
    return sig

rx = [np.sum([process(r, radars[0], ch, tg) for r in radars], axis=0) for _ in range(1)]
rx = np.sum(rx, axis=0)

t = np.linspace(0, 1/prf, len(rx), endpoint=False)
rangegates = constants.c * t / 2
# plt.subplot(2,1,1)
# plt.plot(rangegates, pow2db(tx))
# plt.grid(alpha=.5, ls='--')
# plt.xlabel('range gate(m)')
# plt.ylabel('received power(dB)')
# plt.title('received signal')

#%%
# out = rx - (np.asarray(radars[1].filter(rx)) * rx)

out = radars[0].nfilter(rx)
# plt.subplot(2,1,2)
plt.plot(rangegates, pow2db(out))
# [plt.axvline(rangegates[k], ls='--', color='orange') for k in signal.find_peaks(tx, db2pow(-170), width=30)[0]]
[plt.axvline(norm(k), label='target', ls='--', color='k', alpha=.25) for k in tgtloc]
plt.legend()
plt.grid(alpha=.5, ls='--')
plt.xlabel('range gate(m)')
plt.ylabel('normalized received power(dB)')
plt.title('matched filter output')
plt.tight_layout()
plt.show()



#%% matched filters
