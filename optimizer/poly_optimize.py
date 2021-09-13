import numpy as np

from scipy import constants
from scipy.spatial import distance

def db2pow(x):
    return 10**(x/20)

def pow2db(x):
    x = np.asarray(x)
    return 20*np.log10(x.clip(1e-11))

class Waveform:
    def __init__(self, x, tp, fs, prf):
        self.x = x              # waveform
        self.tp = tp            # pulse width
        self.fs = fs            # sampling frequency
        self.prf = prf          # pulse repetition frequency

    def get_waveform(self):
        return self.x

    def get_pulse(self, return_state=False):
        n = int((1/self.prf - self.tp) * self.fs)  # number of samples for receiving
        pulse = np.concatenate([self.x, np.zeros(n)])
        if return_state:
            state = np.concatenate([np.ones(len(self.x)), np.zeros(n)])
            return pulse, np.asarray(state, dtype=np.int32)
        else:
            return pulse

class Antenna:
    def __init__(self, gain, fc, frange):
        self.gain = gain                # antenna gain
        self.fc = fc                    # carrier frequency
        self.frange = frange            # frequency range

class Radiator(Antenna):
    def radiate(self, x):
        return db2pow(self.gain) * x

class Collector(Antenna):
    def collect(self, x, fc, state=None):
        lmda = 3e8 / fc                                                 # wavelength
        _x = np.sum(x, axis=0)
        pn = constants.k * 290 * (self.frange[1] - self.frange[0])      # noise power density
        noise = np.random.rand(len(_x)) * pn
        y = db2pow(self.gain) * _x * (lmda**2) / (4*np.pi) + noise
        if state is not None:
            mask = np.asarray(state)
            y[~mask] = 0

        return y


class Channel:
    def __init__(self, fs):
        self.fs = fs

    def propagate(self, x, src, dst):
        src, dst = np.asarray(src), np.asarray(dst)
        if src.ndim == 1 and dst.ndim == 1:
            r = distance.euclidean(src, dst)
            n = int((r / constants.c) * self.fs)
            return np.roll(x, n) / (4*np.pi*r**2)
        elif src.ndim == 1 and dst.ndim > 1:    # one radar, n targets
            r = np.array([distance.euclidean(src, k) for k in dst])
            n = list(map(int, (r / constants.c) * self.fs))
            return [np.roll(x, k) / (4*np.pi*d**2) for k, d in zip(n, r)]
        elif src.ndim > 1 and dst.ndim == 1:   # n radars, one target
            r = np.array([distance.euclidean(k, dst) for k in src])
            n = list(map(int, (r / constants.c) * self.fs))
            y = []
            for tgt, d, delay in zip(x, r, n):
                y.append(np.roll(tgt, delay) / (4*np.pi*d**2))

            return np.array(y)

        else:
            pass

class Target:
    def __init__(self, rcs, location):
        self.rcs = rcs              # target radar cross section
        self.location = location    # target location

    def reflect(self, x):
        return self.rcs * np.asarray(x)


class Radar:
    def __init__(self, waveforms, idx, pt, fc, location, radiator: Radiator, collector: Collector):
        self.waveforms = waveforms  # waveforms for matched filter
        self.idx = idx              # index
        self.x = waveforms[idx]     # tx waveform
        self.pt = pt                # peak power
        self.fc = fc                # carrier frequency
        self.location = location    # location of the radar
        self.radiator = radiator    # transmitting antenna
        self.collector = collector  # receiving antenna

    def transmit(self, return_state=False):
        pulse = self.x.get_pulse(return_state)
        if return_state:
            pulse, state = pulse
            return self.radiator.radiate(self.pt * pulse), state
        else:
            return self.radiator.radiate(self.pt * pulse)

    def receive(self, x, state=None):
        return self.collector.collect(x, self.x.fs, state)

    def filter(self, x):
        a = self.x.get_waveform()
        # todo: calculate noise power density
        b = np.concatenate([x.copy(), 1.20116463e-11 * np.random.rand(len(a)-1)])
        n = len(x)
        out = []
        for i in range(n):
            out.append(np.abs(np.dot(b[i:len(a)+i], np.conj(a))))

        return out

    def nfilter(self, x):
        a = self.x.get_waveform()
        b = x.copy()
        pa = np.sum(np.abs(a)**2)   # power of the transmitted signal
        b = np.concatenate([x.copy(), 1.20116463e-11 * np.random.rand(len(a) - 1)])
        n = len(x)
        out = []
        for i in range(n):
            tmp = np.asarray(b[i:len(a) + i])
            pb = np.sum(np.abs(tmp) ** 2) # received signal power
            # for j in range(len(self.waveforms)):
            #     if j == self.idx: continue
            #
            #     w = self.waveforms[j]
            #     matched_out = np.array(np.abs(np.dot(tmp, np.conj(w.x))))
            #     tmp = tmp - (matched_out * w.x)

            matched_out = np.array([np.abs(np.dot(tmp, np.conj(w.x))) for w in self.waveforms])
            tmp = tmp - np.sum([k * x.x for k, x in zip(matched_out, self.waveforms)], axis=0)

            out.append(np.abs(np.dot(tmp, np.conj(a))) / np.sqrt(pb * pa))

        return np.asarray(out)


#%%
# tp, prf, fc = 1e-6, 1/20e-6, 1e9
# fs = 256e6
# lamd = constants.c / fc
#
# rcs = 0.1
# gain = 20
# snr = 10.5298   # neyman-pearson white gaussian noise threshold for Pfa = 1e-6, 1 pulse and coherent radar
#
# maxrange = constants.c / 2 / prf
# dbterm = db2pow(snr - 2*gain)
# loss = constants.k * 290 * 1/tp
# pt = (4*np.pi)**3 * maxrange**4 * dbterm * loss / lamd**2 / rcs * 10
#
# tgtloc = [[500,500,0], [1500,1500,0], [2500,1000,0]]
# radloc = [[0,0,0], [1000,0,0], [2000,0,0], [3000,0,0]]
#
# wavs = [Waveform(x, tp, fs, prf) for x in np.repeat(hadamard(32)[[25,29,19,10]], 8, 1)]
# radars = [Radar(x, pt, fc, loc, Radiator(gain, fc, [0, 3e9]), Collector(gain, fc, [0, 3e9])) \
#           for x, loc in zip(wavs, radloc)]
# tg = Target(rcs, tgtloc)
# ch = Channel(fs)
#
# def process(tx: Radar, rx: Radar, ch: Channel, tg: Target):
#     sig, state = tx.transmit(return_state=True)
#     sig = ch.propagate(sig, tx.location, tg.location)
#     sig = tg.reflect(sig)
#     sig = ch.propagate(sig, tg.location, rx.location)
#     sig = rx.receive(sig, state ^ 1)
#     return sig
#
# tx = np.sum([process(r, radars[0], ch, tg) for r in radars], axis=0)
#
# t = np.linspace(0, 1/prf, len(tx), endpoint=False)
# rangegates = constants.c * t / 2
# plt.subplot(2,1,1)
# plt.plot(rangegates, pow2db(tx))
# plt.grid(alpha=.5, ls='--')
# plt.xlabel('range gate(m)')
# plt.ylabel('received power(dB)')
# plt.title('received signal')
#
# tx = radars[0].filter(tx)
# plt.subplot(2,1,2)
# plt.plot(rangegates, pow2db(tx))
# [plt.axvline(rangegates[k], ls='--', color='orange') for k in signal.find_peaks(tx, db2pow(-170), width=50)[0]]
# [plt.axvline(norm(k), label='target', ls='--', color='k') for k in tgtloc]
# plt.legend()
# plt.grid(alpha=.5, ls='--')
# plt.xlabel('range gate(m)')
# plt.ylabel('received power(dB)')
# plt.title('matched filter output')
# plt.tight_layout()
# plt.show()

#%%
# kk = radars[0].x.get_waveform()
# rxx = signal.correlate(kk, kk)
# plt.plot(np.abs(rxx))
# plt.plot(radars[0].filter(kk))
# plt.show()


