# 거리에 따른 auto and cross-correlation 확인

from numpy import *
from scipy.linalg import hadamard
from scipy.signal import *

class Radar:
    def __init__(self, pos, x):
        self.pos = pos
        self.x = x

    def filter(self, r):
        def _xcorr(x, y):
            rxx0 = correlate(x, x)
            ryy0 = correlate(y, y)
            return correlate(x, y, mode='full') / np.sqrt(rxx0 * ryy0)

        return _xcorr(self.x, r)

class Channel:
    def __init__(self, radar, pos, channel):
        self.radar = radar
        self.pos = pos
        self.channel = channel

    def pass_through(self):
        return self._attenuate(self._impulse_response())

    def _impulse_response(self):
        return convolve(self.radar.x, self.channel)

    def _attenuate(self, r):
        d = 2*linalg.norm(self.radar.pos, self.pos)
        return r / (d ** 4)

pos = [[0, 0], [10, 0], [15, 0]]
x = hadamard(32)[0:2]

radars = [Radar(p, y) for p, y in zip(pos, x)]

