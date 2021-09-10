import abc
from abc import *
import numpy as np
from collections import deque

class CodedLFM(metaclass=ABCMeta):
    l = None
    m = None
    n = None
    a = None
    x = []
    s = []
    y = []

class Optimizer(metaclass=ABCMeta):
    def __init__(self, x: CodedLFM, pertfun, mixfun, costfun):
        self.x = x
        self.hist = []
        self.niter = 200
        self.miter = int(np.round(self.x.m**2 * (self.x.l * self.x.n)**2 / 20))
        self.pertfun = pertfun
        self.mixfun = mixfun
        self.costfun = costfun
        self.mat = np.ones((x.l, x.l))

    @abc.abstractmethod
    def _init_temp(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass

class SimpleOptimizer(Optimizer):
    def _init_temp(self):
        costs = []
        for i in range(self.x.l * self.x.n):
            s, row = self.pertfun(self.x.s, self.x.m)
            y = self.mixfun(self.x.x, s)
            cost, self.mat = self.costfun(y, self.x.a, row, self.mat)
            costs.append(cost)
            self.x.s = s

        return 10 * np.std(costs), costs[-1]

    def optimize(self):
        # temperature
        a = 0.95
        t, cost = self._init_temp()
        print(f'initial cost = {cost}')

        # stop condition
        qsize = 3 * self.x.l * self.x.n
        equil = 1 / qsize * 1e-3

        costs = deque([], qsize)
        costs.append(cost)
        mean = np.mean(costs)
        std = np.std(costs)
        diff = deque([0, 0], 2)

        for i in range(self.niter):
            print(f'[{i}] temperature: {t} ({cost})')
            accepted = 0

            for j in range(self.miter):
                s, row = self.pertfun(self.x.s, self.x.m)
                y = self.mixfun(self.x.x, s)
                _cost, mat = self.costfun(y, self.x.a, row, self.mat)
                d = _cost - cost
                p = d < 0 and 1 or np.exp(-d/t)

                if np.random.rand() <= p:
                    self.x.s = s
                    self.x.y = y
                    self.mat = mat
                    cost = _cost
                    accepted += 1

                    pmean, pstd = mean, std
                    costs.append(cost)
                    mean, std = np.mean(costs), np.std(costs)
                    if accepted > qsize:
                        is_stoppable = np.abs(mean - pmean)/pmean < equil and np.abs(std-pstd)/std < equil
                        if is_stoppable:
                            break

            if accepted == 0 and np.sum(diff) == 0:
                print('no perturbation during 3 consecutive iterations')
                break

            diff.append(accepted)
            t *= a
            self.hist.append(cost)

        # iterative code selection
        i = 1
        phases = np.exp(1j*2*np.pi*np.array(range(self.x.m))/self.x.m)
        while True:
            print(f'[{i}] ({cost})')
            pcf = 0
            for row in range(self.x.l):
                for col in range(self.x.n):
                    for phs in phases:
                        if phs == self.x.s[row][col]:
                            continue

                        s = self.x.s.copy()
                        s[row][col] = phs
                        y = self.mixfun(self.x.x, s)
                        _cost, mat = self.costfun(y, self.x.a, row, self.mat)

                        if _cost < cost:
                            self.x.s = s
                            self.x.y = y
                            self.mat = mat
                            cost = _cost
                            i += 1
                            pcf = 1

            if pcf == 0:
                break






