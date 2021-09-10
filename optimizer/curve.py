from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np

def func(x, a, b):
    return a * x + b

xdata = np.array([1000, 1500, 2000, 2500, 3000])
ydata = np.array([-15, -12.5, -10, -7.5, -5])

plt.plot(xdata, ydata, '-o')
plt.grid(alpha=.5, ls='--')
plt.show()

popt, pcov = curve_fit(func, xdata, ydata, bounds=([0, -50], [10, 50]))
print(popt)