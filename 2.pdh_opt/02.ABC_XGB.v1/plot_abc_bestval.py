import matplotlib.pyplot as plt
import numpy as np

fits = np.loadtxt('fits.csv', delimiter=',')
plt.plot(fits)
plt.savefig('fits.png')