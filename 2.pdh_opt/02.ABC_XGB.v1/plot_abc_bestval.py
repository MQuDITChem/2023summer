import matplotlib.pyplot as plt
import numpy as np

logs = np.loadtxt('ABC_log.csv', delimiter=',')
plt.plot(logs)
plt.savefig('ABC_logs.png')