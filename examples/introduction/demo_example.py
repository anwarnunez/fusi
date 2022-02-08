'''
========================
 Sphinx gallery example
========================
'''

# Some commentary
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(100)
y = x**2

fig, ax = plt.subplots()
ax.plot(x, y)

plt.show()
