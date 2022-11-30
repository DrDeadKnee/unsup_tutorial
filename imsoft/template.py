import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Pay attention here
figname = "template"
x = np.arange(-10, 10, 0.1)
xlabel = f"X"
ylabel = f"f(X)"


def func(x):
    return x**2


# Templated plotting
sns.set_style('ticks')
# f = plt.figure(figsize=(12, 8))

y = func(x)
plt.plot(x, y)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.savefig(f"img/{figname}.jpg")
plt.show()
