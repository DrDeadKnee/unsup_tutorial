import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Pay attention here
figname = "loss_well"
x = np.arange(-2, 10, 0.1)
xlabel = f"X"
ylabel = f"L(y - f(X))"


def func(x):
    return (x - 5)**2


# Templated plotting
sns.set_style('ticks')
# f = plt.figure(figsize=(12, 8))

y = func(x)
plt.plot(x, y, label="L(y - f(X))")
plt.plot(0, 25, "ro", label="first guess")
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
plt.legend(loc="best")

plt.savefig(f"img/{figname}.jpg")
plt.show()
