import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def myPCA(X):
    # mmat = np.matrix([i - np.mean(i) for i in X])
    mmat = np.matrix(X)
    cov = np.cov(mmat.T, mmat.T)
    vals, vecs = np.linalg.eigh(cov)
    vecs = [i[:len(X[0])] for i in vecs.T]
    return {"sv": vals, "components": vecs}


# preample
sns.set_style("ticks")


# Define signals
x = np.arange(0, 1000)
p1 = 1 * np.e**-((500 - x) / 225)**2
p2 = 2 * np.e**-((750 - x) / 100)**2


# Mix
fraction_1 = np.array([0.5, 0.6, 0.55, 0.9, 0.7, 0.77, 0.99])
fraction_2 = np.array([0.9, 0.3, 0.4, 0.5, 0.35, 0.2, 0.1])
mix = [fraction_1[i] * p1 + fraction_2[i] * p2 for i in range(len(fraction_1))]


# Show the mix
for i in mix:
    plt.plot(x, i)
plt.show()


# Get the components
v = myPCA(mix)
print(v["sv"])
plt.plot(v["components"][-1])
plt.plot(v["components"][-2])
plt.title("my PCs")
plt.show()

# Sklearn versoin
skca = PCA(n_components=5)
skca.fit(mix)
print(skca.singular_values_)
plt.plot(skca.components_[0])
plt.plot(skca.components_[1])
plt.show()
