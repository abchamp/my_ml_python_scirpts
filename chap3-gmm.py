from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

print(gm.weights_)
# print(gm.means_)
# print(gm.covariances_)
print(gm.converged_)
print(gm.n_iter_)