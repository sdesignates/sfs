# In[]
import sys

sys.path.append('./src/')


import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

import g_admm as FCLGRN
import kernel as kernel
import warnings


warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from umap import UMAP

plt.rcParams["font.size"] = 20

path = "./data/mESC/"

counts = pd.read_csv(path + "counts.csv", index_col=0).values
annotation = pd.read_csv(path + "anno.csv", index_col=0)
GRN = pd.read_csv(path + "GRN.csv", index_col=0)
ncells, ngenes = counts.shape

print("Raw TimePoints: {}, no.Genes: {}".format(counts.shape[0], counts.shape[1]))
pca_op = PCA(n_components=20)
umap_op = UMAP(n_components=2, min_dist=0.8, random_state=0)

X_pca = pca_op.fit_transform(counts)


# hyper-parameters
bandwidth = 0.1
n_neigh = 30
lamb = 0.1
max_iters = 500
beta = 0.1
alpha = 1
# calculate the kernel function
start_time = time.time()
K, K_trun = kernel.calc_kernel_neigh(X_pca, k=5, bandwidth=bandwidth, truncate=True, truncate_param=n_neigh)
print("number of neighbor being considered: " + str(np.sum(K_trun[int(ncells / 2), :] > 0)))

# estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
empir_cov = FCLGRN.est_cov(X=counts, K_trun=K_trun, weighted_kt=True)

# estimate cell-specific GRNs
fclgrn = FCLGRN.G_admm_batch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize=120)
thetas = fclgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb, beta=beta, alpha=alpha)

np.save(file="./thetas2_" + str(bandwidth) + "_" + str(lamb) + "_" + str(n_neigh) + ".npy", arr=thetas)
print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))

# In[] Plots
thetas = thetas.reshape(thetas.shape[0], -1)
thetas_umap = umap_op.fit_transform(thetas)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(thetas_umap[idx, 0], thetas_umap[idx, 1], label=i, s=10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale=3)
ax.set_title("Our method")
fig.savefig("plot_thetas3_" + str(bandwidth) + "_" + str(lamb) + "_" + str(n_neigh) + "_umap.png", bbox_inches="tight")


