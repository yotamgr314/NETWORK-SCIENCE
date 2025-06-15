import pandas as pd

nodes = pd.read_csv('DataNodes.csv')
clust = nodes['clustering']

n_total       = len(nodes)                         # 156,327
n_ge_05       = (clust >= 0.5).sum()               # ≈8,000
n_eq_1        = (clust == 1.0).sum()               # ≈500
mean_clust    = clust.mean()                       # ≈0.02

print(f"Total nodes:      {n_total}")
print(f"Clustering ≥0.5:   {n_ge_05}")
print(f"Clustering =1.0:   {n_eq_1}")
print(f"Mean clustering:  {mean_clust:.4f}")
