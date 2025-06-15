#!/usr/bin/env python3
# validate_insights.py

import pandas as pd
import networkx as nx
from scipy.stats import pearsonr

def load_data():
    # Load nodes and edges, build NetworkX graph
    nodes = pd.read_csv('DataNodes.csv')
    edges = pd.read_csv('edges.csv')
    G = nx.Graph()
    G.add_nodes_from(nodes['Id'])
    src_col, tgt_col = edges.columns[0], edges.columns[1]
    G.add_edges_from(edges[[src_col, tgt_col]].itertuples(index=False, name=None))
    return nodes, G

def validate_scale_free(nodes):
    # Proportion of nodes in top 10% and top 50% by degree
    total = len(nodes)
    p10 = (nodes['Degree'] >= nodes['Degree'].quantile(0.90)).sum() / total
    p50 = (nodes['Degree'] >= nodes['Degree'].quantile(0.50)).sum() / total
    print(f"Top 10% degree nodes: {p10:.2%}")
    print(f"Top 50% degree nodes: {p50:.2%}")

def validate_degree_pagerank(nodes):
    # Pearson correlation between Degree and PageRank
    corr, p = pearsonr(nodes['Degree'], nodes['pageranks'])
    print(f"Degree vs PageRank Pearson r: {corr:.3f} (p={p:.3g})")

def validate_followers_degree(nodes):
    # Pearson correlation on log-transformed followers vs degree
    import numpy as np
    deg = nodes['Degree']
    foll = nodes['followers']
    mask = (foll > 0) & (deg > 0)
    corr, p = pearsonr(np.log1p(foll[mask]), np.log1p(deg[mask]))
    print(f"log(Followers) vs log(Degree) Pearson r: {corr:.3f} (p={p:.3g})")

def validate_clustering_distribution(nodes):
    # Counts and percentages for clustering coefficient thresholds
    total = len(nodes)
    ge_05 = (nodes['clustering'] >= 0.5).sum()
    eq_1  = (nodes['clustering'] == 1.0).sum()
    mean  = nodes['clustering'].mean()
    print(f"Clustering ≥0.5: {ge_05} / {total} ({ge_05/total:.2%})")
    print(f"Clustering =1.0: {eq_1} / {total} ({eq_1/total:.2%})")
    print(f"Mean clustering: {mean:.4f}")

def validate_community_sizes(nodes):
    # Summary of community size distribution
    sizes = nodes['modularity_class'].value_counts()
    large = (sizes > 1000).sum()
    small = (sizes < 100).sum()
    print(f"Communities >1000 nodes: {large}")
    print(f"Communities <100 nodes: {small}")

def validate_bridges(G, nodes):
    # Identify top bridge nodes by betweenness and show their closeness
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    df = nodes.copy()
    df['betweenness'] = df['Id'].map(bc)
    df['closeness']   = df['Id'].map(cc)
    top_bridges = df.nlargest(5, 'betweenness')
    print("Top 5 bridge nodes (by betweenness):")
    print(top_bridges[['Label','betweenness','closeness']])

def validate_eccentricity_closeness(nodes):
    # Pearson correlation between eccentricity and closeness
    corr, p = pearsonr(nodes['Eccentricity'], nodes['closnesscentrality'])
    print(f"Eccentricity vs Closeness Pearson r: {corr:.3f} (p={p:.3g})")

if __name__ == '__main__':
    nodes, G = load_data()
    print("=== Scale-free structure ===")
    validate_scale_free(nodes)
    print("\n=== Degree ↔ PageRank ===")
    validate_degree_pagerank(nodes)
    print("\n=== Followers ↔ Degree ===")
    validate_followers_degree(nodes)
    print("\n=== Clustering distribution ===")
    validate_clustering_distribution(nodes)
    print("\n=== Community sizes ===")
    validate_community_sizes(nodes)
    print("\n=== Bridge nodes analysis ===")
    validate_bridges(G, nodes)
    print("\n=== Eccentricity ↔ Closeness ===")
    validate_eccentricity_closeness(nodes)
