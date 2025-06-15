#!/usr/bin/env python3
# analyze_collabnet.py

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Create output directory for plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load CSVs
    nodes = pd.read_csv('DataNodes.csv')
    edges = pd.read_csv('edges.csv')

    # 2. Detect source/target columns
    src_col, tgt_col = edges.columns[0], edges.columns[1]
    print(f"Using edge columns: {src_col!r} → {tgt_col!r}")

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(nodes['Id'])
    G.add_edges_from(edges[[src_col, tgt_col]].itertuples(index=False, name=None))

    # Plot helpers
    def save_hist(series, filename, title, xlabel, ylabel):
        plt.figure()
        plt.hist(series.dropna(), bins=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def save_scatter(x, y, filename, title, xlabel, ylabel, logx=False, logy=False):
        plt.figure()
        plt.scatter(x, y, s=5)
        if logx: plt.xscale('log')
        if logy: plt.yscale('log')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # 4. Generate and save plots
    save_hist(nodes['Degree'], 'degree_distribution.png',
              'Degree Distribution', 'Degree', 'Count')

    save_hist(nodes['pageranks'], 'pagerank_distribution.png',
              'PageRank Distribution', 'PageRank', 'Count')

    save_scatter(nodes['Degree'], nodes['pageranks'],
                 'degree_vs_pagerank.png',
                 'Degree vs. PageRank', 'Degree', 'PageRank')

    save_scatter(nodes['followers'], nodes['Degree'],
                 'followers_vs_degree.png',
                 'Followers vs. Degree (log–log)',
                 'Followers', 'Degree',
                 logx=True, logy=True)

    save_hist(nodes['clustering'], 'clustering_distribution.png',
              'Clustering Coefficient Distribution',
              'Clustering Coefficient', 'Count')

    # Correlation matrix
    metrics = ['followers','popularity','Degree','pageranks',
               'clustering','triangles','eigencentrality','betweenesscentrality']
    corr = nodes[metrics].corr()
    plt.figure(figsize=(8,8))
    plt.imshow(corr, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(metrics)), metrics, rotation=45, ha='right')
    plt.yticks(range(len(metrics)), metrics)
    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Boxplot of three centralities
    plt.figure()
    plt.boxplot([nodes['Degree'], nodes['pageranks'], nodes['clustering']])
    plt.xticks([1,2,3], ['Degree','PageRank','Clustering'])
    plt.title('Boxplot of Degree, PageRank & Clustering')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_boxplot.png'))
    plt.close()

    save_scatter(nodes['Degree'], nodes['triangles'],
                 'triangles_vs_degree.png',
                 'Triangles vs. Degree', 'Degree', 'Triangle Count')

    save_scatter(nodes['betweenesscentrality'], nodes['closnesscentrality'],
                 'betweenness_vs_closeness.png',
                 'Betweenness vs. Closeness Centrality',
                 'Betweenness', 'Closeness')

    # Community size histogram
    plt.figure()
    nodes['modularity_class'].value_counts().hist(bins=50)
    plt.title('Community Size Distribution')
    plt.xlabel('Nodes per Community')
    plt.ylabel('Number of Communities')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_size_distribution.png'))
    plt.close()

    save_hist(nodes['Eccentricity'], 'eccentricity_distribution.png',
              'Eccentricity Distribution', 'Eccentricity', 'Count')

    # Spatial layout (X vs Y)
    if 'X' in nodes.columns and 'Y' in nodes.columns:
        save_scatter(nodes['X'], nodes['Y'],
                     'positions_xy.png',
                     'Node Positions (X vs Y)',
                     'X coordinate', 'Y coordinate')

    # Subgraph of top-100 by Degree
    top100 = nodes.nlargest(100, 'Degree')['Id']
    subG = G.subgraph(top100)
    pos = nx.spring_layout(subG)
    plt.figure(figsize=(8,8))
    nx.draw(subG, pos, node_size=20, with_labels=False)
    plt.title('Top-100 Degree Subgraph')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top100_subgraph.png'))
    plt.close()

    print(f"All plots saved under '{output_dir}/'.")

if __name__ == '__main__':
    main()
