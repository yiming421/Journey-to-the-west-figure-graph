from unittest.util import sorted_list_difference
from networkx import average_clustering, diameter, eccentricity, eigenvector_centrality, shortest_path
import pyreadr
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

try:
    # For Windows/macOS
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC'] # Add fonts you might have
    plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus signs
except Exception as e:
    print(e)
    print("Could not configure Chinese font. Please install a compatible font like 'SimHei' or 'Microsoft YaHei'.")
# --- END OF PREAMBLE ---

result = pyreadr.read_r('west.Rdata')

# Extract the DataFrame from the result
df = result['west']

first_column_name = df.columns[0]
char_df = df.drop(columns=[first_column_name])

data_matrix = char_df.to_numpy()
characters = char_df.columns.tolist()

# Calculate the dot product

adj_ori = char_df.T.dot(char_df)
print(adj_ori)

adj = adj_ori.copy()
np.fill_diagonal(adj.values, 0)

# Convert the correlation matrix to a graph
nx_graph = nx.from_pandas_adjacency(adj)

degrees = dict(nx_graph.degree())
avg_degree = np.mean(list(degrees.values()))
print("Graph statistics:")
print(f"Number of nodes: {nx_graph.number_of_nodes()}")
print(f"Number of edges: {nx_graph.number_of_edges()}")
print(f"Average degree: {avg_degree}")

avg_shortest_path_len = nx.average_shortest_path_length(nx_graph)
print(f"Average shortest path length in the largest connected component: {avg_shortest_path_len}")

avg_cluster = average_clustering(nx_graph)
print(f"Average clustering coefficient: {avg_cluster}")

dia = nx.diameter(nx_graph)
print(f"Diameter of the largest connected component: {dia}")
ecc = nx.eccentricity(nx_graph)
poss_node = [node for node, value in ecc.items() if value == dia]
start_node = poss_node[3] if poss_node else None
shortest_paths = nx.shortest_path(nx_graph, source=start_node)
for target_node, path in shortest_paths.items():
    if len(path) == dia + 1:
        print(f"Longest path from {start_node} to {target_node}: {path}")

print("Characters with the highest degree centrality:")
for char, degree in sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"{char}: {degree}")

betweeness = nx.betweenness_centrality(nx_graph)
print("Characters with the highest betweenness centrality:")
for char, betweenness in sorted(betweeness.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"{char}: {betweenness}")

closeness = nx.closeness_centrality(nx_graph)
print("Characters with the highest closeness centrality:")
for char, closeness_value in sorted(closeness.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"{char}: {closeness_value}")

eigen_centrality = eigenvector_centrality(nx_graph)
print("Characters with the highest eigenvector centrality:")
for char, eigen in sorted(eigen_centrality.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"{char}: {eigen}")

community_generator = nx.community.louvain_communities(nx_graph, resolution=1.6, seed=42)
print("Communities found:")
for i, community in enumerate(community_generator):
    print(f"Community {i + 1}: {[char for char in community]}")

community_map = {node: i for i, community in enumerate(community_generator) for node in community}
node_colors = [community_map[node] for node in nx_graph.nodes()]
node_sizes = [degrees[node] * 1 for node in nx_graph.nodes()]
top_nodes = {n for n, d in sorted(nx.degree_centrality(nx_graph).items(), key=lambda x: x[1], reverse=True)[:20]}
labels = {node: node if node in top_nodes else '' for node in nx_graph.nodes()}
edge_colors = []
edge_weights = []
for u, v, data in nx_graph.edges(data=True):
    if data['weight'] > 0:
        edge_colors.append('green')
    else:
        edge_colors.append('red')
    edge_weights.append(abs(data['weight']) * 0.1)

plt.figure(figsize=(20, 20))
pos = nx.spring_layout(nx_graph, seed=42, k=0.9)

nx.draw(nx_graph, pos, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.tab20, 
        width=edge_weights, edge_color=edge_colors, alpha=0.7)
plt.title("Character Network in Journey to the West")
nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_color='red', font_size=10)

plt.show()
plt.savefig("character_network.png", format='png', dpi=300)
    