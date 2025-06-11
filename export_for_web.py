import pandas as pd
import pyreadr
import numpy as np
import networkx as nx
import json

print("--- 方案三：为全网络预计算布局并导出 (已修复) ---")

# --- 1. 数据加载 ---
result = pyreadr.read_r('west.Rdata')
df = result['west']
first_column_name = df.columns[0]
char_df = df.drop(columns=[first_column_name])
print(f"已加载全部 {len(char_df.columns)} 位角色数据。")

# --- 2. 计算共现次数邻接矩阵 (无过滤) ---
adj_matrix = char_df.T.dot(char_df)
adj_matrix[adj_matrix < 1] = 0 
np.fill_diagonal(adj_matrix.values, 0)

# --- 3. 构建完整的NetworkX图 ---
G = nx.from_pandas_adjacency(adj_matrix)
G.remove_edges_from(list(nx.isolates(G)))
print(f"已构建包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边的完整网络。")

# --- 4. 预先计算网络布局 ---
print("\n正在Python端预先计算节点布局，这可能需要几分钟...")
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
print("布局计算完成。")

# --- 5. 计算所有节点属性用于可视化 (核心修复部分) ---
print("正在为节点计算所有中心性和社群属性...")

# a. 计算社群ID
communities_generator = nx.community.louvain_communities(G, weight='weight', resolution=1.5, seed=42)
community_map = {node: i for i, comm in enumerate(communities_generator) for node in comm}
nx.set_node_attributes(G, community_map, 'group')

# b. 计算所有四项中心性指标
# 度中心性 (用作节点大小的基础)
degree_centrality = dict(G.degree())

# 间介中心性
betweenness_centrality = nx.betweenness_centrality(G, weight='weight', seed=42)

# 接近度中心性 (需要先将'权重'转换为'距离')
# 我们定义距离为权重的倒数，权重越高距离越近
for u, v, d in G.edges(data=True):
    if d['weight'] > 0:
        d['distance'] = 1 / d['weight']
closeness_centrality = nx.closeness_centrality(G, distance='distance')

# 特征向量中心性
# 注意：对于非连通图，此计算可能只对最大连通分量有效，NetworkX会自动处理
try:
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
except nx.NetworkXError:
    print("警告: 特征向量中心性计算遇到问题，可能图不完全连通。使用默认值0。")
    eigenvector_centrality = {node: 0.0 for node in G.nodes()}

# c. 将所有计算出的属性设置到图节点上
nx.set_node_attributes(G, degree_centrality, 'degree')
nx.set_node_attributes(G, betweenness_centrality, 'betweenness')
nx.set_node_attributes(G, closeness_centrality, 'closeness')
nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector')


# --- 6. 转换为适用于Vis.js的JSON格式，并包含所有属性 ---
vis_data = {"nodes": [], "edges": []}

for node, attrs in G.nodes(data=True):
    x, y = pos[node]
    vis_data["nodes"].append({
        "id": node,
        "label": node,
        "group": attrs.get('group', 0),
        "value": attrs.get('degree', 1), # 节点大小仍然基于度数
        "x": x * 2000,
        "y": y * 2000,
        "physics": False,
        "title": f"角色: {node}<br>总互动量: {int(attrs.get('degree', 0))}",
        # 【关键修复】将所有中心性值都导出，以便在前端使用
        "value_degree": attrs.get('degree', 0),
        "value_betweenness": attrs.get('betweenness', 0),
        "value_closeness": attrs.get('closeness', 0),
        "value_eigenvector": attrs.get('eigenvector', 0)
    })

for source, target, attrs in G.edges(data=True):
    vis_data["edges"].append({
        "from": source,
        "to": target,
        "value": attrs.get('weight', 1),
        "title": f"共同出场: {int(attrs.get('weight', 0))}次"
    })

# --- 7. 保存为JSON文件 ---
with open('network_data_full.json', 'w', encoding='utf-8') as f:
    json.dump(vis_data, f, ensure_ascii=False)

print("\n成功！已将包含所有中心性指标的完整网络数据导出为 network_data_full.json 文件。")