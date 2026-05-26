import pandas as pd

# Load traffic data
print("Loading 17_Traffic_30min_30days.xlsx...")
df_traffic = pd.read_excel("data/17_Traffic_30min_30days.xlsx")
# Find header row
header_idx = df_traffic[df_traffic.iloc[:, 0] == 'Timestamp'].index[0]
cols = df_traffic.iloc[header_idx].tolist()
traffic_data = df_traffic.iloc[header_idx+1:].copy()
traffic_data.columns = cols
print("Traffic unique nodes:", traffic_data["Node_ID"].nunique())
print("Traffic nodes sample:", traffic_data["Node_ID"].unique()[:10])
print("Traffic columns:", traffic_data.columns.tolist())
print("Traffic shape:", traffic_data.shape)

# Load adjacency matrix
print("\nLoading 16_Adjacency_Matrix.xlsx...")
df_adj = pd.read_excel("data/16_Adjacency_Matrix.xlsx")
adj_header_idx = df_adj[df_adj.iloc[:, 0] == 'Node_i'].index[0]
adj_cols = df_adj.iloc[adj_header_idx].tolist()
adj_data = df_adj.iloc[adj_header_idx+1:].copy()
adj_data.columns = adj_cols
print("Adjacency rows:", len(adj_data))
print("Adjacency columns:", adj_data.columns.tolist())
print("Adjacency sample:\n", adj_data.head(5))

# Load node registry
print("\nLoading 15_Extended_Nodes.xlsx...")
df_nodes = pd.read_excel("data/15_Extended_Nodes.xlsx")
node_header_idx = df_nodes[df_nodes.iloc[:, 0] == 'Node_ID'].index[0]
node_cols = df_nodes.iloc[node_header_idx].tolist()
node_data = df_nodes.iloc[node_header_idx+1:].copy()
node_data.columns = node_cols
print("Nodes in registry:", len(node_data))
print("Node registry sample:\n", node_data.head(3))
