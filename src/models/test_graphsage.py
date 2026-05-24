import torch
from pathlib import Path
from graphsage_model import GraphSAGE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH=PROJECT_ROOT/"data"/"graph"/"financial_graph.pt"

graph_data=torch.load(GRAPH_PATH,weights_only=False)
print(graph_data)

model=GraphSAGE(input_dim=16,hidden_dim=32,output_dim=16)

embeddings = model(graph_data.x,graph_data.edge_index)

print("\nNODE EMBEDDINGS:\n")
print(embeddings)

print("\nEMBEDDING SHAPE:\n")
print(embeddings.shape)