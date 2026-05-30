import torch
import torch.nn.functional as F

from pathlib import Path

from graphsage_model import GraphSAGE


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

GRAPH_PATH = (
    PROJECT_ROOT /
    "data" /
    "graph" /
    "financial_graph.pt"
)

graph_data = torch.load(
    GRAPH_PATH,
    weights_only=False
)

print(graph_data)

device = torch.device(
    "cpu"
)

model = GraphSAGE(
    input_dim=graph_data.num_node_features,
    hidden_dim=32,
    output_dim=16
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

x = graph_data.x.to(device)

edge_index = (
    graph_data.edge_index
    .to(device)
)

epochs = 500

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    embeddings = model(
        x,
        edge_index
    )

    reconstructed_adj = torch.matmul(
        embeddings,
        embeddings.t()
    )

    target_adj = torch.zeros(
        (
            graph_data.num_nodes,
            graph_data.num_nodes
        )
    ).to(device)

    for i in range(
        edge_index.shape[1]
    ):

        src = edge_index[0][i]
        dst = edge_index[1][i]

        target_adj[src][dst] = 1

    loss = F.mse_loss(
        reconstructed_adj,
        target_adj
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:

        print(
            f"Epoch {epoch} "
            f"Loss: {loss.item():.4f}"
        )

MODEL_PATH = (
    PROJECT_ROOT /
    "data" /
    "graph" /
    "graphsage_model.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nMODEL TRAINED AND SAVED\n")

print(
    f"Saved model to:\n"
    f"{MODEL_PATH}"
)

model.eval()

with torch.no_grad():

    final_embeddings = model(
        x,
        edge_index
    )

print("\nFINAL EMBEDDINGS:\n")

print(final_embeddings)

print("\nEMBEDDING SHAPE:\n")

print(final_embeddings.shape)