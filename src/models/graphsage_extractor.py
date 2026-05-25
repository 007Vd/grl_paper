import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)


class GraphSAGEExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        features_dim=128
    ):

        super().__init__(
            observation_space,
            features_dim
        )

        self.num_nodes = (
            observation_space.shape[0]
        )

        self.input_dim = (
            observation_space.shape[1]
        )

        self.conv1 = SAGEConv(
            self.input_dim,
            32
        )

        self.conv2 = SAGEConv(
            32,
            16
        )

        self.fc = nn.Sequential(

            nn.Linear(
                self.num_nodes * 16,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                features_dim
            ),

            nn.ReLU()
        )

        edges = []

        for i in range(
            self.num_nodes - 1
        ):

            edges.append([i, i + 1])
            edges.append([i + 1, i])

        self.edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    def forward(
        self,
        observations
    ):

        batch_size = observations.shape[0]

        outputs = []

        for batch_idx in range(batch_size):

            x = observations[
                batch_idx
            ]

            x = self.conv1(
                x,
                self.edge_index
            )

            x = torch.relu(x)

            x = self.conv2(
                x,
                self.edge_index
            )

            x = torch.relu(x)

            x = x.flatten()

            outputs.append(x)

        x = torch.stack(outputs)

        x = self.fc(x)

        return x