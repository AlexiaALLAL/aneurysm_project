import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import tqdm

class GNNPredictor(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super(GNNPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # First GCN layer with ReLU
        x = self.conv2(x, edge_index)         # Output layer
        return x
    
    def train_model(self, loader, optimizer, epochs=100):
        self.train()
        for epoch in tqdm.tqdm(range(epochs)):
            total_loss = 0
            for data_t_minus_1, data_t in loader:
                optimizer.zero_grad()
                predictions = self.forward(data_t_minus_1.x, data_t_minus_1.edge_index)  # Predict x_t
                loss = F.mse_loss(predictions, data_t.x)  # Compare with actual x_t
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
        