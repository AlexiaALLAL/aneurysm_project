import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import tqdm

from mesh_handler import get_geometric_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNPredictor(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=4):
        super(GNNPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = x.to(device)
        edge_index = edge_index.to(device)
        # x is (x, y, z, vx, vy, vz, p)
        new_x = F.relu(self.conv1(x, edge_index))  # First GCN layer with ReLU
        new_x = self.conv2(new_x, edge_index)         # Output layer
        # new_x is (vx, vy, vz, p)
        new_x = torch.concat((x[:, :3], new_x), 1) # new_x is (x, y, z, vx, vy, vz, p)
        return new_x
    
    def train_model(self, loader, optimizer, epochs=100):
        self.to(device)
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
    
    def test_model(self, meshes):
        self.to(device)
        # test the model on a sequence of meshes
        # start from time t=2 and predict each step until the end
        # use the previous time steps to predict the next one
        # the final error is 1/T * sum_t sqrt(1/N * sum_n (x_t - x_t_pred)^2)

        # create the ground truth data
        edge_index, edge_attr = get_geometric_data(meshes[0])
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        x_list_truth = []
        for time_step in range(len(meshes)):
            node_features = np.hstack([
                meshes[time_step].points,
                meshes[time_step].point_data['Vitesse'],
                meshes[time_step].point_data['Pression'].reshape(-1, 1)
            ]) # Shape: (num_nodes, 7)
            node_features = torch.tensor(node_features, dtype=torch.float, device=device)
            x_list_truth.append(node_features)

        graph_data = Data(x=x_list_truth[1], edge_index=edge_index, edge_attr=edge_attr).to(device)

        # predict each time step
        total_error = 0
        list_errors = []
        for i in range(2, len(meshes)):
            x = self.forward(graph_data.x, graph_data.edge_index)
            v = x[:, 3:6]
            v_truth = x_list_truth[i][:, 3:6]
            
            error = F.mse_loss(v, v_truth)/len(meshes[0].points)
            list_errors.append(error.item())
            total_error += error.item()
            graph_data = Data(x=x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr).to(device)
        total_error /= len(meshes)
        print(f"Total error: {total_error:.4f}")
        return total_error, list_errors