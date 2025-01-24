import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import tqdm

from mesh_handler import get_geometric_data

class MeanPredictor(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=3):
        super(MeanPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean = None # [mean_vx, mean_vy, mean_vz]
    
    def train_model(self, loader, optimizer, epochs=100):
        # compute the mean of the velocities
        total_vx = 0
        total_vy = 0
        total_vz = 0
        total_points = 3
        for graph_data in loader[:3]: # use first 3 time steps
            total_vx += graph_data.y[:, 3].sum()
            total_vy += graph_data.y[:, 4].sum()
            total_vz += graph_data.y[:, 5].sum()
        self.mean = [total_vx/total_points, total_vy/total_points, total_vz/total_points]
        print(f"Mean velocities: {self.mean}")

    def forward(self, x, edge_index):
        # Return [mean_vx, mean_vy, mean_vz]
        print(f"Input shape: {x.shape}, Mean v shape: {len(self.mean)}")
        pred = torch.tensor([self.mean]*x.shape[0], dtype=torch.float)
        print(pred.shape)
        return pred
    
    def test_model(self, meshes):
        # test the model on a sequence of meshes
        # start from time t=2 and predict each step until the end
        # use the previous time steps to predict the next one
        # the final error is 1/T * sum_t sqrt(1/N * sum_n (x_t - x_t_pred)^2)

        # create the ground truth data
        edge_index, edge_attr = get_geometric_data(meshes[0])
        x_list_truth = []
        for time_step in range(len(meshes)):
            node_features = np.hstack([
                meshes[time_step].points,
                meshes[time_step].point_data['Vitesse'],
                meshes[time_step].point_data['Pression'].reshape(-1, 1)
            ]) # Shape: (num_nodes, 7)
            node_features = torch.tensor(node_features, dtype=torch.float)
            x_list_truth.append(node_features)

        graph_data = Data(x=x_list_truth[1], edge_index=edge_index, edge_attr=edge_attr)

        # predict each time step
        total_error = 0
        list_errors = []
        for i in range(2, len(meshes)):
            v = self.forward(graph_data.x, graph_data.edge_index)
            v_truth = x_list_truth[i][:, 3:6]
            
            error = F.mse_loss(v, v_truth)/len(meshes[0].points)
            list_errors.append(error.item())
            total_error += error.item()

        total_error /= len(meshes)
        print(f"Total error: {total_error:.4f}")
        return total_error, list_errors