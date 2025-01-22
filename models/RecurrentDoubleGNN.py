import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GatedGraphConv
import numpy as np
import tqdm

from mesh_handler import get_geometric_data

class RecurrentDoubleGNN(torch.nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=7):
        super(RecurrentDoubleGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_conv = GCNConv(input_dim, hidden_dim)
        self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)  # GRU for temporal info
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Output: vx, vy, vz, p

        # self.gnn = GatedGraphConv(out_channels=hidden_dim, num_layers=3)
        # self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, hidden_state):
        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, self.hidden_dim, device=x.device)

        # print(f"Initial x shape: {x.shape}, hidden_state shape: {hidden_state.shape}")
        x = self.gnn_conv(x, edge_index)
        x, hidden_state = self.rnn(x.unsqueeze(0), hidden_state)  # RNN step
        x = self.fc(x.squeeze(0))
        # print(f"Final x shape: {x.shape}, hidden_state shape: {hidden_state.shape}")
        return x, hidden_state
    
    def train_model(self, loader, optimizer, epochs=100):
        
        for epoch in tqdm.tqdm(range(epochs)):
            self.train()
            total_loss = 0
            hidden_state = None  # Reset hidden state at the start of each epoch

            for data_features, data_t in loader:
                optimizer.zero_grad()
                x_pred, hidden_state = self.forward(data_features.x, data_features.edge_index, hidden_state)
                hidden_state = hidden_state.detach()  # Detach hidden state to prevent backpropagating through time
                loss = F.mse_loss(x_pred, data_t.x)
                loss.backward(retain_graph=True)
                optimizer.step() 
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
    
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

        graph_data = Data(
            x=torch.cat((x_list_truth[0], x_list_truth[1]), 1),
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        # predict each time step
        total_error = 0
        list_errors = []
        hidden_state = None
        for i in range(2, len(meshes)):
            x_t_2 = graph_data.x[:, :self.input_dim//2]
            x_t_1 = graph_data.x[:, self.input_dim//2:]
            
            x, hidden_state = self.forward(graph_data.x, graph_data.edge_index, hidden_state=hidden_state)
            error = F.mse_loss(x, x_list_truth[i])/len(meshes[0].points)
            list_errors.append(error.item())
            total_error += error.item()

            graph_data = Data(
                x=torch.cat((x_t_1, x), 1),
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr
            )
        total_error /= len(meshes)
        print(f"Total error: {total_error:.4f}")
        return total_error, list_errors