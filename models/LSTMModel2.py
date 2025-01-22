import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import tqdm

from mesh_handler import get_geometric_data

class LSTMModel2(torch.nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=3):
        super(LSTMModel2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_conv = GCNConv(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)  # LSTM for temporal info
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Output: vx, vy, vz

    def forward(self, x, edge_index, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(1, 1, self.hidden_dim, device=x.device)
            cell_state = torch.zeros(1, 1, self.hidden_dim, device=x.device)

        # GNN step
        x = self.gnn_conv(x, edge_index)  # Shape: (num_nodes, hidden_dim)

        # LSTM step
        x, (hidden_state, cell_state) = self.lstm(x.unsqueeze(0), (hidden_state, cell_state))  # Shape: (1, num_nodes, hidden_dim)

        # Fully connected layer for prediction
        x = self.fc(x.squeeze(0))  # Shape: (num_nodes, output_dim)
        return x, hidden_state, cell_state

    def train_model(self, loader, optimizer, epochs=100):
        for epoch in tqdm.tqdm(range(epochs)):
            self.train()
            total_loss = 0
            hidden_state, cell_state = None, None  # Reset states at the start of each epoch

            for graph_data in loader:
                optimizer.zero_grad()
                x_pred, hidden_state, cell_state = self.forward(graph_data.x, graph_data.edge_index, hidden_state, cell_state)
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()
                loss = F.mse_loss(x_pred, graph_data.y[:, :3])  # Use only vx, vy, vz for target
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    def test_model(self, meshes):
        # Préparer les données de vérité terrain
        edge_index, edge_attr = get_geometric_data(meshes[0])
        x_list_truth = []
        for time_step in range(len(meshes)):
            node_features = np.hstack([
                meshes[time_step].points,  # Positions (x, y, z)
                meshes[time_step].point_data['Vitesse'],  # Vitesses (vx, vy, vz)
                meshes[time_step].point_data['Pression'].reshape(-1, 1)  # Pression
            ])  # Shape: (num_nodes, 7)
            node_features = torch.tensor(node_features, dtype=torch.float)
            x_list_truth.append(node_features)
    
        # Vérification des dimensions initiales
        print(f"x_list_truth[0].shape = {x_list_truth[0].shape}")
    
        # Initialiser les données pour la prédiction
        graph_data = Data(x=x_list_truth[1][:, :7], edge_index=edge_index, edge_attr=edge_attr)
        print(f"Initial graph_data.x shape: {graph_data.x.shape}")
    
        # Prédire chaque étape temporelle
        total_error = 0
        list_errors = []
        hidden_state, cell_state = None, None
        for i in range(2, len(meshes)):
            #print(f"Step {i}: graph_data.x.shape = {graph_data.x.shape}")
            x_pred, hidden_state, cell_state = self.forward(graph_data.x, graph_data.edge_index, hidden_state, cell_state)
            #print(f"Step {i}: x_pred.shape = {x_pred.shape}")
        
            # Calcul de l'erreur
            error = F.mse_loss(x_pred, x_list_truth[i][:, :3]) / len(meshes[0].points)
            list_errors.append(error.item())
            total_error += error.item()
            # Reconstruire les 7 caractéristiques pour la prochaine étape#####pbl de dimention 
            graph_data.x = torch.cat([
                torch.tensor(meshes[0].points, dtype=torch.float, device=x_pred.device),  # Les positions restent inchangées
                x_pred,  # Les vitesses prédites
                graph_data.x[:, 6:7]  # La pression (dernière colonne de l'entrée précédente)
            ], dim=1)

        total_error /= len(meshes)
        print(f"Total error: {total_error:.4f}")
        return total_error, list_errors
