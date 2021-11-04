import torch
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.nn import MessagePassing, GCNConv, NNConv, GINEConv
from torch_geometric.utils import degree



### A basic MLP    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x
        


class GINENet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden, dropout, num_tasks):
        super(GINENet, self).__init__()
        self.conv1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(num_node_features, hidden)),eps = 0, train_eps = True)
        self.conv2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden, hidden)),eps = 0, train_eps = True)
        self.conv3 = GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden, hidden)),eps = 0, train_eps = True)
        self.lin1 = torch.nn.Linear(num_edge_features, num_node_features, bias = True)
        self.lin2 = torch.nn.Linear(num_node_features, hidden, bias = True)
        self.dropout = dropout
        self.mlp1 = MLP(hidden, hidden, num_tasks, dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.mlp1.reset_parameters()

    def forward(self, batch_data):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

        edge_attr = self.lin1(edge_attr.float())
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_attr = self.lin2(edge_attr.float())
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = self.conv3(x, edge_index, edge_attr)
        out = scatter_mean(x, batch_data.batch, dim=0)  
        out = self.mlp1(out)   #[batch_szie, num_classes]

        return out


class GraphSizeNorm(torch.nn.Module):
    """Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    "Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>
    """
    def __init__(self):
        super(GraphSizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1)
    


class BatchNorm(torch.nn.BatchNorm1d):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        return super(BatchNorm, self).forward(x)


    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)


### MPNN + GraphSizeNorm + BatchNorm   
class NNNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden, dropout, num_tasks):
        super(NNNet, self).__init__()
        self.conv1 = NNConv(num_node_features, hidden, torch.nn.Sequential(torch.nn.Linear(num_edge_features, num_node_features*hidden)))
        self.conv2 = NNConv(hidden, hidden, torch.nn.Sequential(torch.nn.Linear(num_edge_features, hidden*hidden)))
        self.conv3 = NNConv(hidden, hidden, torch.nn.Sequential(torch.nn.Linear(num_edge_features, hidden*hidden)))
        self.mlp1 = MLP(hidden, hidden, num_tasks, dropout)
        self.dropout = dropout
        self.norm1 = GraphSizeNorm()
        self.bn1 = BatchNorm(hidden)
        self.norm2 = GraphSizeNorm()
        self.bn2 = BatchNorm(hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.mlp1.reset_parameters()

    def forward(self, batch_data):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x, batch_data.batch)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x, batch_data.batch)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        out = scatter_mean(x, batch_data.batch, dim=0)  
        out = self.mlp1(out)    #[batch_size, num_tasks]
        return out
    
    
