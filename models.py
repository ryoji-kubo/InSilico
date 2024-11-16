import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import RGCNConv, GINEConv, global_max_pool, global_mean_pool


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_blocks, num_node_features, edge_attr_dim, edge_attr_num_features):
        '''
        num_layers: the number of layers of the GNN
        hidden_channels: the hidden channel of the GNN
        num_node_feature: The input x will be of shape (num_nodes, num_node_features)
        edge_attr_dim: the number of edge attrs for 1 edge (3!)
        edge_attr_num_features: a list of the max value each dim of edge attr can take
        '''
        super().__init__()
        self.feature_emb = Parameter(torch.empty(num_node_features, hidden_channels))

        self.convolutions = []
        for i in range(num_layers):
            layer_wise_conv = []
            for j in range(edge_attr_dim):
                conv = RGCNConv(hidden_channels, hidden_channels, edge_attr_num_features[j],
                                num_blocks=num_blocks)
                layer_wise_conv.append(conv)
                
            self.convolutions.append(layer_wise_conv)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.feature_emb)
        for layer_wise_conv in self.convolutions:
            for conv in layer_wise_conv:
                conv.reset_parameters()

    def forward(self, data):

        # embed the node feature to hidden_dim
        x = self.feature_emb(data.x)

        # for each layer
        for i, layer_wise_conv in enumerate(self.convolutions):
            x_layer = []
            # for each edge attr
            for j, conv in enumerate(layer_wise_conv):
                # do convolution based on the specific edge attr
                hidden = conv(x, data.edge_index, data.edge_attr[:, j]).relu_()
                if i != len(self.convolutions)-1:
                    hidden = F.dropout(hidden, p=0.2, training=self.training)
                x_layer.append(hidden)
            x = torch.cat(x_layer)

        return x


class GINE(torch.nn.Module):
    """
    A GIN model using 3 layers of GIN
    """
    def __init__(self, num_feats, num_classes, edge_dim, use_sigmoid_last=False, hidden_channels=20):
        super().__init__()
        self.mlp_gin1 = torch.nn.Linear(num_feats, hidden_channels)
        self.gin1 = GINEConv(self.mlp_gin1, edge_dim=edge_dim)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINEConv(self.mlp_gin2, edge_dim=edge_dim)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin3 = GINEConv(self.mlp_gin3, edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden_channels*2, num_classes)
        self.use_sigmoid_last = use_sigmoid_last

    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_attr = data.edge_attr.to(torch.float32)
        x = self.gin1(x = x, edge_index = data.edge_index, edge_attr = edge_attr)
        x = x.relu()
        x = self.gin2(x = x, edge_index = data.edge_index, edge_attr = edge_attr)
        x = x.relu()
        x = self.gin3(x = x, edge_index = data.edge_index, edge_attr = edge_attr)
        x = x.relu()

        out1 = global_max_pool(x, data.batch)
        out2 = global_mean_pool(x, data.batch)

        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        
        if self.use_sigmoid_last:
            out = torch.sigmoid(out)
        return out


if __name__ == '__main__':
    print('complete')
