import numpy as np
import torch
import torch.nn as nn

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, k=6):
        super().__init__()
        B = torch.randn(in_features, out_features // 2) * k
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = torch.matmul(2 * np.pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class SkipNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        decoder_hidden_dim=512,
        nl="relu", # relu or softplus
        skip_layers=[3],
        ff_layers=[0],
        decoder_n_hidden_layers=8,
        init_type="geometric_relu"
    ):
        super(SkipNet, self).__init__()
        
        self.init_type = init_type
        self.num_layers = decoder_n_hidden_layers

        
        if nl == "softplus":
            self.activation = nn.Softplus(beta=100)
        elif nl == "relu":
            self.activation = nn.ReLU()
        
        self.skip_layers = skip_layers
        self.ff_layers = ff_layers
        self.dims = [in_dim] + [decoder_hidden_dim] * (decoder_n_hidden_layers - 1) + [1]
            
        for l in range(0, self.num_layers):
            if l in ff_layers:
                layer = FourierLayer(self.dims[l], self.dims[l + 1])
            else:
                if l in skip_layers:
                    layer = nn.Linear(self.dims[l] + in_dim, self.dims[l + 1])
                else:
                    layer = nn.Linear(self.dims[l], self.dims[l + 1])
                    
                # set geometric initialization
                if init_type == "geometric_relu":
                    if l == self.num_layers - 1:
                        # skiplayer is not the last layer
                        torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(self.dims[l]), std=0.00001)
                        # 1.0 here is the constant
                        torch.nn.init.constant_(layer.bias, -1.0)
                    else:
                        torch.nn.init.constant_(layer.bias, 0.0)
                        torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.dims[l+1]))
                        
            # register layer
            setattr(self, "layer_{}".format(l), layer)
            
    def forward_pts(self, x):
        # Store initial input for skip connections
        x_input = x.clone()
        
        for l in range(self.num_layers):
            layer = getattr(self, "layer_{}".format(l))

            # Apply skip connection if it's a skip layer
            if l in self.skip_layers:
                x = torch.cat([x, x_input], dim=-1)

            # Pass through the layer
            x = layer(x)

            # Apply activation function unless it's the last layer
            if l < self.num_layers - 1:
                x = self.activation(x)
            
        return x

    def forward(self, non_mnfld_pnts, mnfld_pnts=None):
        manifold_pnts_pred = None
        nonmanifold_pnts_pred = None

        batch_size = non_mnfld_pnts.shape[0]

        if mnfld_pnts is not None:
            manifold_pnts_pred = self.forward_pts(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
        else:
            manifold_pnts_pred = None

        # Off manifold points
        nonmanifold_pnts_pred = self.forward_pts(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])).reshape(batch_size, -1)
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
        }