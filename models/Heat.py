import torch
import torch.nn as nn
import numpy as np

class Decoder(nn.Module):
    def __init__(self, radius_init=0.0):
        super(Decoder, self).__init__()
        self.radius_init = radius_init
        
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64 + 2, 64)  # 加入输入 (x, y)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 1)
        
        self.num_layers = 7
        # Not exactly correct because there is a 66 dimension input in the fourth layer
        self.dims = [2, 64, 64, 64, 64, 64, 64, 1]

        self.softplus = nn.Softplus(beta=100)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for layer, lin in enumerate([self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7]):
            in_dim = self.dims[layer]
            out_dim = self.dims[layer + 1]

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / 4 / np.sqrt(in_dim), std=0.00001)
                torch.nn.init.constant_(lin.bias, -0.1*self.radius_init)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.005, np.sqrt(2) / np.sqrt(out_dim))

    def forward(self, x):
        out1 = self.softplus(self.fc1(x))
        out2 = self.softplus(self.fc2(out1))
        out3 = self.softplus(self.fc3(out2))
        # 将原先的输入 (x, y) 连接到中间层
        out4 = torch.cat([out3, x], dim=1)
        out5 = self.softplus(self.fc4(out4))
        out6 = self.softplus(self.fc5(out5))
        out7 = self.softplus(self.fc6(out6))
        u = self.fc7(out7)
        return u

class Net(nn.Module):
    def __init__(self, radius_init=0.0):
        super(Net, self).__init__()
        self.radius_init = radius_init

        self.decoder = Decoder()

        self._initialize_weights()

    def _initialize_weights(self):
        self.decoder._initialize_weights()

    def forward(self, non_mnfld_pnts, mnfld_pnts):
        batch_size = non_mnfld_pnts.shape[0]

        nonmanifold_pnts_pred = self.decoder(
            non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])
        ).reshape(batch_size, -1)
        manifold_pnts_pred = self.decoder(
            mnfld_pnts.view(-1, mnfld_pnts.shape[-1])
        ).reshape(batch_size, -1)

        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
            "latent_reg": None,
            "latent": None,
        }