import numpy as np
import torch
import torch.nn as nn

class SkipNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        decoder_hidden_dim=256,
        nl="relu", # relu or softplus
        encoder_type=None,
        decoder_n_hidden_layers=8,
        init_type="siren",
        sphere_init_params=[1.6, 1.0],
    ):
        pass

    def forward(self, non_mnfld_pnts, mnfld_pnts=None):
        manifold_pnts_pred = None
        nonmanifold_pnts_pred = None

        batch_size = non_mnfld_pnts.shape[0]

        if mnfld_pnts is not None:
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(
                batch_size, -1
            )
        else:
            manifold_pnts_pred = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(
            non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])
        ).reshape(batch_size, -1)
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
        }