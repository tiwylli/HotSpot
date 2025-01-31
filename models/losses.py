# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils


def eikonal_loss(nonmnfld_grad, mnfld_grad, nonmnfld_pdfs, eikonal_type="abs"):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == "abs":
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term


def relaxing_eikonal_loss(
    nonmnfld_grad, mnfld_grad, nonmnfld_pdfs, eikonal_type="abs", sigma_min=0.8
):
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    grad_norm = all_grads.norm(2, dim=-1) + 1e-12
    term = torch.relu(-(grad_norm - sigma_min))
    if eikonal_type == "abs":
        eikonal_term = term.abs().mean()
    else:
        eikonal_term = term.square().mean()
    return eikonal_term
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    eikonal_term = torch.relu(-all_grads.norm(2, dim=2) - sigma_min).mean()

    return eikonal_term


def singular_hessian_loss(mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad):
    nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
    nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
    mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
    mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])

    # if dims == 3:
    nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
    nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

    mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
    mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)

    nonmnfld_det = torch.det(nonmnfld_hessian_term)
    mnfld_det = torch.det(mnfld_hessian_term)

    morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
    morse_nonmnfld = torch.tensor([0.0], device=mnfld_points.device)
    # if div_type == 'l1':
    morse_nonmnfld = nonmnfld_det.abs().mean()
    morse_mnfld = mnfld_det.abs().mean()

    morse_loss = 0.5 * (morse_nonmnfld + morse_mnfld)

    return morse_loss


def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
    device = morse_nonmnfld_grad.device
    nonmnfld_hessian_term = torch.cat(
        (nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1
    )
    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1), device=device
    )
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
    morse_nonmnfld = (-1.0 / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
        nonmnfld_hessian_term
    )

    morse_nonmnfld = morse_nonmnfld.abs()

    curvature = morse_nonmnfld.mean()

    return curvature


def cad_loss(mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad):
    nonmnfld_dx = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 0])
    nonmnfld_dy = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 1])
    mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
    mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])

    # if dims == 3:
    nonmnfld_dz = utils.gradient(nonmnfld_points, nonmnfld_grad[:, :, 2])
    nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

    mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
    mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)

    morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
    morse_loss = gaussian_curvature(nonmnfld_hessian_term, nonmnfld_grad)
    # if bidirectional_morse == True:
    # morse_mnfld = gaussian_curvature(mnfld_hessian_term, mnfld_grad)

    morse_loss = 0.5 * (morse_loss + morse_mnfld)

    return morse_loss


def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss


def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = 0.5 * torch.autograd.grad(dot_grad, points, hvp, retain_graph=True, create_graph=True)[0]
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads**2, dim=-1) + 1e-5)
    return div


def full_div(points, grads):
    dx = utils.gradient(points, grads[:, :, 0])
    dy = utils.gradient(points, grads[:, :, 1])
    if points.shape[-1] == 3:
        dz = utils.gradient(points, grads[:, :, 2])
        div = dx[:, :, 0] + dy[:, :, 1] + dz[:, :, 2]
    else:
        div = dx[:, :, 0] + dy[:, :, 1]
    div[div.isnan()] = 0
    return div


def heat_loss(points, preds, grads=None, sample_pdfs=None, heat_lambda=8, in_mnfld=False):
    if grads is None:
        grads = torch.autograd.grad(
            outputs=preds,
            inputs=points,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
        )[0]
    heat = torch.exp(-heat_lambda * preds.abs())
    if not in_mnfld:
        loss = 0.5 * heat**2 * (grads.norm(2, dim=-1) ** 2 + 1)
    else:
        loss = (0.5 * heat**2 * (grads.norm(2, dim=-1) ** 2 + 1)) - heat
    if sample_pdfs is not None:
        sample_pdfs = sample_pdfs.squeeze(-1)
        loss /= sample_pdfs
    loss = loss.sum()

    return loss


def phase_loss(points, preds, sample_pdfs=None, epsilon=0.01):
    grads = torch.autograd.grad(
        outputs=preds,
        inputs=points,
        grad_outputs=torch.ones_like(preds),
        create_graph=True,
        retain_graph=True,
    )[0]
    loss = epsilon * grads.norm(2, dim=-1) ** 2 + preds**2 - 2 * torch.abs(preds) + 1
    loss = loss.mean()

    return loss


class Loss(nn.Module):
    def __init__(
        self,
        weights,
        loss_type,
        div_decay="none",
        div_type="dir_l1",
        heat_lambda=100,
        phase_epsilon=0.01,
        heat_decay="none",
        heat_lambda_decay="none",
        eikonal_decay="none",
        boundary_coef_decay="none",
        importance_sampling=True,
        morse_decay="none",
        cad_decay="none",
    ):
        super().__init__()
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.heat_lambda = heat_lambda
        self.phase_epsilon = phase_epsilon
        self.heat_decay = heat_decay
        self.eikonal_decay = eikonal_decay
        self.heat_lambda_decay = heat_lambda_decay
        self.boundary_coef_decay = boundary_coef_decay
        self.use_div = True if "div" in self.loss_type else False
        self.use_heat = True if "heat" in self.loss_type else False
        self.use_phase = True if "phase" in self.loss_type else False
        self.importance_sampling = importance_sampling
        self.morse_decay = morse_decay
        self.cad_decay = cad_decay

    def forward(
        self,
        output_pred,
        mnfld_points,
        nonmnfld_points,
        nonmnfld_pdfs=None,
        mnfld_normals_gt=None,
        nonmnfld_dists_gt=None,
        nonmnfld_dists_sal=None,
    ):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        nonmnfld_pred = output_pred["nonmanifold_pnts_pred"]
        mnfld_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred.get("latent_reg", None)
        latent = output_pred.get("latent", None)

        div_term = torch.tensor([0.0], device=mnfld_points.device)

        if self.use_phase:
            nonmnfld_dist_pred = (
                -(self.phase_epsilon**0.5)
                * torch.log(1 - torch.abs(nonmnfld_pred))
                * torch.sign(nonmnfld_pred)
            )
        else:
            nonmnfld_dist_pred = nonmnfld_pred

        # if nonmnfld_dist_pred has nan or inf, print and exit
        if torch.isnan(nonmnfld_dist_pred).any():
            raise ValueError("NaN in nonmnfld_dist_pred")
        if torch.isinf(nonmnfld_dist_pred).any():
            raise ValueError("Inf in nonmnfld_dist_pred")

        # compute gradients for div (divergence), curl and curv (curvature)
        if mnfld_pred is not None:
            if self.use_phase:
                mnfld_dist_pred = (
                    -(self.phase_epsilon**0.5)
                    * torch.log(1 - torch.abs(mnfld_pred))
                    * torch.sign(mnfld_pred)
                )
            else:
                mnfld_dist_pred = mnfld_pred
            mnfld_grad = utils.gradient(mnfld_points, mnfld_dist_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, nonmnfld_dist_pred)

        # if mnfld_dist_pred or nonmnfld_dist_pred is nan, print and exit
        if torch.isnan(mnfld_grad).any():
            print("mnfld_grad", mnfld_grad)
            raise ValueError("NaN in mnfld gradients")
        if torch.isnan(nonmnfld_grad).any():
            print("nonmnfld_grad", nonmnfld_grad)
            raise ValueError("NaN in nonmnfld gradients")

        # div_term
        if self.use_div and self.weights[4] > 0.0:
            if self.div_type == "full_l2":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(torch.square(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == "full_l1":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == "dir_l2":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence)
            elif self.div_type == "dir_l1":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence)
            else:
                raise Warning(
                    "unsupported divergence type. only suuports dir_l1, dir_l2, full_l1, full_l2"
                )

            div_term = nonmnfld_divergence_term.mean()  # + mnfld_divergence_term.mean()

        # eikonal term
        eikonal_term = torch.tensor([0.0], device=mnfld_points.device)
        if self.weights[3] > 0.0:
            if self.loss_type == "nsh":
                eikonal_term = relaxing_eikonal_loss(
                    nonmnfld_grad,
                    mnfld_grad=mnfld_grad,
                    nonmnfld_pdfs=nonmnfld_pdfs,
                )
            else:
                eikonal_term = eikonal_loss(
                    nonmnfld_grad,
                    mnfld_grad=mnfld_grad,
                    nonmnfld_pdfs=nonmnfld_pdfs,
                    eikonal_type="abs" if self.loss_type != "phase" else "squared",
                )

        singular_hessian_term = torch.tensor([0.0], device=mnfld_points.device)
        if len(self.weights) > 8 and self.weights[8] > 0.0:
            singular_hessian_term = singular_hessian_loss(
                mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad
            )

        cad_term = torch.tensor([0.0], device=mnfld_points.device)
        if len(self.weights) > 9 and self.weights[9] > 0.0:
            cad_term = cad_loss(mnfld_points, nonmnfld_points, mnfld_grad, nonmnfld_grad)

        # normal term
        normal_term = torch.tensor([0.0], device=mnfld_points.device)
        if mnfld_normals_gt is not None and self.weights[2] > 0.0:
            mnfld_normals_gt = mnfld_normals_gt.to(mnfld_points.device)
            if "igr" in self.loss_type or "phase" in self.loss_type:
                normal_term = ((mnfld_grad - mnfld_normals_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_term = (
                    1
                    - torch.abs(
                        torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_normals_gt, dim=-1)
                    )
                ).mean()

        # signed distance function term
        boundary_term = torch.abs(mnfld_pred).mean()

        # inter term
        inter_term = torch.tensor([0.0], device=mnfld_points.device)
        if self.weights[1] > 0.0:
            inter_term = torch.exp(-1e2 * torch.abs(nonmnfld_dist_pred)).mean()

        # heat term
        heat_term = torch.tensor([0.0], device=mnfld_points.device)
        if self.use_heat and self.weights[6] > 0.0:
            heat_term = heat_loss(
                points=nonmnfld_points,
                preds=nonmnfld_dist_pred,
                grads=nonmnfld_grad,
                sample_pdfs=nonmnfld_pdfs if self.importance_sampling else None,
                heat_lambda=self.heat_lambda,
                in_mnfld=False,
            )
            # + heat_loss(
            #     points=mnfld_points,
            #     preds=manifold_pred,
            #     grads=mnfld_grad,
            #     sample_pdfs=None,
            #     heat_lambda=self.heat_lambda,
            #     in_mnfld=True,
            # )
            heat_term /= nonmnfld_points.reshape(-1, 2).shape[0]
            # heat_term /= nonmnfld_points.reshape(-1, 2).shape[0] + mnfld_points.reshape(-1, 2).shape[0]

        # phase term
        phase_term = torch.tensor([0.0], device=mnfld_points.device)
        if self.use_phase and self.weights[7] > 0.0:
            phase_term = phase_loss(
                points=mnfld_points, preds=mnfld_pred, sample_pdfs=None, epsilon=self.phase_epsilon
            )

        # nonmanifold prediction value loss
        nonmnfld_dists_loss = torch.tensor([0.0], device=mnfld_points.device)
        if nonmnfld_dists_gt is not None:
            nonmnfld_dists_loss = torch.abs(
                nonmnfld_pred.squeeze() - nonmnfld_dists_gt.squeeze()
            ).mean()

        # SAL loss term
        sal_term = torch.tensor([0.0], device=mnfld_points.device)
        if nonmnfld_dists_sal is not None:
            sal_term = torch.abs(
                torch.abs(nonmnfld_pred.squeeze()) - nonmnfld_dists_sal.squeeze()
            ).mean()

        #########################################
        # Losses
        #########################################

        # losses used in the paper
        if self.loss_type == "siren":  # SIREN loss
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[2] * normal_term
                + self.weights[3] * eikonal_term
            )
        elif self.loss_type == "siren_wo_n":  # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[3] * eikonal_term
            )
        elif self.loss_type == "igr":  # IGR loss
            self.weights[1] = 0
            loss = (
                self.weights[0] * boundary_term
                + self.weights[2] * normal_term
                + self.weights[3] * eikonal_term
            )
        elif self.loss_type == "igr_wo_n":  # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0] * boundary_term + self.weights[3] * eikonal_term
        elif self.loss_type == "siren_w_div":  # SIREN loss with divergence term
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[2] * normal_term
                + self.weights[3] * eikonal_term
                + self.weights[4] * div_term
            )
        elif (
            self.loss_type == "siren_wo_n_w_div"
        ):  # SIREN loss without normals and with divergence constraint
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[3] * eikonal_term
                + self.weights[4] * div_term
            )
        elif self.loss_type == "igr_wo_eik_w_heat":
            loss = (
                self.weights[0] * boundary_term
                # + self.weights[3] * eikonal_term
                + self.weights[6] * heat_term
            )
        elif self.loss_type == "igr_w_heat":
            loss = (
                self.weights[0] * boundary_term
                + self.weights[3] * eikonal_term
                + self.weights[6] * heat_term
            )
        elif self.loss_type == "sal":
            loss = self.weights[0] * boundary_term + self.weights[5] * sal_term
        elif self.loss_type == "phase":
            loss = (
                self.weights[0] * boundary_term
                + self.weights[2] * normal_term
                + self.weights[3] * eikonal_term
                + self.weights[7] * phase_term
            )
        elif self.loss_type == "nsh":
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[3] * eikonal_term
                + self.weights[8] * singular_hessian_term
            )
        elif self.loss_type == "cad":
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[3] * eikonal_term
                + self.weights[9] * cad_term
            )
        elif self.loss_type == "everything_including_div_heat_sal":
            loss = (
                self.weights[0] * boundary_term
                + self.weights[1] * inter_term
                + self.weights[2] * normal_term
                + self.weights[3] * eikonal_term
                + self.weights[4] * div_term
                + self.weights[5] * sal_term
                + self.weights[6] * heat_term
            )
        else:
            raise Warning("unrecognized loss type")

        # latent regulariation for multiple shape learning
        latent_reg_term = torch.tensor([0.0], device=mnfld_points.device)
        if latent is not None and latent_reg is not None:
            latent_reg_term = latent_rg_loss(latent_reg, device)
            # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
            loss += self.weights[5] * latent_reg_term

        return {
            "loss": loss,
            "boundary_term": boundary_term,
            "inter_term": inter_term,
            "latent_reg_term": latent_reg_term,
            "eikonal_term": eikonal_term,
            "normal_term": normal_term,
            "div_term": div_term,
            "sal_term": sal_term,
            "heat_term": heat_term,
            "diff_term": nonmnfld_dists_loss,
        }, mnfld_grad

    def update_weight(self, weight_index, current_iteration, n_iterations, params, decay_type):
        if not hasattr(self, f"{weight_index}_decay_params_list"):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            setattr(
                self,
                f"weight[{weight_index}]_decay_params_list",
                list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1])),
            )

        decay_params_list = getattr(self, f"weight[{weight_index}]_decay_params_list")
        curr = current_iteration / n_iterations
        we, e = min([tup for tup in decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        if decay_type == "linear":
            if current_iteration < s * n_iterations:
                self.weights[weight_index] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[weight_index] = w0 + (we - w0) * (
                    current_iteration / n_iterations - s
                ) / (e - s)
            else:
                self.weights[weight_index] = we
        elif decay_type == "quintic":
            if current_iteration < s * n_iterations:
                self.weights[weight_index] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[weight_index] = w0 + (we - w0) * (
                    1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5
                )
            else:
                self.weights[weight_index] = we
        elif decay_type == "step":
            if current_iteration < s * n_iterations:
                self.weights[weight_index] = w0
            else:
                self.weights[weight_index] = we
        elif decay_type == "none":
            pass
        else:
            raise Warning("unsupported decay value")

    def update_div_weight(self, current_iteration, n_iterations, params=None):
        self.update_weight(4, current_iteration, n_iterations, params, self.div_decay)

    def update_heat_weight(self, current_iteration, n_iterations, params=None):
        self.update_weight(6, current_iteration, n_iterations, params, self.heat_decay)

    def update_eikonal_weight(self, current_iteration, n_iterations, params=None):
        self.update_weight(3, current_iteration, n_iterations, params, self.eikonal_decay)

    def update_boundary_coef(self, current_iteration, n_iterations, params=None):
        self.update_weight(0, current_iteration, n_iterations, params, self.boundary_coef_decay)

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        self.update_weight(8, current_iteration, n_iterations, params, self.morse_decay)

    def update_cad_weight(self, current_iteration, n_iterations, params=None):
        self.update_weight(9, current_iteration, n_iterations, params, self.cad_decay)
