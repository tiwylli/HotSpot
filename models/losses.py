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
    nonmnfld_grad, mnfld_grad, nonmnfld_pdfs=None, eikonal_type="abs", sigma_min=0.8
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


class Loss(nn.Module):
    def __init__(
        self,
        weights,
        loss_type,
        importance_sampling=True,
    ):
        super().__init__()
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.importance_sampling = importance_sampling
        self.loss_term_dict = {}
        self.var_dict = {}

        self.loss_type_to_terms = {
            "siren": ["manifold", "area", "normal", "eikonal"],
            "siren_wo_n": ["manifold", "area", "eikonal"],
            "igr": ["manifold", "normal", "eikonal"],
            "igr_wo_n": ["manifold", "eikonal"],
            "siren_w_div": ["manifold", "area", "normal", "eikonal", "div"],
            "siren_wo_n_w_div": ["manifold", "area", "eikonal", "div"],
            "igr_wo_eik_w_heat": ["manifold", "heat"],
            "igr_w_heat": ["manifold", "eikonal", "heat"],
            "sal": ["manifold", "sal"],
            "pull": ["pull"],
            "nsh": ["manifold", "area", "relax_eikonal", "singular_hessian"],
            "cad": ["manifold", "area", "eikonal", "cad"],
            "everything_including_div_heat_sal": [
                "manifold",
                "area",
                "normal",
                "eikonal",
                "div",
                "sal",
                "heat",
            ],
        }
        if self.loss_type not in self.loss_type_to_terms:
            raise Warning("unsupported loss type")

    def register_loss_term(self, name, idx, weight, schedule_type=None, schedule_params=None):
        self.loss_term_dict[name] = {
            "idx": idx,
            "weight": weight,
            "schedule_type": schedule_type,
            "schedule_params": schedule_params,
        }

    def register_variable(self, name, value, schedule_type=None, schedule_params=None):
        self.var_dict[name] = {
            "value": value,
            "schedule_type": schedule_type,
            "schedule_params": schedule_params,
        }
        # create a variable with the name and value
        setattr(self, name, value)

    def _update_hyper_parameter(
        self, param_name: str, current_iteration, n_iterations, params, schedule_type
    ):
        if schedule_type == "none":
            return

        if not hasattr(self, f"{param_name}_decay_params_list"):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            setattr(
                self,
                f"{param_name}_decay_params_list",
                list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1])),
            )

        decay_params_list = getattr(self, f"{param_name}_decay_params_list")
        curr = current_iteration / n_iterations
        we, e = min([tup for tup in decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        if schedule_type == "linear":
            if current_iteration < s * n_iterations:
                value = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                value = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                value = we
        elif schedule_type == "quintic":
            if current_iteration < s * n_iterations:
                value = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                value = w0 + (we - w0) * (
                    1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5
                )
            else:
                value = we
        elif schedule_type == "step":
            if current_iteration < s * n_iterations:
                value = w0
            else:
                value = we
        else:
            raise Warning("unsupported decay value")

        if param_name in self.loss_term_dict:
            idx = self.loss_term_dict[param_name]["idx"]
            self.weights[idx] = value
        elif param_name in self.var_dict:
            setattr(self, param_name, value)
        else:
            raise Warning("variable requested to update not found")

    def update_all_hyper_parameters(self, current_iteration, n_iterations):
        for key, value in self.loss_term_dict.items():
            if value["schedule_type"] is None or self.weights[value["idx"]] == 0:
                continue
            self._update_hyper_parameter(
                key,
                current_iteration,
                n_iterations,
                value["schedule_params"],
                value["schedule_type"],
            )

        for key, value in self.var_dict.items():
            if value["schedule_type"] is None:
                continue
            self._update_hyper_parameter(
                key,
                current_iteration,
                n_iterations,
                value["schedule_params"],
                value["schedule_type"],
            )

    def forward(
        self,
        output_pred,
        mnfld_points,
        nonmnfld_points,
        nonmnfld_pdfs=None,
        mnfld_normals_gt=None,
        nonmnfld_dists_gt=None,
        nonmnfld_dists_sal=None,
        nearest_points=None, # (bs, n_nonmnfld_points, 3)
    ):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        nonmnfld_pred = output_pred["nonmanifold_pnts_pred"]
        mnfld_pred = output_pred["manifold_pnts_pred"]

        nonmnfld_sdf_pred = nonmnfld_pred
        mnfld_sdf_pred = mnfld_pred

        # If nonmnfld_dist_pred has nan or inf, print and exit
        if torch.isnan(nonmnfld_sdf_pred).any():
            raise ValueError("NaN in nonmnfld_dist_pred")
        if torch.isinf(nonmnfld_sdf_pred).any():
            raise ValueError("Inf in nonmnfld_dist_pred")

        # Compute gradients for div (divergence), curl and curv (curvature)
        if mnfld_sdf_pred is not None:
            mnfld_grad_pred = utils.gradient(mnfld_points, mnfld_sdf_pred)
        else:
            mnfld_grad_pred = None

        nonmnfld_grad_pred = utils.gradient(nonmnfld_points, nonmnfld_sdf_pred)

        # If mnfld_grad or nonmnfld_grad is nan, print and exit
        if torch.isnan(mnfld_grad_pred).any():
            print("mnfld_grad", mnfld_grad_pred)
            raise ValueError("NaN in mnfld gradients")
        if torch.isnan(nonmnfld_grad_pred).any():
            print("nonmnfld_grad", nonmnfld_grad_pred)
            raise ValueError("NaN in nonmnfld gradients")

        # Now we have
        # mnfld_sdf_pred (bs, n_mnfld_points, dim)
        # nonmnfld_sdf_pred (bs, n_nonmnfld_points, dim)
        # mnfld_grad_pred (bs, n_mnfld_points)
        # nonmnfld_grad_pred (bs, n_nonmnfld_points)

        # ===============================
        # Start to compute the loss terms
        # ===============================

        loss_terms = {}

        # manifold loss
        if "manifold" in self.loss_type_to_terms[self.loss_type]:
            manifold_loss = torch.abs(mnfld_pred).mean()
            loss_terms["manifold"] = manifold_loss

        # area loss
        if "area" in self.loss_type_to_terms[self.loss_type]:
            area_loss = torch.exp(-1e2 * torch.abs(nonmnfld_sdf_pred)).mean()
            loss_terms["area"] = area_loss

        # normal loss
        if "normal" in self.loss_type_to_terms[self.loss_type] and mnfld_normals_gt is not None:
            mnfld_normals_gt = mnfld_normals_gt.to(device)
            if "igr" in self.loss_type or "phase" in self.loss_type:
                normal_loss = ((mnfld_grad_pred - mnfld_normals_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_loss = (
                    1
                    - torch.abs(
                        torch.nn.functional.cosine_similarity(mnfld_grad_pred, mnfld_normals_gt, dim=-1)
                    )
                ).mean()
            loss_terms["normal"] = normal_loss

        # eikonal loss
        if "eikonal" in self.loss_type_to_terms[self.loss_type]:
            eikonal_term = eikonal_loss(
                nonmnfld_grad_pred,
                mnfld_grad=mnfld_grad_pred,
                nonmnfld_pdfs=nonmnfld_pdfs,
                eikonal_type="abs" if self.loss_type != "phase" else "squared",
            )
            loss_terms["eikonal"] = eikonal_term

        # divergene loss
        if "div" in self.loss_type_to_terms[self.loss_type]:
            if self.div_type == "full_l2":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad_pred)
                nonmnfld_divergence_term = torch.clamp(torch.square(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == "full_l1":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad_pred)
                nonmnfld_divergence_term = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == "dir_l2":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad_pred)
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence)
            elif self.div_type == "dir_l1":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad_pred)
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence)
            else:
                raise Warning(
                    "unsupported divergence type. only suuports dir_l1, dir_l2, full_l1, full_l2"
                )

            div_loss = nonmnfld_divergence_term.mean()  # + mnfld_divergence_term.mean()
            loss_terms["div"] = div_loss

        # SAL loss
        if "sal" in self.loss_type_to_terms[self.loss_type] and nonmnfld_dists_sal is not None:
            sal_loss = torch.abs(
                torch.abs(nonmnfld_pred.squeeze()) - nonmnfld_dists_sal.squeeze()
            ).mean()
            loss_terms["sal"] = sal_loss

        # heat loss
        if "heat" in self.loss_type_to_terms[self.loss_type]:
            heat_term = heat_loss(
                points=nonmnfld_points,
                preds=nonmnfld_sdf_pred,
                grads=nonmnfld_grad_pred,
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
            loss_terms["heat"] = heat_term

        # pulling loss
        if "pull" in self.loss_type_to_terms[self.loss_type]:
            pulled_locations = nonmnfld_points - nonmnfld_sdf_pred[..., None] * nonmnfld_grad_pred / torch.norm(nonmnfld_grad_pred, dim=-1, keepdim=True)
            pull_term = torch.mean(
                torch.norm(pulled_locations - nearest_points, dim=-1)
            )
            loss_terms["pull"] = pull_term

        if "relax_eikonal" in self.loss_type_to_terms[self.loss_type]:
            relax_eikonal_term = relaxing_eikonal_loss(
                nonmnfld_grad_pred, mnfld_grad_pred
            )
            loss_terms["relax_eikonal"] = relax_eikonal_term

        if "singular_hessian" in self.loss_type_to_terms[self.loss_type]:
            singular_hessian_term = singular_hessian_loss(
                mnfld_points, nonmnfld_points, mnfld_grad_pred, nonmnfld_grad_pred
            )
            loss_terms["singular_hessian"] = singular_hessian_term

        if "cad" in self.loss_type_to_terms[self.loss_type]:
            cad_term = cad_loss(
                mnfld_points, nonmnfld_points, mnfld_grad_pred, nonmnfld_grad_pred
            )
            loss_terms["cad"] = cad_term

        # ===============================
        # Compute the total loss
        # ===============================

        loss = torch.tensor([0.0], device=device)
        loss_terms_weighted = {}
        for name, value in loss_terms.items():
            weight = self.weights[self.loss_term_dict[name]["idx"]]
            if weight == 0:
                continue
            loss_terms_weighted[name] = weight * loss_terms[name]

        for name, value in loss_terms_weighted.items():
            loss += value

        return {
            "loss": loss,
            "loss_terms": loss_terms,
            "loss_terms_weighted": loss_terms_weighted,
        }, mnfld_grad_pred
