from collections import defaultdict
import numpy as np
import torch as th
import cv2
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.compile_expression import create_compiled_expr
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate
import coref.language as language
# from pytorch3d.loss import chamfer_distance


class ProgramEvalObject():
    def __init__(self, key, proposed_expressions, best_expression, best_metrics):
        self.key = key
        self.proposed_expressions = proposed_expressions
        self.best_expression = best_expression
        self.best_metrics = best_metrics  # iou, length, objective.


class EvalStatEstimator():

    def __init__(self, eval_config, lang_conf, dtype, device, objective="objective"):
        self.parsimony_factor = eval_config.parsimony_factor
        self.sdf_eval_bs = eval_config.sdf_eval_bs
        self.n_max_tokens = lang_conf.n_max_tokens
        self.n_float_bins = lang_conf.n_float_bins
        self.resolution = lang_conf.resolution
        self.n_dims = lang_conf.n_dims
        self.dtype = dtype
        self.device = device
        lang_specs = getattr(language, lang_conf.name)(lang_conf.n_float_bins,
                                                       n_integer_bins=lang_conf.n_integer_bins,
                                                       tokenization=lang_conf.tokenization)
        self.actions_to_expr = lang_specs.actions_to_expr
        self.revert_to_base_expr = lang_specs.revert_to_base_expr
        self.get_expression_size = lang_specs.get_expression_size
        self.null_expression = lang_specs.null_expression
        self.get_cmd_counts = lang_specs.get_cmd_counts
        self.action_mapper = lang_specs.create_action_mapper(
            device=self.device)
        self.create_action_spec = lang_specs.create_action_spec
        self.n_max_tokens = lang_conf.n_max_tokens
        self.sketcher = Sketcher(device=device, dtype=dtype,
                                 resolution=self.resolution,
                                 n_dims=self.n_dims)
        self.evaluated_objects = {}
        self.objective = objective
        self.selector_map = {
            "objective": self.get_objective_max,
            "loglikelihood": self.get_loglikelihood_max,
            'iou': self.get_iou_max,
            'chamfer': self.get_chamfer_min
        }
        self.val_metric_map = {
            "objective": "mean_objective",
            "loglikelihood": "mean_loglikelihood",
            'iou': "mean_iou",
            'chamfer': "mean_chamfer_neg"
        }

    def reset(self):
        self.evaluated_objects = {}

    def update_from_actions(self, indices, all_occs, pred_actions, objective=None, actions=True):

        if objective is None:
            objective = self.objective

        selector_func = self.selector_map[objective]
        pred_expressions = defaultdict(list)
        pred_loglikes = defaultdict(list)
        expr_objs = []
        expr_sizes = []
        expanded_occs = []
        index_pointers = dict()
        counter = 0
        if actions:
            conversion_function = self.actions_to_expr
        else:
            conversion_function = self.expr_to_expr
        for index, action_list in pred_actions.items():
            start_index = counter
            # convert actions to cpu
            log_likes = [actions[0] for actions in action_list]
            action_list = [actions[1] for actions in action_list]
            for inner_ind, actions in enumerate(action_list):
                try:
                    expression = conversion_function(
                        actions, self.action_mapper)
                    reverted_program = self.revert_to_base_expr(expression)
                    cache_expr = create_compiled_expr(
                        reverted_program, sketcher=self.sketcher)
                    # raise ValueError("Invalid action sequence.")
                    # /(actions.shape[0] - 2)
                    cur_loglike = log_likes[inner_ind]
                except:
                    expression = self.null_expression()
                    reverted_program = self.revert_to_base_expr(expression)
                    cache_expr = create_compiled_expr(
                        reverted_program, sketcher=self.sketcher)
                    cur_loglike = -np.inf
                expr_size = self.get_expression_size(expression)
                pred_expressions[index].append(expression)
                pred_loglikes[index].append(cur_loglike)
                expr_objs.append(cache_expr)
                expr_sizes.append(expr_size)
                counter += 1
            end_index = counter
            if end_index == start_index:
                # add a null expression
                null_expr = self.null_expression()
                pred_expressions[index].append(null_expr)
                pred_loglikes[index].append(-np.inf)
                reverted_program = self.revert_to_base_expr(null_expr)
                cache_expr = create_compiled_expr(
                    reverted_program, sketcher=self.sketcher)
                expr_objs.append(cache_expr)
                expr_size = self.get_expression_size(null_expr)
                expr_sizes.append(expr_size)
                counter += 1
                end_index = counter
            n_exprs = end_index - start_index
            cur_occ = all_occs[index:index +
                               1].bool().reshape(1, -1).expand(n_exprs, -1)
            expanded_occs.append(cur_occ)
            index_pointers[index] = [start_index, end_index]

        # now execute the expressions:

        all_target_occs = th.cat(expanded_occs, dim=0)
        # all_target_occs = all_target_occs.view(all_target_occs.shape[0], -1)

        n_chunks = np.ceil(len(expr_objs) / self.sdf_eval_bs).astype(np.int32)

        convert_to_cuda = True if self.device in [
            th.device("cuda"), "cuda"] else False
        all_pred_occs = []
        for ind in range(n_chunks):
            eval_stack = expr_objs[ind *
                                   self.sdf_eval_bs: (ind+1) * self.sdf_eval_bs]
            with th.no_grad():
                eval_batches = create_evaluation_batches(
                    eval_stack, convert_to_cuda=convert_to_cuda)
                all_sdfs = batch_evaluate(eval_batches, self.sketcher)
            all_pred_occs.append(all_sdfs <= 0)

        all_pred_occs = th.cat(all_pred_occs, dim=0)

        all_ious = (all_target_occs & all_pred_occs).sum(
            1) / (all_target_occs | all_pred_occs).sum(1)
        all_lens = th.tensor(expr_sizes, dtype=th.float32, device=self.device)
        # Measure CD if need be
        if self.n_dims == 2:

            n_samples = all_pred_occs.shape[0]
            all_pred_occs = all_pred_occs.reshape(
                -1, self.resolution, self.resolution).cpu().numpy()
            all_target_occs = all_target_occs.reshape(
                -1, self.resolution, self.resolution).cpu().numpy()
            all_chamfer = chamfer(all_pred_occs, all_target_occs)
            all_chamfer = th.tensor(
                all_chamfer, dtype=th.float32, device=self.device)
            all_objectives = (10 - all_chamfer) - \
                self.parsimony_factor * all_lens
        else:
            all_objectives = all_ious - self.parsimony_factor * all_lens
            all_chamfer = None

        for index in pred_actions.keys():
            start_index, end_index = index_pointers[index]
            best_index = selector_func(start_index=start_index, end_index=end_index,
                                       iou=all_ious, chamfer=all_chamfer,
                                       objective=all_objectives, loglikelihood=pred_loglikes[index])
            best_index_with_delta = start_index + best_index
            best_expression = pred_expressions[index][best_index]
            best_loglike = pred_loglikes[index][best_index]
            best_expression = best_expression.to(self.device)
            metrics = {
                "iou": all_ious[best_index_with_delta].item(),
                "length": all_lens[best_index_with_delta].item(),
                "objective": all_objectives[best_index_with_delta].item(),
                "loglikelihood": best_loglike
            }
            if self.n_dims == 2:
                # add chamfer
                metrics["chamfer"] = all_chamfer[best_index_with_delta].item()
                metrics["chamfer_neg"] = - \
                    all_chamfer[best_index_with_delta].item()
            real_id = indices[index]
            # if isinstance(real_id, th.Tensor):
            #     real_id = real_id.item()
            eval_obj = ProgramEvalObject(
                real_id, pred_expressions[index], best_expression, metrics)
            self.evaluated_objects[real_id] = eval_obj

    def expr_to_expr(self, expr, *args, **kwargs):
        return expr

    def get_objective_max(self, objective, start_index, end_index, *args, **kwargs):
        objective = objective[start_index:end_index]
        best_index = objective.argmax()
        if isinstance(best_index, th.Tensor):
            best_index = best_index.item()
        return best_index

    def get_loglikelihood_max(self, loglikelihood, *args, **kwargs):
        best_index = np.argmax(loglikelihood)
        return best_index

    def get_iou_max(self, iou, start_index, end_index, *args, **kwargs):
        iou = iou[start_index:end_index]
        best_index = iou.argmax()
        if isinstance(best_index, th.Tensor):
            best_index = best_index.item()
        return best_index

    def get_chamfer_min(self, chamfer, start_index, end_index, *args, **kwargs):
        chamfer = chamfer[start_index:end_index]
        best_index = chamfer.argmin()
        if isinstance(best_index, th.Tensor):
            best_index = best_index.item()
        return best_index

    def get_final_metrics(self):
        # get the average metrics across the evaluated objects.

        iou = []
        length = []
        objective = []
        loglikelihood = []
        final_metrics = dict()
        for key, eval_obj in self.evaluated_objects.items():
            iou.append(eval_obj.best_metrics["iou"])
            length.append(eval_obj.best_metrics["length"])
            objective.append(eval_obj.best_metrics["objective"])
            loglikelihood.append(eval_obj.best_metrics["loglikelihood"])

        final_metrics["mean_iou"] = np.nanmean(iou)
        final_metrics["mean_length"] = np.nanmean(length)
        final_metrics["mean_objective"] = np.nanmean(objective)
        final_metrics["mean_loglikelihood"] = np.nanmean(loglikelihood)
        final_metrics["median_iou"] = np.nanmedian(iou)
        final_metrics["median_length"] = np.nanmedian(length)
        final_metrics["median_objective"] = np.nanmedian(objective)
        final_metrics["median_loglikelihood"] = np.nanmedian(loglikelihood)
        val_metric = self.val_metric_map[self.objective]
        final_metrics['val_metric'] = float(final_metrics[val_metric])
        if self.n_dims == 2:
            cd = []
            inverted_cd = []
            for key, eval_obj in self.evaluated_objects.items():
                cd.append(eval_obj.best_metrics["chamfer"])
                inverted_cd.append(eval_obj.best_metrics["chamfer_neg"])
            final_metrics["mean_chamfer"] = np.nanmean(cd)
            final_metrics["median_chamfer"] = np.nanmedian(cd)
            final_metrics["mean_chamfer_neg"] = np.nanmean(inverted_cd)
            final_metrics["median_chamfer_neg"] = np.nanmedian(inverted_cd)

        return final_metrics

    def get_stats(self):

        # number of expressions
        # Language usage?
        stats = {}
        all_n_exprs = []
        all_n_unique_exprs = []
        cmd_usage = defaultdict(int)
        for key, eval_obj in self.evaluated_objects.items():
            n_exprs = len(eval_obj.proposed_expressions)
            n_unique_exprs = len(set(eval_obj.proposed_expressions))
            all_n_exprs.append(n_exprs)
            all_n_unique_exprs.append(n_unique_exprs)
            cmd_counts = self.get_cmd_counts(eval_obj.best_expression)
            for inner_key, value in cmd_counts.items():
                cmd_usage[inner_key] += value

        stats["n_exprs"] = np.mean(all_n_exprs)
        stats["n_unique_exprs"] = np.mean(all_n_unique_exprs)

        # TODO : add as ratio instead of absolute.
        stats.update(cmd_usage)

        return stats

    def get_best_programs(self, to_string=False,
                          return_score=False,
                          objective=None,
                          to_cpu=True,
                          to_sympy=False,
                          revert=False):
        if objective is None:
            objective = self.objective
        all_exprs = []
        for key, eval_obj in self.evaluated_objects.items():
            expr = eval_obj.best_expression
            eval_content = expr
            if to_cpu:
                eval_content = expr.cpu()
            if to_sympy:
                eval_content = expr.sympy()
            if revert:
                expr = self.revert_to_base_expr(expr)
            if to_string:
                eval_content = (key, str(eval_content))
            if return_score:
                eval_content = (key, eval_content,
                                eval_obj.best_metrics[objective])
            all_exprs.append(eval_content)
        return all_exprs

    def get_individual_objective(self, pred, target, prog_len):
        iou = (pred & target).sum() / (pred | target).sum()
        objective = iou - self.parsimony_factor * prog_len
        return objective

# TODO: write this function in torch and with batching.


def chamfer(images1: np.ndarray, images2: np.ndarray) -> np.ndarray:
    """Taken from:https://git.io/JfIpC"""
    # Convert in the opencv data format
    images1 = (images1 * 255).astype(np.uint8)
    images2 = (images2 * 255).astype(np.uint8)
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size ** 2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (
            (summ1[i] == 0)
            or (summ2[i] == 0)
            or (summ1[i] == filled_value)
            or (summ2[i] == filled_value)
        ):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3
        )

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3
        )
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (np.sum(E2, (1, 2)) + 1) + np.sum(
        D2 * E1, (1, 2)
    ) / (np.sum(E1, (1, 2)) + 1)
    # TODO make it simpler
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return distances
