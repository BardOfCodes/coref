import time
import torch as th
import numpy as np
from collections import defaultdict

from coref.utils.evaluator import EvalStatEstimator
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.evaluate_expression import expr_to_sdf
import geolipi.symbolic as gls

import coref.language as language
from .po_rewriter import PORewriter
from .utils import sample_from_bpds
from .cg_utils import FaissIVFCache
from .conversion import graph_with_target, get_new_expression


class CGRewriter(PORewriter):

    def __init__(self, config, device, dtype):
        self.shorthand = "CG"
        self.device = device
        self.dtype = dtype
        self.sample_ratio = config.sample_ratio
        self.valid_origin = config.valid_origin
        self.n_prog_per_item = config.n_prog_per_item
        self.eval_specs = config.eval_specs
        self.lang_conf = config.language_conf
        lang_conf = config.language_conf
        self.lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                            n_integer_bins=lang_conf.n_integer_bins,
                                                            tokenization=lang_conf.tokenization)
        # CG Config
        self.top_k = config.top_k
        self.rewrite_limit = config.rewrite_limit
        self.node_masking_req = config.node_masking_req
        self.add_dummy = config.add_dummy
        self.use_canonical = config.use_canonical
        self.max_expression_len = config.max_expression_len

        # Subexpr cache Config
        self.cache_config = config.cache_config

    def rewrite_expressions(self, bpds, dataset, eval_mode=False):

        start_time = time.time()
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)
        sketcher = Sketcher(device=self.device, dtype=self.dtype,
                            resolution=self.lang_conf.resolution,
                            n_dims=self.lang_conf.n_dims)

        # Generate the cache.
        if eval_mode:
            self.sample_ratio = 1.0
            if not hasattr(self, 'subexpr_cache'):
                subexpr_cache = FaissIVFCache(
                    self.cache_config, self.device, self.dtype)
                with th.no_grad():
                    subexpr_cache.generate_cache_and_index(
                        bpds, sketcher, self.lang_specs)
                self.subexpr_cache = subexpr_cache
            else:
                subexpr_cache = self.subexpr_cache
        else:
            subexpr_cache = FaissIVFCache(
                self.cache_config, self.device, self.dtype)
            with th.no_grad():
                subexpr_cache.generate_cache_and_index(
                    bpds, sketcher, self.lang_specs)

        sampled_expressions = sample_from_bpds(bpds, self.sample_ratio,
                                               self.valid_origin, self.n_prog_per_item)
        expression_stats = []
        rewritten_programs = []

        for ind, expr_dict in enumerate(sampled_expressions):
            if ind % 100 == 0:
                temp_time = time.time()
                print(
                    f"Processing {ind}th expression. Time: {temp_time - start_time}.")
            expression, program_key, prev_obj = expr_dict[
                'expression'], expr_dict['program_key'], expr_dict['prev_obj']
            target = dataset.get_target(program_key)
            with th.no_grad():
                rewrite_info = self.rewrite_expression(expression, target, prev_obj,
                                                    sketcher, stat_estimator, subexpr_cache,
                                                    eval_mode)
            rewrite_info.update({
                "program_key": program_key,
                "origin": self.shorthand,
                "log_prob": 0,
            })
            new_expr = rewrite_info['expression'].cpu()
            objective = rewrite_info['new_obj']
            expression_stats.append(rewrite_info)
            rewritten_programs.append((program_key, new_expr, objective))
        # Get log from the expression_stats
        log_info = self.get_log_info(expression_stats)
        end_time = time.time()
        log_info['time'] = end_time - start_time
        return rewritten_programs, log_info

    def load_subexpr_cache(self, bpds):
        sketcher = Sketcher(device=self.device, dtype=self.dtype,
                            resolution=self.lang_conf.resolution,
                            n_dims=self.lang_conf.n_dims)
        subexpr_cache = FaissIVFCache(
            self.cache_config, self.device, self.dtype)
        subexpr_cache.generate_cache_and_index(bpds, sketcher, self.lang_specs)
        self.subexpr_cache = subexpr_cache

    def rewrite_expression(self, expression, target, prev_obj, sketcher, stat_estimator, subexpr_cache, eval_mode=False):

        cur_expr = expression.to(self.device)
        target_th = th.tensor(target.astype(self.dtype),
                              device=self.device).reshape(-1)
        hard_target = target_th.bool()
        unsqueezed_target = hard_target.unsqueeze(0)
        counter = 0
        initial_obj = prev_obj
        cur_obj = prev_obj
        init_prog_len = stat_estimator.get_expression_size(cur_expr)
        while (counter < self.rewrite_limit):
            counter += 1

            # reverted_expr = self.lang_specs.revert_to_base_expr(cur_expr)
            # cur_prog_len = stat_estimator.get_expression_size(cur_expr)
            # output_sdf = expr_to_sdf(reverted_expr, sketcher)
            # hard_output = (output_sdf <= 0)# .float()
            # cur_obj = stat_estimator.get_individual_objective(hard_output, hard_target, cur_prog_len)

            if self.add_dummy:
                cur_expr = self.add_dummy_node(cur_expr)

            graph = graph_with_target(cur_expr, self.lang_specs, sketcher, use_canonical=self.use_canonical,
                                      target=hard_target)

            graph_nodes = [graph.nodes[i] for i in graph.nodes]

            node_ids = []
            target_list = []
            mask_list = []
            for node_id, node in enumerate(graph_nodes):
                # Use masking rate to remove things.
                node_ids.append(node_id)
                subexpr_target = node['target'][..., 0]
                subexpr_masks = node['target'][..., 1]
                target_list.append(subexpr_target)
                mask_list.append(subexpr_masks)

            if len(target_list) > 0:
                target_list = th.stack(target_list, 0)
                mask_list = th.stack(mask_list, 0)
                candidate_list = subexpr_cache.get_candidates(
                    target_list, mask_list, node_ids, self.top_k)
            else:
                candidate_list = []
            # substitute candidates.

            all_expressions = []

            for candidate in candidate_list:
                if not candidate['expression']:
                    continue

                node = graph_nodes[candidate['node_id']]
                # TODO: Add validity checks if required
                new_expression = get_new_expression(graph, candidate, sketcher, self.lang_specs,
                                                    use_canonical=self.use_canonical,
                                                    eval_mode=eval_mode)
                if self.add_dummy:
                    # check if the dummy is present
                    children = new_expression.args
                    if isinstance(children[0], gls.NullExpression3D):
                        new_expression = new_expression.args[1]
                    elif isinstance(children[1], gls.NullExpression3D):
                        new_expression = new_expression.args[0]
                # Do we need to though? If every thing is pre
                new_len = stat_estimator.get_expression_size(new_expression)
                if new_len <= self.max_expression_len:
                    all_expressions.append(new_expression)
            if len(all_expressions) > 0:
                # Use Stat estimator.
                indices = [0,]
                pred_actions = {0: [(0, x) for x in all_expressions]}
                stat_estimator.update_from_actions(
                    indices, unsqueezed_target, pred_actions, actions=False)
                key, cur_best_expr, cur_best_obj = stat_estimator.get_best_programs(
                    return_score=True, to_cpu=True)[0]
            else:
                cur_best_expr = cur_expr.args[0] # without the dummy
                cur_best_obj = cur_obj
            if cur_best_obj > cur_obj + 1e-3:
                cur_expr = cur_best_expr  # .copy()
                cur_obj = cur_best_obj
            else:
                break
        final_prog_len = stat_estimator.get_expression_size(cur_expr)
        print(
            f"Iteration {counter} : New Obj: {cur_best_obj}, Prev: {prev_obj}")
        print(
            f"Iteration {counter} : New Length: {final_prog_len}, Prev: {init_prog_len}")
        rewrite_info = {
            "expression": cur_expr,
            "new_obj": cur_best_obj,
            "prev_obj": prev_obj,
            "updated": cur_best_obj > initial_obj,
            "delta_obj": cur_best_obj - prev_obj,
            "n_rewrites": counter,
            'init_len': init_prog_len,
            'final_len': final_prog_len,
        }
        return rewrite_info

    def add_dummy_node(self, expression):
        expression = gls.Union(expression, gls.NullExpression3D())
        return expression

    def get_log_info(self, expression_dicts):

        merged_dict = defaultdict(list)
        for expr_dict in expression_dicts:
            for key, value in expr_dict.items():
                if key == 'expression':
                    continue
                if isinstance(value, th.Tensor):
                    value = value.item()
                merged_dict[key].append(value)

        success_ratio = np.mean(merged_dict['updated'])
        mean_new_objective = np.mean(merged_dict['new_obj'])
        mean_prev_objective = np.mean(merged_dict['prev_obj'])
        mean_delta_objective = np.mean(merged_dict['delta_obj'])
        max_delta = np.max(merged_dict['delta_obj'])
        min_delta = np.min(merged_dict['delta_obj'])
        mean_init_len = np.mean(merged_dict['init_len'])
        mean_final_len = np.mean(merged_dict['final_len'])
        mean_n_rewrites = np.mean(merged_dict['n_rewrites'])
        log_info = {
            "success_ratio": success_ratio,
            "mean_new_objective": mean_new_objective,
            "mean_prev_objective": mean_prev_objective,
            "mean_delta_objective": mean_delta_objective,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "n_samples": len(expression_dicts),
            "initial_length": mean_init_len,
            "final_length": mean_final_len,
            "mean_n_rewrites": mean_n_rewrites
        }
        return log_info
