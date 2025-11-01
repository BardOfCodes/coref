import numpy as np
import torch as th
from collections import defaultdict
import time
from .utils import sample_from_bpds, transform_to_tunable, params_from_variables
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.deprecated import expr_to_sdf
from coref.utils.evaluator import EvalStatEstimator
import coref.language as language
import torch.multiprocessing as mp


class PORewriter:

    def __init__(self, config, device, dtype):
        self.shorthand = "PO"
        self.device = device
        self.dtype = dtype
        self.opt_step_size = config.opt_step_size
        self.n_iters = config.n_iters
        self.sample_ratio = config.sample_ratio
        self.lang_conf = config.language_conf
        self.eval_specs = config.eval_specs
        self.n_processes = config.n_processes
        self.config = config.clone()

        lang_conf = config.language_conf
        self.lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                            n_integer_bins=lang_conf.n_integer_bins,
                                                            tokenization=lang_conf.tokenization)
        self.valid_origin = config.valid_origin
        self.n_prog_per_item = config.n_prog_per_item
        scale_factors_param = config.scale_factors_param
        start = np.log(scale_factors_param[0])
        end = np.log(scale_factors_param[1])
        scale_factors = np.exp(
            np.arange(start,  end, (end-start)/float(self.n_iters)))
        self.scale_factors = scale_factors

        self.sigmoid_func = th.nn.Sigmoid()
        self.verbose = True
        if self.n_processes > 1:
            self.rewrite_expressions = self.parallel_rewrite
        else:
            self.rewrite_expressions = self.sequential_rewrite

    def sequential_rewrite(self, bpds, dataset, eval_mode=False):

        if eval_mode:
            self.sample_ratio = 1.0
        start_time = time.time()
        stat_estimator = EvalStatEstimator(self.eval_specs, self.lang_conf,
                                           self.dtype, self.device)
        sketcher = Sketcher(device=self.device, dtype=self.dtype,
                            resolution=self.lang_conf.resolution,
                            n_dims=self.lang_conf.n_dims)

        sampled_expressions = sample_from_bpds(bpds, self.sample_ratio,
                                               self.valid_origin, self.n_prog_per_item)

        expression_stats = []
        rewritten_programs = []
        for expr_dict in sampled_expressions:
            expression, program_key, prev_obj = expr_dict[
                'expression'], expr_dict['program_key'], expr_dict['prev_obj']
            target = dataset.get_target(program_key)
            rewrite_info = self.rewrite_expression(
                expression, target, prev_obj, sketcher, stat_estimator)
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

    def rewrite_expression(self, expression, target, prev_obj,
                           sketcher, stat_estimator):

        expression = expression.cuda()
        tensor_list = expression.gather_tensor_list(type_annotate=True)
        param, variable_list = transform_to_tunable(
            tensor_list, self.lang_specs)
        # Now the optimization loop:

        target_th = th.tensor(target.astype(self.dtype),
                              device=self.device).reshape(-1)

        hard_target = target_th.bool()

        optim = th.optim.Adam(variable_list, lr=self.opt_step_size)
        # The amount of "smoothing"
        best_obj = prev_obj - 0.0001
        best_params = [x.detach() for x in expression.gather_tensor_list()]

        # get the evaluator to specify objective
        prog_len = stat_estimator.get_expression_size(expression)

        updated = False
        best_iter = 0

        for i in range(self.n_iters):
            scale_factor = self.scale_factors[i]
            optim.zero_grad()
            transformed_params = params_from_variables(
                variable_list, tensor_list, self.lang_specs)
            cur_expr = expression.inject_tensor_list(transformed_params)
            reverted_expr = self.lang_specs.revert_to_base_expr(cur_expr)
            output_sdf = expr_to_sdf(reverted_expr, sketcher)

            output_tanh = th.tanh(output_sdf * scale_factor)
            output_shape = self.sigmoid_func(-output_tanh * scale_factor)
            output_loss = th.nn.functional.mse_loss(output_shape, target_th)

            optim.zero_grad()
            output_loss.backward()
            optim.step()

            # measure the reward
            hard_output = (output_sdf <= 0)  # .float()
            new_obj = stat_estimator.get_individual_objective(
                hard_output, hard_target, prog_len)

            # if better then best, update
            if new_obj > best_obj:
                best_obj = new_obj
                best_params = [x.detach()
                               for x in cur_expr.gather_tensor_list()]
                updated = True
                best_iter = i

            # if self.verbose:
        best_expression = expression.inject_tensor_list(best_params)

        iou = new_obj + stat_estimator.parsimony_factor * prog_len
        print(
            f"Iteration: {i}, Objective: {new_obj}, Loss: {output_loss.item()}, IOU: {iou}")

        # also make some log information and return

        if isinstance(best_obj, th.Tensor):
            best_obj = best_obj.item()
        rewrite_info = {
            "expression": best_expression,
            "new_obj": best_obj,
            "prev_obj": prev_obj,
            "updated": updated,
            "delta_obj": best_obj - prev_obj,
            "best_iter": best_iter,

        }

        return rewrite_info

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
        mean_best_iter = np.mean(merged_dict['best_iter'])
        log_info = {
            "success_ratio": success_ratio,
            "mean_new_objective": mean_new_objective,
            "mean_prev_objective": mean_prev_objective,
            "mean_delta_objective": mean_delta_objective,
            "mean_best_iter": mean_best_iter,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "n_samples": len(expression_dicts),
        }
        return log_info

    def rewrite_sampled_expressions(self, expr_dicts, targets, sketcher, stat_estimator, queue):

        expression_stats = []
        rewritten_programs = []
        for ind, expr_dict in enumerate(expr_dicts):
            print('Index:', ind)
            expression, program_key, prev_obj = expr_dict[
                'expression'], expr_dict['program_key'], expr_dict['prev_obj']
            target = targets[ind]
            expression = eval(expression, self.lang_specs.STR_TO_CMD_MAPPER)
            rewrite_info = self.rewrite_expression(
                expression, target, prev_obj, sketcher, stat_estimator)
            rewrite_info.update({
                "program_key": program_key,
                "origin": self.shorthand,
                "log_prob": 0,
            })
            new_expr = str(rewrite_info.pop('expression').cpu())
            objective = rewrite_info['new_obj']
            program_set = (program_key, new_expr, objective)
            queue.put((program_set, rewrite_info))
            # expression_stats.append(rewrite_info)
            # rewritten_programs.append((program_key, new_expr, objective))

        # Get log from the expression_stats
        return rewritten_programs, expression_stats

    def parallel_rewrite(self, bpds, dataset, eval_mode=False):

        if eval_mode:
            self.sample_ratio = 1.0
        start_time = time.time()

        sampled_expressions = sample_from_bpds(bpds, self.sample_ratio,
                                               self.valid_origin, self.n_prog_per_item)
        # divide it by the n_processes.
        # per_process_expr_dicts [[] for x in range(self.n_processes)]
        per_process_targets = [[] for x in range(self.n_processes)]
        per_process_expr_dicts = [[] for x in range(self.n_processes)]

        for ind, expr_dict in enumerate(sampled_expressions):
            expression, program_key, prev_obj = expr_dict[
                'expression'], expr_dict['program_key'], expr_dict['prev_obj']
            # to make it picklable:
            expression = str(expression.cpu())
            expr_dict['expression'] = expression
            target = dataset.get_target(program_key)
            process_ind = ind % self.n_processes
            per_process_expr_dicts[process_ind].append(expr_dict)
            per_process_targets[process_ind].append(target)

        queue = mp.Manager().Queue()

        processes = []
        for proc_id in range(self.n_processes):
            # print("starting process %d of %d" % (proc_id, self.n_processes))
            # out = parallel_po(proc_id, self.config, per_process_expr_dicts[proc_id],
            #                   per_process_targets[proc_id], queue)
            p = mp.Process(target=parallel_po, args=(proc_id, self.config,
                           per_process_expr_dicts[proc_id],
                per_process_targets[proc_id],
                self.device, self.dtype,
                queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())

        expression_stats = []
        rewritten_programs = []
        for output in outputs:
            cur_rewritten_programs, cur_expression_stats = output
            rewritten_programs.append(cur_rewritten_programs)
            expression_stats.append(cur_expression_stats)

        for ind, prog_tuple in enumerate(rewritten_programs):
            program_key, new_expr, objective = prog_tuple
            new_expr = eval(new_expr, self.lang_specs.STR_TO_CMD_MAPPER)
            rewritten_programs[ind] = (program_key, new_expr, objective)

        # Ideally just combine them all
        log_info = self.get_log_info(expression_stats)
        process_time = time.time() - start_time
        log_info['time'] = process_time
        return rewritten_programs, log_info


def parallel_po(proc_id, config, expr_dicts, targets, device, dtype, queue):
    sketcher = Sketcher(device=device, dtype=dtype,
                        resolution=config.language_conf.resolution,
                        n_dims=config.language_conf.n_dims)
    stat_estimator = EvalStatEstimator(config.eval_specs, config.language_conf,
                                       dtype, device)

    rewriter = PORewriter(config, device, dtype)

    output = rewriter.rewrite_sampled_expressions(
        expr_dicts, targets, sketcher, stat_estimator, queue)
