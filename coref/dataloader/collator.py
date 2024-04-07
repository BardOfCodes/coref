import torch as th
import numpy as np
from collections import defaultdict
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.evaluate_expression import expr_to_sdf
from geolipi.torch_compute.compile_expression import create_compiled_expr
from geolipi.torch_compute.batch_evaluate_sdf import create_evaluation_batches, batch_evaluate
# TODO: Move to utils if it will be used for other purposes too.
import coref.language as language


class SynthCollator:

    def __init__(self, eval_size, device, dtype, resolution, n_dims, dl_specs, lang_conf):
        self.sketcher = Sketcher(
            device=device, dtype=dtype, resolution=resolution, n_dims=n_dims)
        self.resolution = resolution
        self.eval_size = eval_size
        self.cached_form = dl_specs.data_gen.cached_form
        if self.cached_form:
            self.construct_block = self.construct_block_cached
            self.lang_specs = None
        else:
            lang_specs = getattr(language,  lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                            n_integer_bins=lang_conf.n_integer_bins,
                                                            tokenization=lang_conf.tokenization)
            self.lang_specs = lang_specs
            if dl_specs.data_gen.batch_uncached:
                self.construct_block = self.construct_block_uncached_batched
            else:
                self.construct_block = self.construct_block_uncached

    def construct_block_uncached_batched(self, batch, convert_to_cuda=True):

        eval_stack = []
        all_actions = []
        all_action_validity = []
        all_n_actions = []

        for item in batch:
            program, actions, action_validity, n_actions = item
            reverted_program = self.lang_specs.revert_to_base_expr(program)
            reverted_program = reverted_program.to(self.sketcher.device)
            expr, transforms, inversions, params = create_compiled_expr(
                reverted_program, self.sketcher)
            eval_stack.append([expr, transforms, inversions, params])
            all_actions.append(actions)
            all_action_validity.append(action_validity)
            all_n_actions.append(n_actions)

        with th.no_grad():
            eval_batches = create_evaluation_batches(
                eval_stack, convert_to_cuda=convert_to_cuda)
            all_sdfs = batch_evaluate(eval_batches, self.sketcher)

        return all_sdfs, all_actions, all_action_validity, all_n_actions

    def construct_block_uncached(self, batch, convert_to_cuda=True):
        # TODO: convert_to_cuda is not used here. Check if it is needed.
        all_sdfs = []
        all_actions = []
        all_action_validity = []
        all_n_actions = []

        for item in batch:
            program, actions, action_validity, n_actions = item
            reverted_program = self.lang_specs.revert_to_base_expr(program)
            reverted_program = reverted_program.to(self.sketcher.device)
            cur_sdf = expr_to_sdf(reverted_program, self.sketcher)
            all_actions.append(actions)
            all_action_validity.append(action_validity)
            all_n_actions.append(n_actions)
            all_sdfs.append(cur_sdf)

        all_sdfs = th.stack(all_sdfs, dim=0)

        return all_sdfs, all_actions, all_action_validity, all_n_actions

    def construct_block_cached(self, batch, convert_to_cuda=True):

        eval_stack = []
        all_actions = []
        all_action_validity = []
        all_n_actions = []

        for item in batch:
            expr, transforms, inversions, params, actions, action_validity, n_actions = item
            eval_stack.append([expr, transforms, inversions, params])
            all_actions.append(actions)
            all_action_validity.append(action_validity)
            all_n_actions.append(n_actions)

        with th.no_grad():
            eval_batches = create_evaluation_batches(
                eval_stack, convert_to_cuda=convert_to_cuda)
            all_sdfs = batch_evaluate(eval_batches, self.sketcher)

        return all_sdfs, all_actions, all_action_validity, all_n_actions

    def __call__(self, batch, convert_to_cuda=True):

        all_sdfs = []
        all_actions = []
        all_action_validity = []
        all_n_actions = []

        n_chunks = np.ceil(len(batch) / self.eval_size).astype(np.int32)

        for ind in range(n_chunks):
            cur_batch = batch[ind * self.eval_size: (ind+1) * self.eval_size]
            cur_sdf, cur_actions, cur_action_validity, cur_n_actions = self.construct_block(
                cur_batch, convert_to_cuda=convert_to_cuda)
            all_sdfs.append(cur_sdf)
            all_actions.extend(cur_actions)
            all_action_validity.extend(cur_action_validity)
            all_n_actions.extend(cur_n_actions)

        all_sdfs = th.cat(all_sdfs, dim=0)
        all_occs = (all_sdfs <= 0).float()

        actions = th.stack(all_actions, dim=0)
        action_validity = th.stack(all_action_validity, dim=0)
        n_actions = th.tensor(all_n_actions, dtype=th.long)
        if convert_to_cuda:
            actions = actions.cuda()
            action_validity = action_validity.cuda()
            n_actions = n_actions.cuda()
        resize_shape = [-1] + [self.resolution] * self.sketcher.n_dims
        all_occs = all_occs.reshape(*resize_shape)
        out_pack = {
            "occs": all_occs,
            "actions": actions,
            "action_validity": action_validity,
            "n_actions": n_actions
        }
        return out_pack


class SynthShapesCollator(SynthCollator):

    def construct_block_cached(self, batch, convert_to_cuda=True):

        eval_stack = []
        all_indices = []

        for item in batch:
            expr, transforms, inversions, params, index = item
            eval_stack.append([expr, transforms, inversions, params])
            index = "_".join([str(y) for y in index])
            all_indices.append(index)

        eval_batches = create_evaluation_batches(
            eval_stack, convert_to_cuda=convert_to_cuda)
        all_sdf = batch_evaluate(eval_batches, self.sketcher)

        return all_sdf, all_indices

    def construct_block_uncached_batched(self, batch, convert_to_cuda=True):

        eval_stack = []
        all_indices = []

        for item in batch:
            program, index = item
            reverted_program = self.lang_specs.revert_to_base_expr(program)
            reverted_program = reverted_program.to(self.sketcher.device)
            expr, transforms, inversions, params = create_compiled_expr(
                reverted_program, self.sketcher)
            eval_stack.append([expr, transforms, inversions, params])
            index = "_".join([str(y) for y in index])
            all_indices.append(index)

        with th.no_grad():
            eval_batches = create_evaluation_batches(
                eval_stack, convert_to_cuda=convert_to_cuda)
            all_sdfs = batch_evaluate(eval_batches, self.sketcher)

        return all_sdfs, all_indices

    def construct_block_uncached(self, batch, convert_to_cuda=True):
        # TODO: convert_to_cuda is not used here. Check if it is needed.
        all_sdfs = []
        all_indices = []

        for item in batch:
            program, index = item
            reverted_program = self.lang_specs.revert_to_base_expr(program)
            reverted_program = reverted_program.to(self.sketcher.device)
            cur_sdf = expr_to_sdf(reverted_program, self.sketcher)
            all_sdfs.append(cur_sdf)
            index = "_".join([str(y) for y in index])
            all_indices.append(index)

        all_sdfs = th.stack(all_sdfs, dim=0)

        return all_sdfs, all_indices

    def __call__(self, batch, convert_to_cuda=True):

        all_sdfs = []
        all_indices = []
        n_chunks = np.ceil(len(batch) / self.eval_size).astype(np.int32)

        for ind in range(n_chunks):
            cur_batch = batch[ind * self.eval_size: (ind+1) * self.eval_size]
            cur_sdfs, indices = self.construct_block(
                cur_batch, convert_to_cuda=convert_to_cuda)
            all_sdfs.append(cur_sdfs)
            all_indices.extend(indices)
        all_sdfs = th.cat(all_sdfs, dim=0)
        all_sdfs = (all_sdfs <= 0).float()
        reshape_shape = [-1] + [self.resolution] * self.sketcher.n_dims
        all_sdfs = all_sdfs.reshape(*reshape_shape)
        # all_indices = ["_".join([str(y) for y in x]) for x in all_indices]
        out_pack = {
            "occs": all_sdfs,
            "indices": all_indices
        }
        return out_pack


class PLADCollator(SynthCollator):

    def __init__(self, eval_size, device, dtype, resolution, n_dims):
        self.sketcher = Sketcher(
            device=device, dtype=dtype, resolution=resolution, n_dims=n_dims)
        self.resolution = resolution
        self.eval_size = eval_size

    def construct_block(self, batch, convert_to_cuda=True):

        data_eval_stack = []
        data_all_actions = []
        data_all_action_validity = []
        data_all_n_actions = []

        expr_eval_stack = []
        expr_all_actions = []
        expr_all_action_validity = []
        expr_all_n_actions = []

        for item in batch:
            expr_obj, data, action_obj = item
            actions, action_validity, n_actions = action_obj
            if len(expr_obj) == 0:
                data_all_actions.append(actions)
                data_all_action_validity.append(action_validity)
                data_all_n_actions.append(n_actions)
                data_eval_stack.append(data.astype(np.float32))
            else:
                expr, transforms, inversions, params = expr_obj
                expr_all_actions.append(actions)
                expr_all_action_validity.append(action_validity)
                expr_all_n_actions.append(n_actions)
                expr_eval_stack.append([expr, transforms, inversions, params])

        data_eval_stack = np.stack(data_eval_stack, axis=0)
        data_eval_stack = th.tensor(
            data_eval_stack, dtype=th.float32, device=self.sketcher.device)
        if expr_eval_stack:
            with th.no_grad():
                eval_batches = create_evaluation_batches(
                    expr_eval_stack, convert_to_cuda=convert_to_cuda)
                all_sdfs = batch_evaluate(eval_batches, self.sketcher)
                all_sdfs = (all_sdfs <= 0).float()

                reshape_shape = [-1] + [self.resolution] * self.sketcher.n_dims
                all_sdfs = all_sdfs.reshape(*reshape_shape)
            all_sdfs = th.cat([data_eval_stack, all_sdfs], dim=0)
        else:
            all_sdfs = data_eval_stack
        all_actions = data_all_actions + expr_all_actions
        all_action_validity = data_all_action_validity + expr_all_action_validity
        all_n_actions = data_all_n_actions + expr_all_n_actions
        return all_sdfs, all_actions, all_action_validity, all_n_actions

    def __call__(self, batch, convert_to_cuda=True):

        all_sdfs = []
        all_actions = []
        all_action_validity = []
        all_n_actions = []

        n_chunks = np.ceil(len(batch) / self.eval_size).astype(np.int32)

        for ind in range(n_chunks):
            cur_batch = batch[ind * self.eval_size: (ind+1) * self.eval_size]
            cur_sdf, actions, action_validity, n_actions = self.construct_block(
                cur_batch, convert_to_cuda=convert_to_cuda)

            all_sdfs.append(cur_sdf)
            all_actions.extend(actions)
            all_action_validity.extend(action_validity)
            all_n_actions.extend(n_actions)

        all_sdfs = th.cat(all_sdfs, dim=0)

        actions = th.stack(all_actions, dim=0)
        action_validity = th.stack(all_action_validity, dim=0)
        n_actions = th.tensor(all_n_actions, dtype=th.long)
        if convert_to_cuda:
            actions = actions.cuda()
            action_validity = action_validity.cuda()
            n_actions = n_actions.cuda()
        out_pack = {
            "occs": all_sdfs,
            "actions": actions,
            "action_validity": action_validity,
            "n_actions": n_actions
        }
        return out_pack


def shapes_collator(batch, device="cuda"):

    all_occs = []
    all_keys = []
    for ind, value in enumerate(batch):
        occ = value[0]
        key = value[1]
        key = "_".join([str(y) for y in key])
        occ = th.tensor(occ, dtype=th.float32, device=device)
        all_occs.append(occ)
        all_keys.append(key)
    all_occs = th.stack(all_occs, dim=0)
    out_pack = {
        "occs": all_occs,
        "indices": all_keys
    }
    return out_pack
