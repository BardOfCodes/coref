from collections import defaultdict
import torch as th

import numpy as np
import os
import faiss
import _pickle as cPickle
import time
from pathlib import Path
from .conversion import cg_expr_to_graph

# TODO: Add canonicalization.


class FaissIVFCache:

    def __init__(self, cache_config, device, dtype):

        self.device = device
        self.dtype = dtype
        self.cache = []
        self.max_masking_rate = cache_config.max_masking_rate
        self.cache_size = cache_config.cache_size

        self.merge_nlist = cache_config.merge_nlist
        self.merge_nprobe = cache_config.merge_nprobe
        self.merge_bit_distance = cache_config.merge_bit_distance
        #
        self.search_nlist = cache_config.search_nlist
        self.search_nprobe = cache_config.search_nprobe
        self.n_program_to_sample = cache_config.n_program_to_sample
        self.eval_mode = cache_config.eval_mode
        self.cache_max_size = cache_config.cache_max_size

        self.save_dir = cache_config.save_dir
        self.subexpr_load_path = cache_config.subexpr_load_path
        self.load_previous_subexpr_cache = False

        self.use_canonical = cache_config.use_canonical

    def generate_cache_and_index(self, bpd, sketcher, lang_specs):
        stats = {}
        main_st = time.time()
        if self.eval_mode:
            subexpr_cache = []
        else:
            subexpr_cache = self.generate_subexpr_cache(
                bpd, sketcher, lang_specs, use_canonical=self.use_canonical)
        stats['n_new_expr'] = len(subexpr_cache)

        if len(subexpr_cache) > self.cache_max_size:
            subexpr_cache = np.random.choice(
                subexpr_cache, self.cache_max_size, replace=False)

        # Setting some primitives
        if self.eval_mode:
            all_previous_exprs = self.load_all_exprs(
                lang_specs, self.subexpr_load_path)
            subexpr_cache.extend(all_previous_exprs)
        else:
            if self.load_previous_subexpr_cache:
                st = time.time()
                all_previous_exprs = self.load_all_exprs()
                if all_previous_exprs:
                    subexpr_cache.extend(all_previous_exprs)
                print("time for loading data", time.time() - st)
        print("Number of subexpressions in cache = %d" % len(subexpr_cache))

        if len(subexpr_cache) > self.cache_max_size:
            # Need better mechanism here
            # Store the ones used more often
            subexpr_cache = np.random.choice(
                subexpr_cache, self.cache_max_size, replace=False)

        if not self.eval_mode:
            self.save_all_exprs(subexpr_cache)
        # turn on loading:
        self.load_previous_subexpr_cache = True

        stats['n_loaded_expr'] = len(subexpr_cache)

        th.cuda.empty_cache()
        subexpr_cache = self.merge_cache(subexpr_cache)

        stats['n_merged_expr'] = len(subexpr_cache)

        node_item = subexpr_cache[0]
        self.tensor_shape = node_item['execution'].shape
        self.cache_d = node_item['execution'].shape[0]
        self.empty_item = defaultdict(lambda: None)

        self.centroids, self.invlist_list, self.invlist_lookup_list = self.create_final_index(
            subexpr_cache)
        lookup_index_list = []
        for idx, invlist in enumerate(self.invlist_list):
            cur_lookup = [(idx, x) for x in range(len(invlist))]
            lookup_index_list.append(cur_lookup)
        self.lookup_index_list = lookup_index_list
        et = time.time()
        print("Overall Time", et - main_st)
        return stats

    def generate_subexpr_cache(self, bpd, sketcher, lang_specs, use_canonical=False):

        st = time.time()
        # random sample 10k expressions:
        keys = list(bpd.programs.keys())
        rand_keys = np.random.choice(
            keys, min(len(keys), self.n_program_to_sample), replace=False)

        subexpr_cache = []
        counter = 0
        for iteration, key in enumerate(rand_keys):
            programs = bpd.programs[key][:1]  # only the "best program"

            if iteration % 500 == 0:
                print("cur iteration %d. Cur Time %f" %
                      (iteration, time.time() - st))
                if len(subexpr_cache) > self.cache_max_size:
                    print("subexpr cache is full", len(
                        subexpr_cache), ". Sampling from it")
                    subexpr_cache = np.random.choice(
                        subexpr_cache, self.cache_max_size, replace=False)

            for program_set in programs:
                expression = program_set[0].to(sketcher.device)
                with th.no_grad():
                    graph = cg_expr_to_graph(
                        expression, lang_specs, sketcher, use_canonical=use_canonical)
                graph_nodes = [graph.nodes[i] for i in graph.nodes]
                for node in graph_nodes:
                    # just accept every subexpression which is
                    shape = node['execution']
                    node_item = {'execution': shape,
                                 'expression': node['expression'],
                                 # Add for canonical comparison
                                 'canonical_transform_param': node['canonical_transform_param']

                                 }
                    subexpr_cache.append(node_item)
                    counter += 1
        et = time.time()
        print("Subexpr Discovery Time", et - st)
        print("found %d sub-expressions" % counter)
        return subexpr_cache

    def merge_cache(self, subexpr_cache):
        # Now from this cache create unique:

        avg_length = np.mean([len(x['expression']) for x in subexpr_cache])
        print("Starting merge with  %d sub-expressions with avg. length %f" %
              (len(subexpr_cache), avg_length))

        st = time.time()
        cached_expression_shapes = [x['execution'] for x in subexpr_cache]
        cached_expression_shapes = th.stack(cached_expression_shapes, 0)
        # cached_expression_shapes.shape
        cached_np = cached_expression_shapes.cpu().data.numpy()
        chached_np_packed = np.packbits(cached_np, axis=-1, bitorder="little")

        self.cache_d = cached_expression_shapes.shape[1]
        merge_nb = cached_expression_shapes.shape[0]
        # Initializing index.
        quantizer = faiss.IndexBinaryFlat(self.cache_d)  # the other index

        index = faiss.IndexBinaryIVF(quantizer, self.cache_d, self.merge_nlist)
        assert not index.is_trained
        index.train(chached_np_packed)
        assert index.is_trained
        index.add(chached_np_packed)
        index.nprobe = self.merge_nprobe
        lims, D, I = index.range_search(
            chached_np_packed, self.merge_bit_distance)
        lims_shifted = np.zeros(lims.shape)
        lims_shifted[1:] = lims[:-1]

        all_indexes = set(list(range(merge_nb)))
        counter = 0
        selected_subexprs = []
        while (len(all_indexes) > 0):
            cur_ind = next(iter(all_indexes))
            sel_lims = (lims[cur_ind], lims[cur_ind+1])
            selected_ids = I[sel_lims[0]:sel_lims[1]]
            sel_exprs = [subexpr_cache[x] for x in selected_ids]
            min_len = np.inf
            for ind, expr in enumerate(sel_exprs):
                cur_len = len(expr['expression'])
                if cur_len < min_len:
                    min_len = cur_len
                    min_ind = ind

            selected_subexprs.append(sel_exprs[min_ind])
            for ind in selected_ids:
                if ind in all_indexes:
                    all_indexes.remove(ind)

        avg_length = np.mean([len(x['expression']) for x in selected_subexprs])
        print("found %d unique sub-expressions with avg. length %f" %
              (len(selected_subexprs), avg_length))
        et = time.time()
        print("Merge Process Time", et - st)

        return selected_subexprs

    def create_final_index(self, selected_subexprs):

        subselected_subexpr = []
        for subexpr in selected_subexprs:
            new_subexpr = dict(
                expression=subexpr['expression'],
                # canonical_expression=subexpr['canonical_expression']
                canonical_transform_param=subexpr.get(
                    'canonical_transform_param', None),
            )
            subselected_subexpr.append(new_subexpr)

        st = time.time()
        cached_expression_shapes = [x['execution'] for x in selected_subexprs]
        cached_expression_shapes = th.stack(cached_expression_shapes, 0)
        # cached_expression_shapes.shape
        cached_np = cached_expression_shapes.cpu().data.numpy()
        chached_np_packed = np.packbits(cached_np, axis=-1, bitorder="little")

        self.cache_d = cached_expression_shapes.shape[1]
        merge_nb = cached_expression_shapes.shape[0]
        # Initializing index.
        invl_sizes = [0]
        count = 0
        while (np.min(invl_sizes) <= 0):
            count += 1
            print("Trying to make a non-empty index for the %d th time." % count)
            quantizer = faiss.IndexBinaryFlat(self.cache_d)  # the other index
            index = faiss.IndexBinaryIVF(
                quantizer, self.cache_d, self.search_nlist)
            index.nprobe = self.search_nprobe
            assert not index.is_trained
            index.train(chached_np_packed)
            index.add(chached_np_packed)
            invl_sizes = np.array([index.invlists.list_size(i)
                                  for i in range(self.search_nlist)])
            print("Minimum list size", np.min(invl_sizes))
            if np.min(invl_sizes) <= 0:
                # should update the sizes:
                self.search_nlist = int(self.search_nlist * 0.75)
                print("setting search_nlist to %d" % self.search_nlist)
        # fetch the different centroids, and create the different lists:
        centroid_list = []
        invlist_list = []
        inv_lookup_list = []
        # Now we need to keep it within the index size:
        # Option 1: within each invlist, keep %n random shapes where %n = min(avg(n), l)

        cur_cache_size = np.sum(invl_sizes)
        if self.cache_size < cur_cache_size:
            avg_size = self.cache_size / self.search_nlist
            keep_as_is = invl_sizes <= avg_size
            # (self.cache_size - np.sum(invl_sizes[keep_as_is])) / (self.search_nlist - np.sum(keep_as_is) + 1e-9)
            new_size = int(avg_size)
            # new_size = int(new_size)
        else:
            avg_size = int(self.cache_size / self.search_nlist)
            new_size = np.max(invl_sizes)
            new_size = min(avg_size, new_size)

        for ind in range(self.search_nlist):
            centroid_list.append(quantizer.reconstruct(ind))
            invlist_ids = self.get_invlist_idx(index.invlists, ind)
            # now create the NN-Lookup array:
            invlist_ids = list(invlist_ids)
            if len(invlist_ids) > new_size:
                invlist_ids = np.random.choice(invlist_ids, new_size)
                pad = False
            else:
                pad = True
            invlist = [subselected_subexpr[x] for x in invlist_ids]
            shapelist = [selected_subexprs[x] for x in invlist_ids]
            cached_expression_shapes = [x['execution'] for x in shapelist]
            # How to fix if this is empty?

            inv_lookup = th.stack(cached_expression_shapes, 0)
            if pad:
                pad_size = new_size - len(invlist_ids)
                pad_invlist = [self.empty_item.copy() for i in range(pad_size)]
                invlist.extend(pad_invlist)
                pad_invlookup = th.zeros(
                    (1, self.cache_d), dtype=th.bool).to(self.device)
                pad_invlookup = pad_invlookup.expand(pad_size, -1)
                inv_lookup = th.cat([inv_lookup, pad_invlookup], 0)
            invlist_list.append(invlist)
            inv_lookup_list.append(inv_lookup)

        centroid_list = np.stack(centroid_list, 0)
        centroids = self.packed_np_to_tensor(centroid_list)
        et = time.time()
        print("Final Index Creation Time", et - st)

        return centroids, invlist_list, inv_lookup_list

    def packed_np_to_tensor(self, array):
        unpacked_np = np.unpackbits(array, axis=-1, bitorder="little")
        tensor = th.from_numpy(unpacked_np)
        # tensor = th.reshape(tensor.shape[0], self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2])

        tensor = tensor.to(self.device)
        return tensor

    def get_invlist_idx(self, invlists, ind):
        ls = invlists.list_size(ind)
        list_ids = np.zeros(ls, dtype='int64')
        x = invlists.get_ids(ind)
        faiss.memcpy(faiss.swig_ptr(list_ids), x, list_ids.nbytes)
        invlists.release_ids(ind, x)
        return list_ids

    def save_all_exprs(self, subexpr_cache):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(self.save_dir, "all_subexpr.pkl")
        print("Saving all expressions at %s" % file_name)
        # all the expressions...
        # convert to string form
        new_cache = []
        for expr_dict in subexpr_cache:
            new_entry = {}
            for key, value in expr_dict.items():
                if key == 'expression':
                    new_entry[key] = str(value.cpu())
                else:
                    new_entry[key] = value
            new_cache.append(new_entry)
        cPickle.dump(new_cache, open(file_name, "wb"))

    def load_all_exprs(self, lang_specs, path=None):
        if path is None:
            path = os.path.join(self.save_dir, "all_subexpr.pkl")
        print("Loading all expressions at %s" % path)
        if os.path.exists(path):
            subexpr_cache = cPickle.load(open(path, "rb"))
            for expr_dict in subexpr_cache:
                expression = eval(
                    expr_dict['expression'], lang_specs.STR_TO_CMD_MAPPER)
                expr_dict['expression'] = expression.to(self.device)
            return subexpr_cache
        else:
            print("file %s does not exist. No data Loading" % path)
            return None

    def get_candidates(self, targets, masks, node_ids, top_k):
        candidate_list = []

        # find masked iou with centroids
        if len(self.centroids.shape) == 2:
            self.centroids = self.centroids.unsqueeze(0)
        # Given the targets, find the k nearest for each target:
        targets = targets.unsqueeze(1)
        masks = masks.unsqueeze(1)

        scores = batched_masked_scores_with_stack(
            self.centroids, targets, masks)
        maxes, arg_maxes = th.topk(scores, dim=1, k=self.search_nprobe)

        arg_maxes = arg_maxes.cpu().numpy()
        # each lookup will return k, we can create a list of k
        all_lookup_lists = []
        all_scores = []

        for ind, node_id in enumerate(node_ids):
            cur_target = targets[ind, :]
            cur_mask = masks[ind, :]
            sel_invlist_index = list(arg_maxes[ind])

            lookup_lists = []
            for idx in sel_invlist_index:
                lookup_lists.extend(self.lookup_index_list[idx])
            all_lookup_lists.append(lookup_lists)

            inv_lookups = [self.invlist_lookup_list[idx]
                           for idx in sel_invlist_index]
            inv_lookups = th.cat(inv_lookups, 0)
            scores = batched_masked_scores(inv_lookups, cur_target, cur_mask)
            all_scores.append(scores)

        all_scores = th.stack(all_scores, 0)
        real_k = min(top_k, all_scores.shape[1])
        cur_maxes, cur_arg_maxes = th.topk(all_scores, dim=1, k=real_k)
        cur_arg_maxes = list(cur_arg_maxes.cpu().numpy())

        for ind, node_id in enumerate(node_ids):
            node_arg_maxes = cur_arg_maxes[ind]
            node_lookup_lists = all_lookup_lists[ind]
            node_cur_maxes = cur_maxes[ind]
            for idx, sel_ind in enumerate(node_arg_maxes):
                sel_indexes = node_lookup_lists[sel_ind]
                inv_list_id = sel_indexes[0]  # sel_invlist_index[nprobe_id]
                list_position = sel_indexes[1]  # sel_ind % invlist_size
                sel_item = self.invlist_list[inv_list_id][list_position]
                if 'expression' in sel_item.keys():
                    candidate = self.get_candidate_format(
                        node_id, idx, node_cur_maxes, sel_item)
                    candidate_list.append(candidate)
        return candidate_list

    def get_candidate_format(self, node_id, idx, node_cur_maxes, sel_item):
        candidate = {'node_id': node_id,
                     'masked_score': node_cur_maxes[idx],
                     'expression': sel_item['expression'],
                     'canonical_transform_param': sel_item['canonical_transform_param']
                     #  'bool_count': bool_count(sel_item),
                     #  'canonical_expression': sel_item['canonical_expression']
                     }

        return candidate


def batched_masked_scores(cache, targets, masks):
    stacked_cache = cache  # .expand(targets.shape[0], -1, -1)
    stacked_targets = targets  # .expand(-1, cache.shape[1], -1)
    stacked_masks = masks  # .expand(-1, cache.shape[1], -1)
    R = th.sum(th.logical_and(th.logical_and(stacked_cache, stacked_targets), stacked_masks), -1) / \
        (th.sum(th.logical_and(th.logical_or(stacked_cache,
         stacked_targets), stacked_masks), -1) + 1e-6)
    return R


def batched_masked_scores_with_stack(cache, targets, masks):
    stacked_cache = cache.expand(targets.shape[0], -1, -1)
    stacked_targets = targets.expand(-1, cache.shape[1], -1)
    stacked_masks = masks.expand(-1, cache.shape[1], -1)
    R = th.sum(th.logical_and(th.logical_and(stacked_cache, stacked_targets), stacked_masks), -1) / \
        (th.sum(th.logical_and(th.logical_or(stacked_cache,
         stacked_targets), stacked_masks), -1) + 1e-6)
    return R


class MergeSpliceCache(FaissIVFCache):
    ...
    # For the first version we don't need this.
