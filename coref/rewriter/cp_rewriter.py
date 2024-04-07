import numpy as np
import torch as th
from collections import defaultdict
import networkx as nx
import coref.language as language
from .po_rewriter import PORewriter
from .conversion import cp_expr_to_graph


class CPRewriter(PORewriter):

    def __init__(self, config, device, dtype):
        self.shorthand = "CP"
        self.device = device
        self.dtype = dtype
        self.ex_threshold = config.ex_threshold
        self.ex_diff_threshold = config.ex_diff_threshold
        self.sample_ratio = config.sample_ratio
        self.lang_conf = config.language_conf
        self.eval_specs = config.eval_specs
        self.valid_origin = config.valid_origin
        self.n_prog_per_item = config.n_prog_per_item

        lang_conf = config.language_conf
        self.lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins, 
                                                            n_integer_bins=lang_conf.n_integer_bins,
                                                            tokenization=lang_conf.tokenization)
        self.rewrite_expressions = self.sequential_rewrite

    def rewrite_expression(self, expression, target, prev_obj,
                           sketcher, stat_estimator):
        ...
        # Create the tree with execution -> mainly rewards
        graph = cp_expr_to_graph(expression, self.lang_specs, sketcher)
        # use skecher and stat_estimator to get all the rewards
        init_len = stat_estimator.get_expression_size(expression)
        expression = self.prune_bottom_up(graph)
        new_len = stat_estimator.get_expression_size(expression)
        bu_prune_amount = init_len - new_len

        target_th = th.tensor(target.astype(self.dtype), device=self.device).reshape(-1)
        hard_target = target_th.bool()

        graph = cp_expr_to_graph(expression, self.lang_specs, sketcher, measure_obj=True, hard_target=hard_target, 
                              stat_estimator=stat_estimator)
        
        # Next top-down measure reward at each step.
        best_expression, best_obj = self.prune_top_down(graph)
        best_len = stat_estimator.get_expression_size(best_expression)

        td_prune_amount = new_len - best_len
        updated = True
        if best_obj <= prev_obj:
            best_expression = expression
            best_obj = prev_obj
            best_len = init_len
            bu_prune_amount = 0
            td_prune_amount = 0
            updated = False
        if isinstance(best_obj, th.Tensor):
            best_obj = best_obj.item()
        rewrite_info = {
            "expression": best_expression,
            "new_obj": best_obj,
            "prev_obj": prev_obj,
            "delta_obj": best_obj - prev_obj,
            "updated": updated,
            "bu_prune_amount": bu_prune_amount,
            "td_prune_amount": td_prune_amount,
            'init_len': init_len,
            'final_len': best_len,
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
        n_success = float(np.sum(merged_dict['updated']))
        mean_new_objective = np.mean(merged_dict['new_obj'])
        mean_prev_objective = np.mean(merged_dict['prev_obj'])
        mean_delta_objective = np.mean(merged_dict['delta_obj'])
        max_delta = np.max(merged_dict['delta_obj'])
        min_delta = np.min(merged_dict['delta_obj'])
        mean_bu_prune_amount = np.sum(merged_dict['bu_prune_amount'])/n_success
        mean_td_prune_amount = np.sum(merged_dict['td_prune_amount'])/n_success
        mean_init_len = np.mean(merged_dict['init_len'])
        mean_final_len = np.mean(merged_dict['final_len'])
        log_info = {
            "success_ratio": success_ratio,
            "mean_new_objective": mean_new_objective,
            "mean_prev_objective": mean_prev_objective,
            "mean_delta_objective": mean_delta_objective,
            "max_delta": max_delta,
            "min_delta": min_delta,
            "mean_bu_prune_amount": mean_bu_prune_amount,
            "mean_td_prune_amount": mean_td_prune_amount,
            "n_samples": len(expression_dicts),
            "initial_length": mean_init_len,
            "final_length": mean_final_len
        }
        return log_info


    def prune_bottom_up(self, graph, clone_param=False):
        # Rejection Criteria: Execution is ~0, or
        root = 0

        # Perform a breadth-first search to get all nodes in BFS order
        bfs_tree = list(nx.breadth_first_search.bfs_tree(graph, source=root))

        # Reverse the BFS sequence for processing from leaves to root
        bfs_tree.reverse()

        # Dictionary to store the expressions for each node
        expr_dict = {}

        # Process each node in reverse BFS order
        for node_id in bfs_tree:
            node = graph.nodes[node_id]
            func_class = node['function_class']
            params = node.get('parameters', [])

            # Combine expressions of child nodes and parameters
            children_id = list(graph.successors(node_id))
            if len(children_id) > 0:
                parse_mode = self.get_BU_parse_mode(graph, node_id, children_id)
            else:
                parse_mode = 2
            
            if parse_mode == 2:
                valid_children = [expr_dict.get(child, None) for child in children_id] 
                args_and_params = valid_children + params
                # Create the expression for the current node
                expr_dict[node_id] = func_class(*args_and_params)
            elif parse_mode == 0:
                # select the 0th child
                children = [expr_dict.get(child) for child in children_id]
                selected_child = children[0]
                expr_dict[node_id] = selected_child
            elif parse_mode == 1:
                # select the 1st child
                children = [expr_dict.get(child) for child in children_id]
                selected_child = children[1]
                expr_dict[node_id] = selected_child
        # The expression corresponding to the root node is the final result
        return expr_dict[root]

    def get_BU_parse_mode(self, graph, node_id, children_id):
        parse_mode = 0
        cur_node = graph.nodes[node_id]
        children_nodes = [graph.nodes[child] for child in children_id]
        parent_ex = cur_node['execution']
        children_execution = [child['execution'] for child in children_nodes]
        total = parent_ex.nelement()
        validity = []
        for child_ex in children_execution:
            val_1 = th.sum(child_ex) / total
            val_2 = th.sum(child_ex ^ parent_ex) / th.sum(parent_ex)
            cond_1 = val_1 > self.ex_threshold
            cond_2 = val_2 > self.ex_diff_threshold
            validity.append(cond_1 and cond_2)
        if all(validity):
            parse_mode = 2
        elif validity[0]:
            parse_mode = 0
        elif validity[1]:
            parse_mode = 1
        else:
            # both are bad, pick 0 or 1 at random
            parse_mode = np.random.randint(0, 2)
        return parse_mode

    def prune_top_down(self, graph):
        node_id_to_obj = {}
        root = 0

        # Perform a breadth-first search to get all nodes in BFS order
        bfs_tree = list(nx.breadth_first_search.bfs_tree(graph, source=root))


        # Process each node in reverse BFS order
        for node_id in bfs_tree:
            cur_node = graph.nodes[node_id]
            node_id_to_obj[node_id] = cur_node['objective']
        
        # get the best id
        best_id = max(node_id_to_obj, key=node_id_to_obj.get)
        best_node = graph.nodes[best_id]
        best_expression = best_node['expression']
        best_obj = best_node['objective']
        return best_expression, best_obj