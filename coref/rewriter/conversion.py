
import sympy as sp
import numpy as np
import networkx as nx
import torch as th
from geolipi.symbolic.base_symbolic import GLFunction
from .function_inversions import INVERSION_MAP
from geolipi.torch_compute.evaluate_expression import expr_to_sdf
from geolipi.symbolic.types import COMBINATOR_TYPE
import geolipi.symbolic as gls
from .function_inversions import shape_scale, shape_translate


def cp_expr_to_graph(expr, lang_specs, sketcher,
                     measure_obj=False,
                     hard_target=None,
                     stat_estimator=None):
    G = nx.DiGraph()
    index_count = 0
    stack = [(expr, None)]  # Stack of tuples: (expression, parent node ID)
    while stack:
        current_expr, parent_id = stack.pop()

        if isinstance(current_expr, GLFunction):
            # Process the current GLFunction expression
            instance_id = index_count
            G.add_node(instance_id, function_class=current_expr.__class__)
            index_count += 1
            # add important details here - Reward
            reverted_expr = lang_specs.revert_to_base_expr(current_expr)
            output_sdf = expr_to_sdf(reverted_expr, sketcher)
            hard_output = (output_sdf <= 0)  # .float()

            G.nodes[instance_id]['expression'] = current_expr
            G.nodes[instance_id]['execution'] = hard_output
            # Link to parent if there is one
            if measure_obj:
                prog_len = stat_estimator.get_expression_size(current_expr)
                new_obj = stat_estimator.get_individual_objective(
                    hard_output, hard_target, prog_len)
                G.nodes[instance_id]['objective'] = new_obj
            if parent_id is not None:
                G.add_edge(parent_id, instance_id)

            # Push child expressions onto the stack
            stack_adds = []
            for arg in current_expr.args:
                if isinstance(arg, GLFunction):
                    stack_adds.append((arg, instance_id))
                elif isinstance(arg, sp.Symbol):
                    tensor = current_expr.lookup_table[arg]
                    if 'parameters' in G.nodes[instance_id]:
                        G.nodes[instance_id]['parameters'].append(tensor)
                    else:
                        G.nodes[instance_id]['parameters'] = [tensor]
                # Additional argument processing can be added here
            stack.extend(stack_adds[::-1])
    return G


def cg_expr_to_graph(expr, lang_specs, sketcher, use_canonical=False):
    G = nx.DiGraph()
    index_count = 0
    stack = [(expr, None)]  # Stack of tuples: (expression, parent node ID)
    while stack:
        current_expr, parent_id = stack.pop()

        if isinstance(current_expr, GLFunction):
            # Process the current GLFunction expression
            instance_id = index_count
            G.add_node(instance_id, function_class=current_expr.__class__)
            index_count += 1
            # add important details here - Reward
            reverted_expr = lang_specs.revert_to_base_expr(current_expr)
            output_sdf = expr_to_sdf(reverted_expr, sketcher)
            hard_output = (output_sdf <= 0)  # .float()
            G.nodes[instance_id]['expression'] = current_expr
            if use_canonical:
                if isinstance(current_expr, gls.NullExpression3D):
                    canonical_execution = hard_output
                    canonical_param = None
                    unnorm_bbox = None
                else:
                    canonical_execution, canonical_param, unnorm_bbox = get_canonical_output(
                        hard_output, sketcher, lang_specs)
                    canonical_execution = canonical_execution[..., 0].reshape(
                        -1)
                G.nodes[instance_id]['execution'] = canonical_execution
                G.nodes[instance_id]['canonical_transform_param'] = canonical_param
                G.nodes[instance_id]['unnorm_bbox'] = unnorm_bbox
            else:
                G.nodes[instance_id]['execution'] = hard_output
                G.nodes[instance_id]['canonical_transform_param'] = None
                G.nodes[instance_id]['unnorm_bbox'] = None
            # Link to parent if there is one
            if parent_id is not None:
                G.add_edge(parent_id, instance_id)

            # Push child expressions onto the stack
            stack_adds = []
            for arg in current_expr.args:
                if isinstance(arg, GLFunction):
                    stack_adds.append((arg, instance_id))
                elif isinstance(arg, sp.Symbol):
                    tensor = current_expr.lookup_table[arg]
                    if 'parameters' in G.nodes[instance_id]:
                        G.nodes[instance_id]['parameters'].append(tensor)
                    else:
                        G.nodes[instance_id]['parameters'] = [tensor]
                # Additional argument processing can be added here
            stack.extend(stack_adds[::-1])
    return G


def get_canonical_output(hard_output, sketcher, lang_specs):
    res = sketcher.resolution
    # estimate parameters
    grid_divider = [(res/2.), (res/2.), (res/2.)]
    output_reshaped = hard_output.reshape(res, res, res)
    unnorm_bbox = get_bbox(output_reshaped, grid_divider, normalized=False)
    canonical_output, canonical_param = canonical_exec(
        output_reshaped, unnorm_bbox, sketcher, lang_specs)
    return canonical_output, canonical_param, unnorm_bbox


def canonical_exec(output_reshaped, unnorm_bbox, sketcher, lang_specs):
    res = sketcher.resolution
    # estimate parameters
    grid_divider = [(res/2.), (res/2.), (res/2.)]
    normalized_bbox = (-1 + (unnorm_bbox + 0.5)/grid_divider)
    center = normalized_bbox.mean(0)
    unnorm_bbox[1] += 1
    normalized_bbox = (-1 + (unnorm_bbox + 0.5)/grid_divider)
    size = normalized_bbox[1] - normalized_bbox[0]
    inverse_translate_params = -center
    inverse_size_params = (2 - 1/res)/(size + 1e-9)
    inverse_translate_params = th.tensor(
        inverse_translate_params, device=output_reshaped.device, dtype=sketcher.dtype)
    inverse_size_params = th.tensor(
        inverse_size_params, device=output_reshaped.device, dtype=sketcher.dtype)
    # perform inversion
    coords = sketcher.get_base_coords()[None, ...]
    if output_reshaped.shape[-1] != 2:
        inversion_input = th.stack(
            [output_reshaped, th.ones_like(output_reshaped)], -1).float()
    else:
        inversion_input = output_reshaped.float()
    adjusted_output = shape_translate(
        inverse_translate_params, inversion_input, res, coords)
    canonical_output = shape_scale(
        inverse_size_params, adjusted_output.float(), res, coords)
    canonical_param = th.stack(
        [inverse_size_params, inverse_translate_params], 0)
    return canonical_output, canonical_param

    # get bounding box:


def get_bbox(output_reshaped, grid_divider, normalized=True):
    sdf_np = output_reshaped.cpu().numpy()
    sdf_coords = np.stack(np.where(sdf_np == 1), -1)
    if sdf_coords.shape[0] == 0:
        sdf_coords = np.zeros((2, 3), dtype=np.float32)
    if normalized:
        sdf_coords = -1 + (sdf_coords + 0.5)/grid_divider

    min_x, max_x = np.min(sdf_coords[:, 0]), np.max(sdf_coords[:, 0])
    min_y, max_y = np.min(sdf_coords[:, 1]), np.max(sdf_coords[:, 1])
    min_z, max_z = np.min(sdf_coords[:, 2]), np.max(sdf_coords[:, 2])
    bbox = np.array([[min_x, min_y, min_z],
                     [max_x, max_y, max_z]])
    return bbox


def graph_with_target(expr, lang_specs, sketcher, use_canonical=False, target=None):
    graph = cg_expr_to_graph(expr, lang_specs, sketcher,
                             use_canonical=use_canonical)
    state_list = [0]
    target = th.stack(
        [target, th.ones(target.shape, dtype=th.bool).to(target.device)], -1)
    target_stack = [target.clone()]
    while (state_list):
        cur_node_id = state_list[0]
        state_list = state_list[1:]
        cur_node = graph.nodes[cur_node_id]
        cur_target = target_stack.pop()
        cur_expr = cur_node['expression']
        # Perform processing according to the node type
        if use_canonical:
            if isinstance(cur_expr, gls.NullExpression3D):
                # consider "reminder target shape as the target"
                masked_target = th.logical_and(
                    cur_target[..., 0], cur_target[..., 1]).float()
                res = sketcher.resolution
                grid_divider = [(res/2.), (res/2.), (res/2.)]
                output_reshaped = masked_target.reshape(res, res, res)
                unnorm_bbox = get_bbox(
                    output_reshaped, grid_divider, normalized=False)
                cur_target_reshaped = cur_target.reshape(res, res, res, 2)
                canonical_target, inversion_params = canonical_exec(cur_target_reshaped.clone(), unnorm_bbox,
                                                                    sketcher, lang_specs)
                cur_node['target'] = canonical_target.reshape(-1, 2)
                cur_node['canonical_transform_param'] = inversion_params
            else:
                # has a bbox
                res = sketcher.resolution
                cur_target_reshaped = cur_target.reshape(res, res, res, 2)
                unnorm_bbox = cur_node['unnorm_bbox']
                canonical_target, _ = canonical_exec(
                    cur_target_reshaped.clone(), unnorm_bbox, sketcher, lang_specs)
                cur_node['target'] = canonical_target.reshape(-1, 2)
        else:
            cur_node['target'] = cur_target.clone().detach()

        if isinstance(cur_expr, COMBINATOR_TYPE):
            children = list(graph.successors(cur_node_id))
            child_a = graph.nodes[children[0]]
            child_b = graph.nodes[children[1]]
            child_b_canvas = child_b['execution']
            child_a_canvas = child_a['execution']
            target_a = INVERSION_MAP[cur_expr.__class__](
                cur_target, child_b_canvas, 0)
            target_b = INVERSION_MAP[cur_expr.__class__](
                cur_target, child_a_canvas, 1)
            target_stack.append(target_b)
            target_stack.append(target_a)

        children = list(graph.successors(cur_node_id))
        for child_id in children[::-1]:
            state_list.insert(0, child_id)
    return graph


def get_canonical_target(target, sketcher, lang_specs, node):
    inverse_param = node['canonical_transform_param']
    res = lang_specs.resolution
    coords = sketcher.get_base_coords()[None, ...]
    # target = th.stack([target, th.ones(target.shape, dtype=th.bool).to(target.device)], -1)
    adjusted_output = shape_translate(
        inverse_param[0], target.float(), res, coords)
    canonical_output = shape_scale(
        inverse_param[1], adjusted_output.float(), res, coords)
    return canonical_output


def get_new_expression(G, candidate, sketcher, lang_specs, use_canonical=False, eval_mode=False):
    """
    Substitutes the expression of a given node with a new expression and
    returns the overall expression from the root of the graph without altering
    the original graph.

    Args:
    G (networkx.DiGraph): The graph representing the expression tree.
    target_node: The node in the graph whose expression is to be substituted.
    new_expression: The new expression to substitute at the target node.

    Returns:
    The new overall expression from the root after substitution.
    """
    # Apply resolution here on target node, based on new_expr and the canonical cmds.
    # Revert to base
    # add the transform nodes
    # map to primal.
    # Apply quantization + limits.
    target_node_id = candidate['node_id']
    new_expression = candidate['expression']
    if use_canonical:
        reverted_expr = lang_specs.revert_to_base_expr(new_expression)
        candidate_transform_param = candidate['canonical_transform_param']
        scale_param = candidate_transform_param[0]
        translate_param = candidate_transform_param[1]
        updated_expr = gls.Scale3D(
            gls.Translate3D(reverted_expr, translate_param),
            scale_param)
        # Target's transform
        target_node = G.nodes[target_node_id]
        target_transform_param = target_node['canonical_transform_param']
        scale_param = 1/(target_transform_param[0] + 1e-9)
        translate_param = -target_transform_param[1]
        updated_expr = gls.Translate3D(
            gls.Scale3D(updated_expr, scale_param),
            translate_param)
        # updated_expr = gls.Scale3D(
        #     gls.Translate3D(updated_expr, translate_param),
        #     scale_param)
        # Now invert to primal
        # HACKY
        if eval_mode:
            clip = False
            quantize = False
        else:
            clip = True
            quantize = True
        new_expression = lang_specs.revert_to_current_langauge(updated_expr,
                                                               device=sketcher.device,
                                                               dtype=sketcher.dtype,
                                                               clip=clip,
                                                               quantize=quantize,
                                                               resolution=lang_specs.n_float_bins,)
    # Dictionary to keep track of updated expressions
    updated_expressions = {target_node_id: new_expression}
    # Traverse upwards from the target node to the root
    current_node_id = target_node_id
    while True:
        # Get the parent nodes
        parents = list(G.predecessors(current_node_id))

        # If no parents, this is the root
        if not parents:
            break

        # Assuming a tree structure, there should be only one parent
        parent = parents[0]

        # Recompute the expression for the parent
        func_class = G.nodes[parent]['function_class']
        params = G.nodes[parent].get('parameters', [])

        # Combine updated expressions of child nodes (if available) with original expressions and parameters
        children = list(G.successors(parent))
        args_and_params = [updated_expressions.get(
            child, G.nodes[child]['expression']) for child in children] + params

        # Update the expression for the parent node in the dictionary
        updated_expressions[parent] = func_class(*args_and_params)

        # Move up the tree
        current_node_id = parent

    # Return the updated expression at the root
    root_expression = updated_expressions[0]
    return root_expression


def graph_to_expr(G, clone_param=False):

    # Get the root node
    root = G.graph[0]

    # Perform a breadth-first search to get all nodes in BFS order
    bfs_tree = list(nx.breadth_first_search.bfs_tree(G, source=root))

    # Reverse the BFS sequence for processing from leaves to root
    bfs_tree.reverse()

    # Dictionary to store the expressions for each node
    expr_dict = {}

    # Process each node in reverse BFS order
    for node in bfs_tree:
        func_class = G.nodes[node]['function_class']
        params = G.nodes[node].get('parameters', [])
        if clone_param:
            params = [x.clone() for x in params]

        # Combine expressions of child nodes and parameters
        children = list(G.successors(node))
        args_and_params = [expr_dict.get(child, None)
                           for child in children] + params

        # Create the expression for the current node
        expr_dict[node] = func_class(*args_and_params)

    # The expression corresponding to the root node is the final result
    return expr_dict[root]
