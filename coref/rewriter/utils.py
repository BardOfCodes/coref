from collections import defaultdict
import numpy as np
# given an amount sample programs from bpds.


def sample_from_bpds(bpds, selection_ratio, valid_origin, n_prog_per_item):

    selected_expressions = []

    # merge all based on target
    merged_dict = defaultdict(list)
    for key, program_list in bpds.programs.items():
        new_key = key
        for program in program_list:
            origin = program[2]
            if origin in valid_origin:
                merged_dict[new_key].extend(program_list)
    # dict of expression,
    sorted_dict = {}
    for key, value in merged_dict.items():
        value.sort(key=lambda x: x[1], reverse=True)
        # if remove_do_fail:
        #     value = [x for x in value if not x['do_fail']]
        # if remove_cs_fail:
        #     value = [x for x in value if not x['cs_fail']]
        if value:
            sorted_dict[key] = value

    keys = list(sorted_dict.keys())
    n_keys = len(keys)
    sample_count = min(
        int(selection_ratio * len(bpds.programs.keys())), n_keys)

    rand_indexes = np.random.choice(range(n_keys), sample_count, replace=False)

    for ind in rand_indexes:
        key = keys[ind]
        value = sorted_dict[key]  # Valu
        cur_n = min(len(value), n_prog_per_item)
        selected_dict = value[:cur_n]
        for program in selected_dict:
            program_dict = {
                "expression": program[0],
                "program_key": key,
                "prev_obj": program[1]
            }
            selected_expressions.append(program_dict)

    return selected_expressions


def transform_to_tunable(variable_list, language_specs):
    params = []
    parsed_variables = []
    for cur_var in variable_list:
        param, inverted_variable = language_specs.invert_variable(cur_var)
        params.append(param)
        parsed_variables.append(inverted_variable)

    return params, parsed_variables


def params_from_variables(variable_list, tensor_list, language_specs):
    params = []
    for ind, inverted_variable in enumerate(variable_list):
        info = tensor_list[ind]
        cur_var = (inverted_variable, info[1], info[2])
        param = language_specs.revert_variable(cur_var)
        params.append(param)

    return params
