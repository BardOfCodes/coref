from collections import defaultdict

import numpy as np

LE_TYPES = ["BS", "PO", "CP", "CG"]


class BestProgramDataStruct:

    def __init__(self, config):
        self.n_expr_per_entry = config.n_expr_per_entry
        self.programs = defaultdict(list)

    def get_stats(self):
        program_count = 0
        obj_score = []
        for key, eval_obj in self.programs.items():
            program_count += len(eval_obj)
            for program in eval_obj:
                if program[2] in LE_TYPES:
                    obj_score.append(program[1])
        stats = {
            "n_programs": program_count,
            "mean_objective": np.mean(obj_score),
            "min_objective": np.min(obj_score),
            "max_objective": np.max(obj_score)
        }

        return stats

    def update_programs(self, new_programs, origin_type):
        for prog_obj in new_programs:
            key, eval_content, metric = prog_obj
            self.programs[key].append((eval_content, metric, origin_type))
        for key, eval_content in self.programs.items():
            eval_content = sorted(
                eval_content, key=lambda x: x[1], reverse=True)
            # Delete the ones below the WS version
            bs_entry = [x for x in eval_content if x[2] == "BS"][0]
            ws_entry = [x for x in eval_content if x[2] == "WS"]

            eval_content_sans_bs = [
                x for x in eval_content if x[2] not in ["BS", "WS"]]
            eval_content_sans_bs = [
                x for x in eval_content_sans_bs if x[1] > bs_entry[1]]
            eval_content_sans_bs = eval_content_sans_bs[:self.n_expr_per_entry]

            self.programs[key] = eval_content_sans_bs + [bs_entry] + ws_entry
        print("All entries have BS")

    def get_programs(self, avoid_keys=[]):

        all_progs = []
        for real_key, eval_obj in self.programs.items():
            key_0 = "_".join(real_key.split("_")[:-1])
            key_1 = int(real_key.split("_")[-1])
            key = (key_0, key_1)
            bs_entry = [x for x in eval_obj if x[2] == "BS"]
            ws_entry = [x for x in eval_obj if x[2] == "WS"]
            eval_content_sans_bs = [
                x for x in eval_obj if x[2] not in ["BS", "WS"]]
            real_n = min(self.n_expr_per_entry, len(eval_content_sans_bs))
            valid_progs = eval_content_sans_bs[:real_n] + \
                bs_entry[:1] + ws_entry[:1]
            for cur_prog in valid_progs:
                # cur_prog = eval_content_sans_bs[i]
                prog = cur_prog[0]
                origin_type = cur_prog[2]
                if origin_type in avoid_keys:
                    continue
                # Hack
                obj = (key, prog, 1, origin_type)
                all_progs.append(obj)
                if origin_type in LE_TYPES:
                    obj = (key, prog, 0, origin_type)
                    all_progs.append(obj)

        return all_progs

    def get_best_programs(self, avoid_keys=[]):

        all_progs = {}
        for real_key, eval_obj in self.programs.items():
            eval_content = sorted(eval_obj, key=lambda x: x[1], reverse=True)
            key_0 = "_".join(real_key.split("_")[:-1])
            key_1 = int(real_key.split("_")[-1])
            key = (key_0, key_1)
            key_str = "_".join(list(map(str, key)))

            selected_entry = eval_content[0]
            all_progs[key_str] = selected_entry[0]
        return all_progs
# Store a few more details about the program
# target_key, origin, expression, objective score
