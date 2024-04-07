
# Inverted state machine
import numpy as np
import torch as th

FLOAT_OR_END_STATE = 0  # next actions
FLOAT_ONLY_STATE = 1
DRAW_ONLY_STATE = 2
BOOL_OR_FLOAT_STATE = 3
END_ONLY_STATE = 4

BOOL_ACTION = 0
DRAW_ACTION = 1
FLOAT_ACTION = 2
START_ACTION = 3
END_ACTION = 4


class BatchedStateMachinePostF:
    def __init__(self, n_t_pre_float, n_prim_param_tokens,
                 n_float_bins, max_canvas_count,
                 batch_size, device):

        self.max_canvas_count = max_canvas_count
        self.n_float_bins = n_float_bins
        self.param_count = n_prim_param_tokens
        self.device = device
        self.batch_size = batch_size

        self.float_count = th.zeros(batch_size, dtype=th.int, device=device)
        self.bool_count = th.zeros(batch_size, dtype=th.int, device=device)
        self.canvas_count = th.zeros(batch_size, dtype=th.int, device=device)
        # bool, draw, float, start, end
        state = th.tensor([0, 0, 1, 0, 0], dtype=th.bool, device=device)
        self.state = state.unsqueeze(0).expand(batch_size, -1)

        self.n_bool_actions = 3
        self.n_draw_actions = n_t_pre_float - 3
        self.n_float_actions = self.n_float_bins

        n_actions = n_t_pre_float + 2 + self.n_float_bins

        action_type_mappers = {}
        for i in range(self.n_bool_actions):
            action_type_mappers[i] = BOOL_ACTION
        for i in range(self.n_draw_actions):
            action_type_mappers[i + self.n_bool_actions] = DRAW_ACTION
        for i in range(self.n_float_actions):
            action_type_mappers[i + self.n_bool_actions +
                                self.n_draw_actions] = FLOAT_ACTION
        start_id = self.n_bool_actions + self.n_draw_actions + self.n_float_actions
        end_id = start_id + 1
        action_type_mappers[start_id] = START_ACTION
        action_type_mappers[end_id] = END_ACTION
        self.action_type_mappers = th.tensor(
            [action_type_mappers[i] for i in range(n_actions)], dtype=th.int, device=device)

        bool_point = self.n_bool_actions
        draw_point = bool_point + self.n_draw_actions
        float_bin_point = draw_point + self.n_float_actions
        start_point = float_bin_point + 1
        end_point = start_point + 1
        self.action_size_mapper = {
            0: (0, bool_point),
            1: (bool_point, draw_point),
            2: (draw_point, float_bin_point),
            3: (float_bin_point, start_point),
            4: (start_point, end_point),
        }

    def update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        float_actions = action_types == FLOAT_ACTION
        draw_actions = action_types == DRAW_ACTION
        bool_actions = action_types == BOOL_ACTION
        end_actions = action_types == END_ACTION
        bool_draw_actions = draw_actions | bool_actions

        self.float_count[float_actions] += 1
        self.float_count[bool_draw_actions] = 0
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1

        float_sufficient = self.float_count == self.param_count

        self.state[:, 1][float_sufficient & float_actions] = True
        self.state[:, 2][(~float_sufficient) & float_actions] = True

        can_draw = self.canvas_count < self.max_canvas_count
        can_bool = self.bool_count < (self.canvas_count - 1)
        can_end = self.canvas_count == self.bool_count + 1

        self.state[:, 2][can_draw & bool_draw_actions] = True
        self.state[:, 0][can_bool & bool_draw_actions] = True
        self.state[:, 4][can_end & bool_draw_actions] = True

        self.state[end_actions, 3] = True

    def gather(self, id_list):
        self.float_count = th.gather(self.float_count, 0, id_list)  # .clone()
        self.bool_count = th.gather(self.bool_count, 0, id_list)  # .clone()
        self.canvas_count = th.gather(
            self.canvas_count, 0, id_list)  # .clone()
        state_ids = id_list.unsqueeze(1).expand(-1, self.state.size(1))
        self.state = th.gather(self.state, 0, state_ids)  # .clone()

    def apply_validity_mask(self, cmd_output, mask_ll_pen=-1e10):

        mask = self.state
        mask_list = []
        for ind, value in self.action_size_mapper.items():
            start_index, end_index = value
            cur_mask = mask[:, ind:ind+1].expand(-1, end_index - start_index)
            mask_list.append(cur_mask)
        mask = th.cat(mask_list, dim=1)
        cmd_output = th.where(mask, cmd_output, mask_ll_pen)
        return cmd_output


class BatchedStateMachinePreF(BatchedStateMachinePostF):
    def __init__(self, n_t_pre_float, n_prim_param_tokens,
                 n_float_bins, max_canvas_count,
                 batch_size, device):
        super(BatchedStateMachinePreF, self).__init__(n_t_pre_float, n_prim_param_tokens,
                                                      n_float_bins, max_canvas_count,
                                                      batch_size, device)
        state = th.tensor([1, 0, 0, 0, 0], dtype=th.bool, device=device)
        self.state = state.unsqueeze(0).expand(batch_size, -1)

    def update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        float_actions = action_types == FLOAT_ACTION
        draw_actions = action_types == DRAW_ACTION
        bool_actions = action_types == BOOL_ACTION

        self.float_count[float_actions] += 1
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1
        float_sufficient = self.float_count == self.param_count

        sub = float_sufficient & float_actions
        not_sub = (~float_sufficient) & float_actions
        self.float_count[sub] = 0

        can_end = self.bool_count + 1 == self.canvas_count
        can_bool = (self.bool_count < (self.max_canvas_count - 1)) & (~can_end)
        can_draw = (self.bool_count > self.canvas_count - 1)

        # note - don't really need to predict "end" action
        self.state[:, 0][sub & can_bool] = True
        self.state[:, 1][sub & can_draw] = True
        self.state[:, 2][sub] = False
        self.state[:, 4][sub & can_end] = True

        self.state[:, 2][not_sub] = True

        self.state[:, 0][bool_actions & can_bool] = True
        self.state[:, 1][bool_actions & can_draw] = True

        self.state[:, 2][draw_actions] = True


class BatchedStateMachineMixed(BatchedStateMachinePostF):
    """Draw parameters in prefix, and expression in postfix.
    """

    def __init__(self, n_t_pre_float, n_prim_param_tokens,
                 n_float_bins, max_canvas_count,
                 batch_size, device):
        super(BatchedStateMachineMixed, self).__init__(n_t_pre_float, n_prim_param_tokens,
                                                       n_float_bins, max_canvas_count,
                                                       batch_size, device)
        state = th.tensor([0, 1, 0, 0, 0], dtype=th.bool, device=device)
        self.state = state.unsqueeze(0).expand(batch_size, -1)

    def update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        float_actions = action_types == FLOAT_ACTION
        draw_actions = action_types == DRAW_ACTION
        bool_actions = action_types == BOOL_ACTION
        end_actions = action_types == END_ACTION

        self.float_count[float_actions] += 1
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1
        float_sufficient = self.float_count == self.param_count

        sub = float_sufficient & float_actions
        not_sub = (~float_sufficient) & float_actions
        self.float_count[sub] = 0

        can_draw = self.canvas_count < self.max_canvas_count
        can_bool = self.bool_count < (self.canvas_count - 1)
        can_end = self.canvas_count == self.bool_count + 1

        # note - don't really need to predict "end" action
        self.state[:, 0][sub & can_bool] = True
        self.state[:, 1][sub & can_draw] = True
        self.state[:, 2][sub] = False
        self.state[:, 4][sub & can_end] = True

        self.state[:, 2][not_sub] = True

        self.state[:, 0][bool_actions & can_bool] = True
        self.state[:, 1][bool_actions & can_draw] = True
        self.state[:, 4][bool_actions & can_end] = True

        self.state[:, 2][draw_actions] = True
