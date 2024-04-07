
# Inverted state machine
from collections import defaultdict
import numpy as np
import torch as th

BOOL_ACTION = 0
MOD_ACTION = 1
DRAW_ACTION = 2
FLOAT_ACTION = 3
INT_ACTION = 4
START_ACTION = 5
END_ACTION = 6


class MacroBatchedStateMachinePostF:
    def __init__(self, n_float_bins, n_integer_bins, op_specs,
                 max_canvas_count, batch_size, device, max_mod_count=50,
                 compile=False):

        self.max_canvas_count = max_canvas_count
        self.n_float_bins = n_float_bins
        self.n_integer_bins = n_integer_bins
        self.max_mod_count = max_mod_count

        self.device = device
        self.batch_size = batch_size

        n_cmds = len(op_specs)
        n_actions = n_cmds + self.n_float_bins + self.n_integer_bins + 2

        self.n_floats = [x[0] for x in op_specs.values()]
        self.n_integers = [x[1] for x in op_specs.values()]
        n_canvas = [x[2] for x in op_specs.values()]
        self.max_floats = max(self.n_floats)
        self.max_integers = max(self.n_integers)
        self.max_req_canvas = max(n_canvas)
        param_states = {}

        # get n_floats, n_integers, and at least n_canvas
        for i in range(self.max_floats+1):
            for j in range(self.max_integers+1):
                for k in range(self.max_req_canvas+1):
                    key = (i, j, k)
                    param_states[key] = [1 if y[0] == i and y[1] ==
                                         j and y[2] <= k else 0 for x, y in op_specs.items()]

        param_state_tensor = th.zeros(
            (self.max_floats+1, self.max_integers+1, self.max_req_canvas+1, n_cmds), dtype=th.bool, device=device)

        for key, value in param_states.items():
            i, j, k = key
            param_state_tensor[i, j, k] = th.tensor(
                value, dtype=th.bool, device=device)
        self.param_state_tensor = param_state_tensor

        self.float_count = th.zeros(batch_size, dtype=th.long, device=device)
        self.bool_count = th.zeros(batch_size, dtype=th.long, device=device)
        self.canvas_count = th.zeros(batch_size, dtype=th.long, device=device)
        self.integer_count = th.zeros(batch_size, dtype=th.long, device=device)
        self.mod_count = th.zeros(batch_size, dtype=th.long, device=device)
        state_size = n_cmds + 2 + 2  # cmds + float/int + start/stop.

        action_type_mappers = {}
        action_type_count = defaultdict(int)

        for ind, (key, value) in enumerate(op_specs.items()):
            action_type_mappers[ind] = value[-1]
            action_type_count[value[-1]] += 1

        for i in range(self.n_float_bins):
            action_type_mappers[i + n_cmds] = FLOAT_ACTION
        for i in range(self.n_integer_bins):
            action_type_mappers[i + n_cmds + self.n_float_bins] = INT_ACTION

        start_id = n_cmds + self.n_float_bins + self.n_integer_bins
        end_id = start_id + 1
        action_type_mappers[start_id] = START_ACTION
        action_type_mappers[end_id] = END_ACTION

        self.action_type_mappers = th.tensor(
            [action_type_mappers[i] for i in range(n_actions)], dtype=th.int, device=device)

        n_bool_actions = action_type_count[0]
        n_mod_actions = action_type_count[1]
        n_draw_actions = action_type_count[2]
        # assumption that the op_seq is ordered
        self.bool_start = 0
        self.bool_end = n_bool_actions
        self.mod_start = n_bool_actions
        self.mod_end = n_bool_actions + n_mod_actions
        self.draw_start = n_bool_actions + n_mod_actions
        self.draw_end = n_bool_actions + n_mod_actions + n_draw_actions

        state = th.zeros(state_size, dtype=th.bool, device=device)
        state[self.draw_start:self.draw_end] = True
        self.state = state.unsqueeze(0).expand(batch_size, -1)

        self.n_cmds = n_cmds
        self.n_actions = n_actions
        self.compile_commands(compile)

    def compile_commands(self, compile):

        if compile:
            self.update_state = th.compile(self._update_state)
            self.gather = th.compile(self._gather)
            self.apply_validity_mask = th.compile(self._apply_validity_mask)
        else:
            self.update_state = self._update_state
            self.gather = self._gather
            self.apply_validity_mask = self._apply_validity_mask

    def _update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        bool_actions = action_types == BOOL_ACTION
        mod_actions = action_types == MOD_ACTION
        draw_actions = action_types == DRAW_ACTION
        float_actions = action_types == FLOAT_ACTION
        integer_actions = action_types == INT_ACTION
        end_actions = action_types == END_ACTION

        cmd_actions = draw_actions | bool_actions | mod_actions

        self.float_count[float_actions] += 1
        self.integer_count[integer_actions] += 1

        self.float_count[cmd_actions] = 0
        self.integer_count[cmd_actions] = 0
        # global vars
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1
        self.mod_count[mod_actions] += 1

        clipped_canvas_count = th.clamp(
            self.canvas_count, 0, self.max_req_canvas)
        # All possible cmds at this time.
        param_state = self.param_state_tensor[self.float_count,
                                              self.integer_count, clipped_canvas_count]
        self.state[:, :self.n_cmds] = param_state.clone()

        float_sufficient_1 = self.float_count == self.max_floats
        float_sufficient_2 = self.float_count == 1
        int_sufficient = self.integer_count == self.max_integers
        float_sufficient = th.where(
            int_sufficient, float_sufficient_2, float_sufficient_1)

        cant_bool = self.bool_count == (self.canvas_count - 1)
        cant_mod = self.mod_count == self.max_mod_count
        cant_draw = self.canvas_count == self.max_canvas_count
        float_validity = (~float_sufficient) & (~cant_mod)
        int_validity = (~int_sufficient) & (
            self.float_count == 0) & (~cant_mod)

        self.state[:, self.n_cmds][float_validity] = True
        self.state[:, self.n_cmds + 1][int_validity] = True

        self.state[:, self.bool_start:self.bool_end][cant_bool] = False
        self.state[:, self.mod_start:self.mod_end][cant_mod] = False
        self.state[:, self.draw_start:self.draw_end][cant_draw] = False

        can_end = self.canvas_count == self.bool_count + 1
        self.state[:, self.n_cmds+3][can_end & cmd_actions] = True
        # on end allow start token for processing.
        self.state[end_actions, self.n_cmds+2] = True

    def _gather(self, id_list):
        self.float_count = th.gather(self.float_count, 0, id_list)  # .clone()
        self.integer_count = th.gather(self.integer_count, 0, id_list)
        self.bool_count = th.gather(self.bool_count, 0, id_list)  # .clone()
        self.canvas_count = th.gather(
            self.canvas_count, 0, id_list)  # .clone()
        self.mod_count = th.gather(self.mod_count, 0, id_list)  # .clone()
        state_ids = id_list.unsqueeze(1).expand(-1, self.state.size(1))
        self.state = th.gather(self.state, 0, state_ids)  # .clone()

    def _apply_validity_mask(self, cmd_output, mask_ll_pen=-1e10):

        cmd_mask = self.state[:, :self.n_cmds]
        float_mask = self.state[:, self.n_cmds:self.n_cmds +
                                1].expand(-1, self.n_float_bins)
        integer_mask = self.state[:, self.n_cmds +
                                  1:self.n_cmds+2].expand(-1, self.n_integer_bins)
        start_mask = self.state[:, self.n_cmds+2:self.n_cmds+3]
        end_mask = self.state[:, self.n_cmds+3:self.n_cmds+4]
        mask = th.cat((cmd_mask, float_mask, integer_mask,
                      start_mask, end_mask), dim=1)
        cmd_output = th.where(mask, cmd_output, mask_ll_pen)

        return cmd_output


class MacroBatchedStateMachinePreF(MacroBatchedStateMachinePostF):

    def __init__(self, n_float_bins, n_integer_bins, op_specs,
                 max_canvas_count, batch_size, device, max_mod_count=50):
        super(MacroBatchedStateMachinePreF, self).__init__(n_float_bins, n_integer_bins, op_specs,
                                                           max_canvas_count, batch_size, device, max_mod_count)

        n_cmds = len(op_specs)
        state_size = n_cmds + 2 + 2  # cmds + float/int + start/stop.

        n_actions = n_cmds + self.n_float_bins + self.n_integer_bins + 2

        state = th.zeros(state_size, dtype=th.bool, device=device)
        state[self.mod_start:self.mod_end] = True
        state[self.bool_start:self.bool_end] = True
        self.state = state.unsqueeze(0).expand(batch_size, -1)

        self.float_req = th.zeros(batch_size, dtype=th.int64, device=device)
        self.int_req = th.zeros(batch_size, dtype=th.int64, device=device)

        self.cmd_to_floats = th.zeros(n_actions, dtype=th.long, device=device)
        cmd_to_floats = th.tensor(
            [x[0] for x in op_specs.values()], dtype=th.long, device=device)
        self.cmd_to_floats[:n_cmds] = cmd_to_floats

        self.cmd_to_integers = th.zeros(
            n_actions, dtype=th.long, device=device)
        cmd_to_integers = th.tensor(
            [x[1] for x in op_specs.values()], dtype=th.long, device=device)
        self.cmd_to_integers[:n_cmds] = cmd_to_integers

    def _update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        bool_actions = action_types == BOOL_ACTION
        mod_actions = action_types == MOD_ACTION
        draw_actions = action_types == DRAW_ACTION
        float_actions = action_types == FLOAT_ACTION
        integer_actions = action_types == INT_ACTION
        end_actions = action_types == END_ACTION

        # cmd_actions = draw_actions | bool_actions | mod_actions
        # float_int_actions = float_actions | integer_actions

        self.float_count[float_actions] += 1
        self.integer_count[integer_actions] += 1

        # self.float_count[cmd_actions] = 0
        # self.integer_count[cmd_actions] = 0
        # global vars
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1
        self.mod_count[mod_actions] += 1

        actions_id = actions.long()
        self.float_req += self.cmd_to_floats[actions_id]
        self.int_req += self.cmd_to_integers[actions_id]

        # REQ FLOAT and REQ INT
        float_req_sat = self.float_count == self.float_req
        int_req_sat = self.integer_count == self.int_req
        all_sat = float_req_sat & int_req_sat
        # Float and Ints
        self.state[:, self.n_cmds][~float_req_sat] = True
        self.state[:, self.n_cmds + 1][float_req_sat & ~int_req_sat] = True

        self.state[:, :self.n_cmds][all_sat] = True

        can_end = self.canvas_count == self.bool_count + 1
        cant_bool = th.logical_or(self.bool_count == (
            self.max_canvas_count - 1), can_end)
        cant_draw = th.logical_or(
            self.canvas_count == self.bool_count + 1, can_end)
        cant_mod = th.logical_or(
            self.mod_count == self.max_mod_count, cant_draw)

        self.state[:, self.bool_start:self.bool_end][cant_bool] = False
        self.state[:, self.mod_start:self.mod_end][cant_mod] = False
        self.state[:, self.draw_start:self.draw_end][cant_draw] = False

        self.state[:, self.n_cmds+3][can_end & draw_actions] = True
        self.state[end_actions, self.n_cmds+2] = True

    def _gather(self, id_list):
        super(MacroBatchedStateMachinePreF, self)._gather(id_list)
        self.float_req = th.gather(self.float_req, 0, id_list)  # .clone()
        self.int_req = th.gather(self.int_req, 0, id_list)  # .clone()


class MacroBatchedStateMachineMixed(MacroBatchedStateMachinePreF):

    def __init__(self, n_float_bins, n_integer_bins, op_specs,
                 max_canvas_count, batch_size, device, max_mod_count=50):
        super(MacroBatchedStateMachineMixed, self).__init__(n_float_bins, n_integer_bins, op_specs,
                                                            max_canvas_count, batch_size, device, max_mod_count)

        n_cmds = len(op_specs)
        state_size = n_cmds + 2 + 2  # cmds + float/int + start/stop.

        state = th.zeros(state_size, dtype=th.bool, device=device)
        state[self.draw_start:self.draw_end] = True
        self.state = state.unsqueeze(0).expand(batch_size, -1)

    def _update_state(self, actions):
        self.state[:, :] = False

        action_types = self.action_type_mappers[actions]
        bool_actions = action_types == BOOL_ACTION
        mod_actions = action_types == MOD_ACTION
        draw_actions = action_types == DRAW_ACTION
        float_actions = action_types == FLOAT_ACTION
        integer_actions = action_types == INT_ACTION
        end_actions = action_types == END_ACTION

        cmd_actions = draw_actions | bool_actions | mod_actions
        # float_int_actions = float_actions | integer_actions

        self.float_count[float_actions] += 1
        self.integer_count[integer_actions] += 1

        # self.float_count[cmd_actions] = 0
        # self.integer_count[cmd_actions] = 0
        # global vars
        self.canvas_count[draw_actions] += 1
        self.bool_count[bool_actions] += 1
        self.mod_count[mod_actions] += 1

        self.float_req += self.cmd_to_floats[actions]
        self.int_req += self.cmd_to_integers[actions]

        # REQ FLOAT and REQ INT
        float_req_sat = self.float_count == self.float_req
        int_req_sat = self.integer_count == self.int_req
        all_sat = float_req_sat & int_req_sat
        # Float and Ints
        self.state[:, self.n_cmds][~float_req_sat] = True
        self.state[:, self.n_cmds + 1][float_req_sat & ~int_req_sat] = True

        self.state[:, :self.n_cmds][all_sat] = True

        can_end = self.canvas_count == self.bool_count + 1
        cant_draw = self.canvas_count == self.max_canvas_count
        cant_bool = self.bool_count == (self.canvas_count - 1)
        cant_mod = self.mod_count == self.max_mod_count

        self.state[:, self.bool_start:self.bool_end][cant_bool] = False
        self.state[:, self.mod_start:self.mod_end][cant_mod] = False
        self.state[:, self.draw_start:self.draw_end][cant_draw] = False

        self.state[:, self.n_cmds+3][can_end & all_sat] = True
        self.state[end_actions, self.n_cmds+2] = True
