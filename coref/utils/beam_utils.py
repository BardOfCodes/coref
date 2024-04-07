from collections import defaultdict
import torch as th
import numpy as np
import coref.language as language
import time
from coref.utils.logger import profileit

# MIN_LL_THRESH = -100000.
# MIN_LL_PEN = -1000.
# MASK_LL_PEN = -1000.

MIN_LL_THRESH = -1000
MIN_LL_PEN = -10000
MASK_LL_PEN = -1e5


def batch_beam_decode(model, batch_inp, lang_conf, batch_size, beam_size, device,
                      stochastic_beam_search=False, temperature=100, use_validity_mask=True,
                      elastic_mode=True,
                      max_returns=50):
    if use_validity_mask:
        lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                       n_integer_bins=lang_conf.n_integer_bins,
                                                       tokenization=lang_conf.tokenization)
        state_machine = lang_specs.get_batched_state_machine(lang_conf.max_canvas_count, batch_size=batch_size*beam_size,
                                                             device=device)
        delta_seq = th.arange(batch_size).to(device) * beam_size
        delta_seq = delta_seq.unsqueeze(-1).expand(-1, beam_size).flatten()

    # forward pass
    embedding_seq = model.forward_beam_init(batch_inp, beam_size)

    token_seq = th.zeros(batch_size, model.prog_seq_len,
                         device=model.device).long()
    start_token = model.token_embedding.num_embeddings - 2
    token_seq[:, 0] = start_token
    token_seq = token_seq.unsqueeze(1).expand(-1, beam_size, -1)

    loglike_seq = th.zeros(batch_size, beam_size, device=device)
    loglike_seq[:, 1:] += MIN_LL_PEN

    stop_token = model.token_embedding.num_embeddings - 1

    finished_token_seq = {i: [] for i in range(batch_size)}

    for index in range(0, model.prog_seq_len-1):

        # Stop if all very low prob.
        if (loglike_seq < MIN_LL_THRESH).all():
            break

        # Don't evaluate more than beam_size^2 candidate programs
        length_valid = [len(x) >= (beam_size ** 2)
                        for x in finished_token_seq.values()]
        if all(length_valid):
            break

        out = embedding_seq[:, :, :model.infer_start_pos + index + 1].clone()
        # in out add the token embeddings for sequences
        restricted_seq = token_seq[:, :, :index + 1]
        restricted_seq = restricted_seq.reshape(
            batch_size * beam_size, -1).contiguous()
        token_enc = model.token_embedding(restricted_seq)
        token_enc = token_enc.view(
            batch_size, beam_size, -1, token_enc.size(-1))
        out[:, :, model.infer_start_pos:] += token_enc

        out = out.view(batch_size * beam_size, out.size(2), out.size(3))
        # model forward
        cmd_output = model.unrolled_inference_forward(out, index)
        # make it random
        # cmd_output = th.rand_like(cmd_output)
        if stochastic_beam_search:
            logprobgs = model.cmd_logsmax(cmd_output)
            u_noise = th.rand_like(logprobgs)
            g_noise = -th.log(-th.log(u_noise)) / temperature
            cmd_output = (logprobgs + g_noise)

        beam_dist = model.cmd_logsmax(cmd_output)
        if use_validity_mask:
            beam_dist = state_machine.apply_validity_mask(
                beam_dist, mask_ll_pen=MASK_LL_PEN)
            # cmd_output = state_machine.apply_validity_mask(cmd_output, mask_ll_pen=MASK_LL_PEN)
        # cmd_output = cmd_output/100
        # beam_dist = model.cmd_logsmax(cmd_output)
        beam_liks, beam_choices = th.topk(beam_dist, beam_size, dim=1)
        beam_liks = beam_liks.view(batch_size, beam_size, beam_size)
        beam_choices = beam_choices.view(batch_size, beam_size, beam_size)
        # .view(beams, batch_size, beams)
        next_liks = (beam_liks + loglike_seq.view(batch_size, beam_size, 1))

        next_liks = next_liks.view(batch_size, beam_size * beam_size)
        beam_choices = beam_choices.view(batch_size, beam_size * beam_size)
        loglike_seq, ranked_beams = th.topk(next_liks, beam_size)
        # This should be redone. after setting all the ending seq to zero prob.abs

        prev_beam_id = ranked_beams // beam_size
        nt = th.gather(beam_choices, 1, ranked_beams)  # .flatten()

        # Remove
        if elastic_mode:
            fin_end = (nt == stop_token)
            if th.any(fin_end):
                batch_ids_th, beam_ids = th.where(fin_end == True)
                batch_ids = batch_ids_th.cpu().numpy().tolist()
                beam_ids = beam_ids.cpu().numpy().tolist()
                for ind, batch_id in enumerate(batch_ids):
                    beam_id = beam_ids[ind]
                    cur_prev_beam_id = prev_beam_id[batch_id, beam_id].item()
                    next_liks_beam_id = ranked_beams[batch_id, beam_id]
                    if loglike_seq[batch_id, beam_id] > MIN_LL_THRESH:
                        log_like = loglike_seq[batch_id, beam_id].item()
                        action_seq = token_seq[batch_id,
                                               cur_prev_beam_id, :index+1].cpu().numpy()
                        action_seq = np.concatenate(
                            (action_seq, [nt[batch_id, beam_id].item()]))
                        finished_token_seq[batch_id].append(
                            (log_like, action_seq))

                    next_liks[batch_id, next_liks_beam_id] += MIN_LL_PEN

                # do this only for the updated batch ids
                unique_ids = th.unique(batch_ids_th)
                unique_ids_extended = unique_ids.unsqueeze(
                    -1).expand(-1, next_liks.shape[-1])
                cur_next_likes = th.gather(next_liks, 0, unique_ids_extended)
                cur_beam_choices = th.gather(
                    beam_choices, 0, unique_ids_extended)
                loglike_seq_update, ranked_beams_update = th.topk(
                    cur_next_likes, beam_size)
                prev_beam_id_update = ranked_beams_update // beam_size
                nt_update = th.gather(cur_beam_choices, 1, ranked_beams_update)
                prev_beam_id[unique_ids] = prev_beam_id_update
                nt[unique_ids] = nt_update
                loglike_seq[unique_ids] = loglike_seq_update
                # for ind, batch_id in enumerate(unique_ids):
                #     prev_beam_id[batch_ids_th] = prev_beam_id_update[ind]
                #     nt[batch_id] = nt_update[ind]
                #     loglike_seq[batch_id] = loglike_seq_update[ind]
                # loglike_seq, ranked_beams = th.topk(next_liks, beam_size)
                # prev_beam_id = ranked_beams // beam_size
                # nt = th.gather(beam_choices, 1, ranked_beams)# .flatten()

        if use_validity_mask:
            nt_flat = nt.view(batch_size * beam_size)

            prev_beam_id_flat = prev_beam_id.view(batch_size * beam_size)
            # add the batch deltas
            state_machine.gather(prev_beam_id_flat + delta_seq)
            state_machine.update_state(nt_flat)

        restr_token_seq = token_seq[:, :, :index+1]
        ids = prev_beam_id.unsqueeze(-1).expand(-1, -1,
                                                restr_token_seq.shape[-1])

        prev_token_seq = th.gather(restr_token_seq, 1, ids)  # .flatten()
        # bseqs  = bseqs[old_index].clone()
        new_token_seq = token_seq.clone()
        new_token_seq[:, :, :index + 1] = prev_token_seq
        new_token_seq[:, :, index + 1] = nt
        token_seq = new_token_seq
        # now create the embedding seq for the next iteration:
        fin_end = (nt == stop_token)
        if th.any(fin_end):
            batch_ids, beam_ids = th.where(fin_end == True)
            batch_ids = batch_ids.cpu().numpy().tolist()
            beam_ids = beam_ids.cpu().numpy().tolist()
            for ind, batch_id in enumerate(batch_ids):
                beam_id = beam_ids[ind]
                if loglike_seq[batch_id, beam_id] > MIN_LL_THRESH:
                    log_like = loglike_seq[batch_id, beam_id].item()
                    action_seq = token_seq[batch_id,
                                           beam_id, :index+2].cpu().numpy()
                    finished_token_seq[batch_id].append((log_like, action_seq))
                loglike_seq[batch_id, beam_id] += MIN_LL_PEN

    # now sort and select top max_returns programs for each:
    for batch_id, finished_programs in finished_token_seq.items():
        finished_programs = sorted(
            finished_programs, key=lambda x: x[0], reverse=True)
        finished_programs = finished_programs[:max_returns]
        finished_token_seq[batch_id] = finished_programs
    return finished_token_seq
