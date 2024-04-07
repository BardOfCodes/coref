
import torch as th
import torch.nn.functional as F
import geolipi.symbolic as gls


def union_invert(target, other_child, index=0):
    # Doesnt matter which index is used:
    target_shape = target[..., 0]
    prior_mask = target[..., 1]
    masked_region = shape_intersection(target_shape, other_child)
    output_mask = th.logical_and(prior_mask,  ~masked_region)
    output_shape = th.stack([target_shape, output_mask], -1)
    return output_shape


def difference_invert(target, other_child, index=0):
    target_shape = target[..., 0]
    prior_mask = target[..., 1]

    if index == 0:
        output_target = target_shape
        output_mask = th.logical_and(prior_mask, ~other_child)
    else:
        output_target = ~target_shape
        masked_region = shape_union(target_shape, other_child)
        output_mask = th.logical_and(prior_mask, masked_region)
    output_shape = th.stack([output_target, output_mask], -1)
    return output_shape


def intersection_invert(target, other_child, index=0):
    target_shape = target[..., 0]
    prior_mask = target[..., 1]
    masked_region = shape_union(target_shape, other_child)
    output_mask = th.logical_and(prior_mask, masked_region)
    output_shape = th.stack([target_shape, output_mask], -1)
    return output_shape


def shape_union(obj1, obj2, *args):
    output = th.logical_or(obj1, obj2)
    return output


def shape_intersection(obj1, obj2, *args):
    output = th.logical_and(obj1, obj2)
    return output


def shape_difference(obj1, obj2):
    output = th.logical_and(obj1, ~obj2)
    return output


def shape_translate(param, input_shape, res, coords):
    # coords = th.reshape(coords, (-1, 1, 3))
    scaled_coords = coords - param
    grid = _direct_grid_operation(scaled_coords, res)
    pred_shape = _shape_sample(grid, input_shape)
    return pred_shape


def shape_scale(param, input_shape, res, coords):
    # coords = th.reshape(coords, (-1, 1, 3))
    scaled_coords = coords / param
    grid = _direct_grid_operation(scaled_coords, res)
    pred_shape = _shape_sample(grid, input_shape)
    return pred_shape


def _direct_grid_operation(scaled_coords, res):
    grid = th.stack([scaled_coords[:, :, 2], scaled_coords[:, :, 1],
                    scaled_coords[:, :, 0]], dim=2).view(1, res, res, res, 3)
    return grid


def _shape_sample(grid, input_shape):

    input_shape_sampling = input_shape.permute(3, 0, 1, 2)
    input_shape_sampling = input_shape_sampling.to(input_shape.dtype)
    input_shape_sampling = input_shape_sampling.unsqueeze(0)
    pred_shape = F.grid_sample(
        input=input_shape_sampling, grid=grid, mode='nearest',  align_corners=False)
    # pred_shape = pred_shape > 0.0
    pred_shape = pred_shape.bool()
    pred_shape = pred_shape.squeeze(0)
    grid_mask = (th.max(th.abs(grid[0]), 3)[0] <= 1)
    pred_shape[1] = th.logical_and(pred_shape[1], grid_mask)

    pred_shape = pred_shape.permute(1, 2, 3, 0)
    return pred_shape


INVERSION_MAP = {
    gls.Union: union_invert,
    gls.Intersection: intersection_invert,
    gls.Difference: difference_invert,

}
