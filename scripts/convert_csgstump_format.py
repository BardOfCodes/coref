"""Script for mapping CSGStump to Geolipi format."""

from geolipi.symbolic import *
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShapeNet
from simple_model import ReducedCSGStumpNet
from loss import Loss
from config import Config
import numpy as np
from model import CSGStumpNet
from utils import point_inside_box
import argparse
from marchingcube import MarchingCubes
from scipy.spatial.transform import Rotation as R
import _pickle as cPickle
from pathlib import Path
import torch as th
import sys
sys.path.insert(0, "/home/aditya/projects/iccv_23/repos/geolipi")


def generate_mesh(prim_params, intersections, unions, csg_func, surface_point_cloud, file_prefix, file_start_ind, iso_value=0.5):
    def occ_func(sample_points): return (csg_func(sample_points, prim_params,
                                                  intersections, unions, is_training=False)[0]).detach().cpu().numpy()
    def padded_occ_func(sample_points): return occ_func(
        sample_points) * point_inside_box(sample_points, surface_point_cloud).detach().cpu().numpy()
    mc = MarchingCubes(config.real_size, config.test_size, use_pytorch=True)
    mc.batch_export_mesh(file_prefix, file_start_ind,
                         surface_point_cloud.shape[0], padded_occ_func, iso_value)


def get_boxes(n_objs, box_params):
    primitive_list = []
    for prim_ind in range(n_objs):
        params = get_transform_params(box_params[:, prim_ind])
        # Adjust the scales
        box_expr = Translate3D(EulerRotate3D(Scale3D(NoParamCuboid3D(),
                                                     params[Scale3D]),
                                             params[EulerRotate3D]),
                               params[Translate3D])
        primitive_list.append(box_expr)
    return primitive_list


def get_transform_params(box_params):
    quaternion = box_params[:4]
    translate = box_params[4:7]
    box_scale = box_params[7:]
    translate = translate * 2
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=False)
    # euler[-1]  = - euler[-1]
    box_scale = np.abs(box_scale * 4)

    transforms_params = {
        Translate3D: (translate[-1].item(), translate[-2].item(), translate[-3].item()),
        EulerRotate3D: (euler[0].item(), euler[1].item(), euler[2].item()),
        Scale3D: (box_scale[-1].item(),
                  box_scale[-2].item(), box_scale[-3].item())
    }
    return transforms_params

# output is CSG-Stump expressions in Geolipi.


def convert_to_prog(prim_params, intersections, unions):
    # For the simple model we have only one type of primitive
    n_objs = prim_params.shape[0]
    n_primitives = prim_params.shape[2]
    programs = []
    for obj_ind in range(n_objs):
        box_params = prim_params[obj_ind].cpu().numpy()
        cur_prims = get_boxes(n_primitives, box_params)
        cur_unions = unions[obj_ind].cpu().numpy()
        cur_intersection = intersections[obj_ind].cpu().numpy()
        un_dim = cur_unions.shape[0]
        in_dim = cur_intersection.shape[0]
        union_candidates = []
        for un_ind in range(un_dim):
            if cur_unions[un_ind] == 1:
                # Add this intersection to the list:
                cur_intersection_row = cur_intersection[:, un_ind]
                inter_candidates = []
                for in_ind in range(in_dim):
                    if cur_intersection_row[in_ind] == 1:
                        inter_candidates.append(cur_prims[in_ind])
                inter = Intersection(*inter_candidates)
                union_candidates.append(inter)
        program = Union(*union_candidates)
        programs.append(program)
    return programs


def eval(config, args):
    test_dataset = ShapeNet(partition='test', category=config.category, shapenet_root=config.dataset_root,
                            balance=config.balance, num_surface_points=config.num_surface_points, num_sample_points=config.num_sample_points)
    test_loader = DataLoader(test_dataset, pin_memory=True, num_workers=2,
                             batch_size=config.test_batch_size_per_gpu*config.num_gpu, shuffle=False, drop_last=True)

    device = torch.device("cuda")
    # model = CSGStumpNet(config).to(device)
    model = ReducedCSGStumpNet(config).to(device)
    pre_train_model_path = './checkpoints/%s/models/model.th' % config.experiment_name
    assert os.path.exists(pre_train_model_path), "Cannot find pre-train model for experiment: {}\nNo such a file: {}".format(
        config.experiment_name, pre_train_model_path)
    model.load_state_dict(torch.load(
        './checkpoints/%s/models/model.th' % config.experiment_name))
    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    criterion = Loss(config)
    model.eval()
    start_time = time.time()
    test_iter = 0
    intermediates = [[], [], []]
    all_programs = []
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    file_prefix = os.path.join(*[args.save_dir, config.experiment_name])
    with torch.no_grad():
        for test_iter, (surface_pointcloud, testing_points) in enumerate(test_loader):
            surface_pointcloud = surface_pointcloud.to(device)
            testing_points = testing_points.to(device)
            prim_params, intersections, unions = model.forward_save_mode(
                surface_pointcloud.transpose(2, 1), testing_points[:, :, :3], is_training=False)
            if args.save_mesh:
                file_start_ind = test_iter*surface_pointcloud.shape[0]
                # TODO: Fix this.
                generate_mesh(prim_params, intersections, unions, model.csg_stump,
                              surface_pointcloud.transpose(2, 1), file_prefix, file_start_ind, iso_value=0.5)
            if args.save_intermediate:
                intermediates[0].append(prim_params.cpu())
                intermediates[1].append(intersections.cpu())
                intermediates[2].append(unions.cpu())
            if args.save_programs:
                programs = convert_to_prog(prim_params, intersections, unions)
                all_programs.extend(programs)

    if args.save_intermediate:
        intermediates[0] = th.cat(intermediates[0], dim=0)
        intermediates[1] = th.cat(intermediates[1], dim=0)
        intermediates[2] = th.cat(intermediates[2], dim=0)
        print(
            f"Saving intermediates at {os.path.join(args.save_dir, 'intermediates.pt')}")
        torch.save(intermediates, os.path.join(
            args.save_dir, 'intermediates.pt'))
    if args.save_programs:
        print(
            f"Saving programs at {os.path.join(args.save_dir, 'programs.pkl')}")
        cPickle.dump(all_programs, open(
            os.path.join(args.save_dir, 'programs.pkl'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EvalPartAwareReconstruction')
    parser.add_argument('--config-path', type=str, default='./configs/config_default.json', metavar='N',
                        help='config_path')
    parser.add_argument('--save-mesh', action='store_true')
    parser.add_argument('--save-intermediate', action='store_true')
    parser.add_argument('--save-programs', action='store_true')
    parser.add_argument('--save-dir', type=str)
    args = parser.parse_args()
    config = Config((args.config_path))
    eval(config, args)
