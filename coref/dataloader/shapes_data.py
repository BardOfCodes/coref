
import os
import h5py
import torch as th
import numpy as np
from collections import defaultdict
from .synthetic_data import SynthDataset
from .synthetic_data import TEST, TRAIN, VALIDATION

split_dirs = ['03001627_chair', '04379243_table',
              '02828884_bench', '04256520_couch']

DATA_PATHS = {x: os.path.join(x, x.split(
    '_')[0]+"_%s_vox.hdf5") for x in split_dirs}

DATA_PATHS['3DCoMPaT'] = "voxel_%s.h5"
MODE_MAP = {"TRAIN": "train", "VALIDATION": "val", "TEST": "test"}
CAD_2D_FILE = 'cad/cad.h5'


class GTShapesDataset(SynthDataset):

    def __init__(self, config, subset_config, lang_conf, device, dtype, seed, *args, **kwargs):

        self.category_labels = subset_config.category_labels
        self.data_dir = subset_config.data_dir
        # self.batch_size = subset_config.batch_size

        super(GTShapesDataset, self).__init__(config, subset_config,
                                              lang_conf, device, dtype, seed, *args, **kwargs)

    def load_precompiled_data(self, *args, **kwargs):

        if self.lang_conf.n_dims == 3:
            cache = defaultdict(list)
            data_keys = []
            for category in self.category_labels:
                data_path = os.path.join(
                    self.data_dir, DATA_PATHS[category] % (MODE_MAP[self.mode]))
                hf_loader = h5py.File(data_path, 'r')
                data = hf_loader.get('voxels')
                data = preprocess_data(data, self.resolution)
                hf_loader.close()
                n_shapes = data.shape[0]
                for ind in range(n_shapes):
                    cache[category].append(data[ind])
                    data_keys.append((category, ind))

            self.data_keys = data_keys
        elif self.lang_conf.n_dims == 2:
            cache = dict()
            data_path = os.path.join(self.data_dir, CAD_2D_FILE)
            hf = h5py.File(data_path, "r")
            images = np.array(hf.get(name=f"{MODE_MAP[self.mode]}_images"))
            div = images.shape[0]
            cache["CAD"] = [images[i] for i in range(div)]
            data_keys = [("CAD", i) for i in range(div)]
            self.data_keys = data_keys

        return cache

    def __getitem__(self, index):
        index = index % len(self.data_keys)
        category, index = self.data_keys[index]
        data = self.cache[category][index]
        key = (category, index)
        return data, key

    def generate_data_keys(self):
        data_keys = []
        for key, value in self.cache.items():
            data_keys.extend([(key, i) for i in range(len(value))])
        self.data_keys = data_keys

    def to_shapes_mode(self):
        # Dummy function.
        return None

    def __len__(self):
        return len(self.data_keys)
        # return np.ceil(len(self.data_keys)/float(self.batch_size)).astype(int)


def preprocess_data(data, resolution):
    # TODO: Make this work for arbitrary resolution.
    data = data[...]

    if len(data.shape) == 2:
        # compressed form
        data_res = data.shape[1]
        data_res = np.round(data_res ** (1/3))
        data_res = int(data_res)
        data = data.reshape(-1, data_res, data_res, data_res, 1)
    else:
        data_res = data.shape[1]
    if resolution < data_res:

        max_pool = th.nn.MaxPool3d(data_res//resolution)
        small_data = max_pool(th.from_numpy(data).squeeze(-1).float())
        data = small_data.numpy()

        # data = th.from_numpy(data)# .unsqueeze(1)
        # ndim = resolution
        # a = th.arange(ndim).view(1,1,-1) * 2 * 1 + th.arange(ndim).view(1,-1,1) * 2 * ndim * 2 + th.arange(ndim).view(-1,1,1) * 2 * ((ndim * 2) ** 2)
        # b = a.view(-1,1).repeat(1, 8)
        # rs_inds = b + th.tensor([[0,1,ndim*2,ndim*2+1,(ndim*2)**2,((ndim*2)**2)+1, ((ndim*2)**2)+(ndim*2), ((ndim*2)**2)+(ndim*2)+1]])
        # data_list = []
        # for i in range(data.shape[0]):
        #     new_data = data[i].flatten()[rs_inds].max(dim=1).values.view(ndim, ndim, ndim).T
        #     data_list.append(new_data)
        # data = th.stack(data_list, 0).numpy()

    else:
        data = data[:, :, :, :, 0]
    # set data limits here:
    if len(data.shape) == 5:
        data = data[:, :, :, :, 0]
    return data
