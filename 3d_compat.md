# Instructions for running SIRI/PLAD on 3D Compat challenge dataset

The Visual Shape Inference Challenge is being organized as a part of the ...

## Loading the Dataset

Please download the data from the Challenge website. Save the data in the following format:

```bash
data_dir/
  | -- 3d_csg/...
  | -- 3DCoMPaT/
  | ----- | -- voxel_train.h5
  | ----- | -- voxel_val.h5
  | ----- | -- voxel_test.h5
```

## Installation

Please follow the instruction in `README.md` for installation.

## Training

1. Pretrain the model on synthetic data. This will run the pretrain, and the best model will be saved at `project_dir/models/pretrain/best_model.pt`.

```bash
python scripts/train.py --config-file configs/pretrain.py --name pretrain 
```

2. Finetune with SIRI: Let `$model_path` denote the path of the best pretrained model from step 1.

```bash
python scripts/train.py --config-file configs/siri.py --name siri --cfg.plad.starting_weights $model_path --cfg.ws_config.starting_weights $model_path --target 3DCoMPaT
```

## Inference

Inference (with rewriting) depends on two pickle files which are generated during the training. `$model_path` which is the "best" model on the validation set, and `$subexpr_cache_path` which are all the subexpressions discovered during the code grafting process (The previous command would save it as `project_dir/models/siri/all_subexpr.pkl`). With these two files, you can run inference on the test set using:

```bash
python scripts/eval_sequential.py --config-file configs/eval.py --name eval --cfg.siri.rewriters.CGRewriter.cache_config.subexpr_load_path $subexpr_cache_path --cfg.trainer.load_weights $model_path --target 3DCoMPaT
```

Note that we run this process sequentially (no batched beam search or multiple processes for PO Rewriter) so that we can measure the inference time of the system. The average per-shape inference time will be printed at the end of the evaluation. This will aid with classifying the approach. This should be around.
This will save all the expressions in a pickle file `project_dir/logs/eval/expressions_{iter}.pkl`. You can convert it to the json file format required in the submission simply as follows:

```python
import json
import _pickle as cPickle

expr_file = "" # Add the expr.pkl filepath here.
output_file  = "*.json" # The output file.
exprs = cPickle.load(open(expr_file, 'rb'))
expr_strings = [x[1][1] for x in exprs]

with open(output_file, "w") as f:
    json.dump(expr_strings, f)

```

Finally submit the file to the evaluation server [here](). Remeber to report the measured `per shape inference time`.

**Note**: You can run the process on the validation set to generate the programs for the val set, and use the evaluation scripts provided in the [VSIC]() repository to measure the evaluation metrics on the val set.
