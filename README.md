# OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features

This repo is the implementation of the following paper:

OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features<br>
Anton Osokin, Denis Sumin, Vasily Lomakin<br>
In proceedings of the European Conference on Computer Vision (ECCV), 2020

If you use our ideas, code or data, please, cite our paper ([available on arXiv](https://arxiv.org/abs/2003.06800)).

<details>
<summary>Citation in bibtex</summary>

```
@inproceedings{osokin20os2d,
    title = {{OS2D}: One-Stage One-Shot Object Detection by Matching Anchor Features},
    author = {Anton Osokin and Denis Sumin and Vasily Lomakin},
    booktitle = {proceedings of the European Conference on Computer Vision (ECCV)},
    year = {2020} }
```
</details>

## License
This software is released under the [MIT license](./LICENSE), which means that you can use the code in any way you want.

## Requirements
1. python >= 3.7
2. pytorch >= 1.4, torchvision >=0.5
3. NVIDIA GPU, tested with V100 and GTX 1080 Ti
4. Installed CUDA, tested with v10.0

See [INSTALL.md](INSTALL.md) for the package installation.

## Demo
See our [demo-notebook](./demo.ipynb) for an illustration of our method.

## Demo-API
See our [demo-API-notebook](./demo-api.ipynb) for an illustration of deploying the method in a Docker Container.
> See [Image Build instructions here](FASTAPI.md)

## 数据集
1. Grozi-3.2k dataset with our annotation (0.5GB): download from [Google Drive](https://drive.google.com/open?id=1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp) or with the [magic command](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99) and unpack to $OS2D_ROOT/data
```bash
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh data/grozi.zip 1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp
unzip data/grozi.zip -d data
```
2. 额外的零售产品测试集 (0.1GB):  download from [Google Drive](https://drive.google.com/open?id=1Vp8sm9zBOdshYvND9EPuYIu0O9Yo346J) or with the [magic command](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99) and unpack to $OS2D_ROOT/data
```bash
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh data/retail_test_sets.zip 1Vp8sm9zBOdshYvND9EPuYIu0O9Yo346J
unzip data/retail_test_sets.zip -d data
```
3. INSTRE datasets (2.3GB) are re-hosted in Center for Machine Perception in Prague (thanks to [Ahmet Iscen](http://cmp.felk.cvut.cz/~iscenahm/code.html)!): 
```bash
cd $OS2D_ROOT
wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/gnd_instre.mat -P data/instre  # 200KB
wget ftp://ftp.irisa.fr/local/texmex/corpus/instre/instre.tar.gz -P data/instre  # 2.3GB
tar -xzf data/instre/instre.tar.gz -C data/instre
```
4. 如果你想添加你自己的数据集，你应该创建一个实例 `DatasetOneShotDetection` class and then pass it into the functions creating dataloaders `build_train_dataloader_from_config` or `build_eval_dataloaders_from_cfg` from [os2d/data/dataloader.py](os2d/data/dataloader.py). See [os2d/data/dataset.py](os2d/data/dataset.py) for docs and examples.

## Trained models
We release three pretrained models:
| Name | mAP on "grozi-val-new-cl" | link |
| -- | -- | -- |
| OS2D V2-train | 90.65 | [Google Drive](https://drive.google.com/open?id=1l_aanrxHj14d_QkCpein8wFmainNAzo8) |
| OS2D V1-train | 88.71 | [Google Drive](https://drive.google.com/open?id=1ByDRHMt1x5Ghvy7YTYmQjmus9bQkvJ8g) |
| OS2D V2-init  | 86.07 | [Google Drive](https://drive.google.com/open?id=1sr9UX45kiEcmBeKHdlX7rZTSA4Mgt0A7) |

The results (mAP on "grozi-val-new-cl") can be computed with the commands given [below](#evaluation).

你可以下载预训练模型  with [the magic commands](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99):
```bash
#大概40MB大小 模型
cd $OS2D_ROOT
./os2d/utils/wget_gdrive.sh models/os2d_v2-train.pth 1l_aanrxHj14d_QkCpein8wFmainNAzo8
./os2d/utils/wget_gdrive.sh models/os2d_v1-train.pth 1ByDRHMt1x5Ghvy7YTYmQjmus9bQkvJ8g
./os2d/utils/wget_gdrive.sh models/os2d_v2-init.pth 1sr9UX45kiEcmBeKHdlX7rZTSA4Mgt0A7
```

## Evaluation
1. OS2D V2-train (best model)
为了对验证集进行快速评估，我们可以用这个脚本使用单一比例的图像（在验证集 "grozi-val-new-cl "上会得到85.58个mAP）
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-train.pth eval.scales_of_image_pyramid "[1.0]"
```
多规模评估可以得到更好的结果--下面的脚本使用默认设置，有7个规模：0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6。请注意，这种评估可能会比较慢，因为数据集中有多个规模和*多的*类。

要对具有多个规模的验证集进行评估，请运行。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-train.pth
```

2. OS2D V1-train

对验证集的运行进行评估。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model False model.use_simplified_affine_model True model.backbone_arch ResNet101 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v1-train.pth
```


3. OS2D V2-init

对验证集的运行进行评估。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False model.backbone_arch ResNet50 train.do_training False eval.dataset_names "[\"grozi-val-new-cl\"]" eval.dataset_scales "[1280.0]" init.model models/os2d_v2-init.pth
```


## Training

### Pretrained models
在这个项目中，我们不从头开始训练模型，而是从一些预训练的模型开始。关于如何获得这些模型的说明，见[models/README.md](models/README.md)。

### Best models
我们在Grozi-3.2k数据集上的V2-train模型是用这个命令训练的。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False train.objective.loc_weight 0.0 train.model.freeze_bn_transform True model.backbone_arch ResNet50 init.model models/imagenet-caffe-resnet50-features-ac468af-renamed.pth init.transform models/weakalign_resnet101_affine_tps.pth.tar train.mining.do_mining True output.path output/os2d_v2-train
```

我的测试, 按照models/README.md 下载和转换模型
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False train.objective.loc_weight 0.0 train.model.freeze_bn_transform True model.backbone_arch ResNet50 init.model models/imagenet-caffe-resnet50-features-ac468af-converted.pth init.transform models/weakalign_resnet101_affine_tps.pth.tar train.mining.do_mining True output.path output/os2d_v2-train


由于采用了困难碎片挖掘，这个过程相当缓慢。如果没有它，训练会更快，但产生的结果略差。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model True model.use_simplified_affine_model False train.objective.loc_weight 0.0 train.model.freeze_bn_transform True model.backbone_arch ResNet50 init.model models/imagenet-caffe-resnet50-features-ac468af-renamed.pth init.transform models/weakalign_resnet101_affine_tps.pth.tar train.mining.do_mining False output.path output/os2d_v2-train-nomining
```
对于V1-训练模型，我们使用这个命令。
```bash
cd $OS2D_ROOT
python main.py --config-file experiments/config_training.yml model.use_inverse_geom_model False model.use_simplified_affine_model True train.objective.loc_weight 0.2 train.model.freeze_bn_transform False model.backbone_arch ResNet101 init.model models/gl18-tl-resnet101-gem-w-a4d43db-converted.pth train.mining.do_mining False output.path output/os2d_v1-train
```
请注意，由于对整个训练集的缓存，这些运行需要大量的RAM。如果这对你不适用，你可以使用参数`train.cache_images False`，这将在飞行中加载图像，但可能很慢。还要注意的是，由于 "预热"，即在Os2dBoxCoder中计算anchor的网格，训练的第一次迭代可能会很慢。这些计算被缓存起来了，所以每一次撞击最终都会运行得更快。

For the rest of the training scripts see [below](#rerunning-experiments-on-retail-and-instre-datasets).

### Rerunning experiments
这个项目的所有实验都是用 [our job helper](./os2d/utils/launcher.py).
对于每个实验，一个程序是一个实验结构（在python中）并调用运行程序提供的几个技术功能。
See, e.g., [this file](./experiments/launcher_exp1.py) for an example.

运行结果如下
```bash
# add OS2D_ROOT to the python path - can be done, e.g., as follows
export PYTHONPATH=$OS2D_ROOT:$PYTHONPATH
# call the experiment script
python ./experiments/launcher_exp1.py LIST_OF_LAUNCHER_FLAGS
```
额外参数 in `LIST_OF_LAUNCHER_FLAGS` are parsed by [the launcher](./os2d/utils/launcher.py) and 包含一些关于运行的有用选项。
1. `--no-launch` 允许准备所有的实验脚本而不需要实际运行。
2. `--slurm` 允许用sbatch准备SLURM作业和启动（如果没有`--no-launch`）。
3. `--stdout-file` and `--stderr-file` - 文件，分别用于保存stdout和stderr（相对于实验描述中定义的log_path）。
4. 对于许多与SLURM相关的参数, see [the launcher](./os2d/utils/launcher.py).

我们的实验可以在这里找到。
1. [Experiments with OS2D](experiments/README.md)
2. [Experiments with the detector-retrieval baseline](baselines/detector_retrieval/README.md)
3. [Experiments with the CoAE baseline](baselines/CoAE/README.md)
4. [Experiments on the ImageNet dataset](experiments/README_ImageNet.md)


### Baselines
我们在此版本中增加了两条基线。
1. Class-agnostic detector + image retrieval system: see [README](baselines/detector_retrieval/README.md) for details.
2. Co-Attention and Co-Excitation, CoAE ([original code](https://github.com/timy90022/One-Shot-Object-Detection), [paper](https://arxiv.org/abs/1911.12529)): see [README](baselines/CoAE/README.md) for details.


### Acknowledgements
We would like to personally thank [Ignacio Rocco](https://www.irocco.info/), [Relja Arandjelović](http://www.relja.info/), [Andrei Bursuc](https://abursuc.github.io/), [Irina Saparina](https://github.com/saparina) and [Ekaterina Glazkova](https://github.com/EkaterinaGlazkova) for amazing discussions and insightful comments without which this project would not be possible.

This research was partly supported by Samsung Research, Samsung Electronics, by the Russian Science Foundation grant 19-71-00082 and through computational resources of [HPC facilities](https://it.hse.ru/hpc) at NRU HSE.

This software was largely inspired by a number of great repos: [weakalign](https://github.com/ignacio-rocco/weakalign), [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch), [torchcv](https://github.com/kuangliu/torchcv), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
Special thanks goes to the amazing [PyTorch](https://pytorch.org/).
