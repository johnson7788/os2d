import os
import argparse

import torch

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg


def parse_opts():
    parser = argparse.ArgumentParser(description="对OS2D模型的训练和评估")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="配置文件地址",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="使用命令行修改配置选项, 其它所有命令",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args.config_file


def init_logger(cfg, config_file):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("OS2D", output_dir if cfg.output.save_log_to_file else None)

    if config_file:
        logger.info("加载的配置文件是 {}".format(config_file))
        with open(config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    else:
        logger.info("Config file was not provided")

    logger.info("具体的配置时:\n{}".format(cfg))

    # save config file only when training (to run multiple evaluations in the same folder)
    if output_dir and cfg.train.do_training:
        output_config_path = os.path.join(output_dir, "config.yml")
        logger.info("保存训练的配置文件到: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)


def main():
    cfg, config_file = parse_opts()
    init_logger(cfg, config_file)

    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # 创建模型
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

    # Optimizer
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)

    # 加载数据集， 数据集目录路径： data_path， eg: 'xxx/os2d/data'
    data_path = get_data_path()
    # 加载训练集
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(cfg, box_coder, img_normalization,
                                                                                   data_path=data_path)
    # 加载评估集
    dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
                                                       datasets_for_eval=datasets_train_for_eval,
                                                       data_path=data_path)

    # 开始训练（验证在里面）
    trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=dataloaders_eval)


if __name__ == "__main__":
    main()
