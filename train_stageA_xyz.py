#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A Training: sRGB -> CIE XYZ (ReRAW-style)

- 复用官方 ReRAW 的 ReRAW 模型 / DatasetSamples / utils
- 区别：
    * target 改为 XYZ patch (.npy, 3 通道)
    * 模型 out_size=3（而不是 4 通道 RGGB）
    * cfg 中 rggb_max 建议设为 1.0（假设 XYZ 已经归一到 [0,1]）

用法示例：
    python train_stageA_xyz.py -g 0 -c cfg_stageA_xyz_srgb.py
"""

import os
import time
import argparse
import multiprocessing

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from resources.models import ReRAW, hard_log_loss
from resources.dataset import DatasetSamples
from resources.utils import (
    save_model,
    save_cfg,
    test_patches,
    load_cfg,
)


def train(model, dataloader, testloader, save_path, cfg, writer):
    lr = cfg["lr"]
    lr_scaling = cfg["lr_scaling"]
    restart = cfg["restart"]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, restart, eta_min=lr * lr_scaling
    )

    # 对 outputs (多头) 和 y (聚合输出) 都用 hard_log_loss
    criterion1 = hard_log_loss()
    criterion2 = hard_log_loss()
    psnr_max = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        last_loss = 0.0

        with tqdm(dataloader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                last_loss = last_loss * 0.9 + loss.item() * 0.1
                tepoch.set_description(
                    f'------->>>> Epoch {epoch} | loss={"%.5f" % round(last_loss, 5)}'
                )

                sample, targets, context, target = (
                    data[0].cuda(non_blocking=True),
                    data[1].cuda(non_blocking=True),
                    data[2].cuda(non_blocking=True),
                    data[3].cuda(non_blocking=True),
                )

                optimizer.zero_grad()
                # ReRAW 前向：sample = 局部 sRGB，context = 全图/大 crop sRGB
                y, outputs, _ = model(sample, context)

                # outputs: 所有 gamma head 的输出（按 DatasetSamples 拼起来）
                # targets: 对应 gamma 版本的 XYZ 监督
                loss1 = criterion1(outputs, targets)
                # y: 聚合后的最终 XYZ 预测
                loss2 = criterion2(y, target)

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

        # TensorBoard: 记录最后一个 batch 的 loss
        writer.add_scalar("loss", loss.item(), epoch)

        # 用 patch-level PSNR 在 XYZ 域上评估
        psnr = test_patches(model, testloader, gammas=cfg["gammas"])
        writer.add_scalar("psnr_mean_patch", psnr, epoch)

        # 保存 best.pth
        if psnr > psnr_max:
            save_model(model, os.path.join(save_path, "best.pth"))
            psnr_max = psnr

        lr_scheduler.step()


# ######################################################################################################


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Train Stage A (sRGB -> CIE XYZ) with ReRAW-style reverse ISP."
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU id used to train."
    )
    parser.add_argument(
        "-c", "--cfg", type=str, default="", help="config file (e.g. cfg_stageA_xyz_srgb.py)"
    )
    args = parser.parse_args()

    cuda_no = str(args.gpu)
    device = torch.device(f"cuda:{cuda_no}")
    torch.cuda.set_device(device)

    # 载入 cfg_stageA_xyz_srgb.py 中的 dataset / cfg_sample / cfg_train
    dataset, cfg_sample, cfg_train = load_cfg(args.cfg)

    # DatasetSamples 仍然读取 sample/context/target 三个目录
    folder_data_train = DatasetSamples(dataset, cfg_sample, cfg_train, mode="train")
    folder_data_test = DatasetSamples(dataset, cfg_sample, cfg_train, mode="test")

    trainloader = DataLoader(
        folder_data_train,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    testloader = DataLoader(
        folder_data_test,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # ***** 关键修改：out_size=3，用于输出 XYZ 三通道 *****
    model = ReRAW(
        in_size=3,  # sRGB
        out_size=3,  # CIE XYZ 三通道
        target_size=cfg_train["target_size"],
        hidden_size=cfg_train["hidden_size"],
        n_layers=cfg_train["depth"],
        gammas=cfg_train["gammas"],
    )

    model.cuda()

    TIME_STAMP = int(time.time())
    project_name = f"{TIME_STAMP}"

    # workspace 目录
    save_path = os.path.join(cfg_train["save_path"], project_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print("Saving model at:", save_path)
    writer = SummaryWriter(os.path.join(cfg_train["tensorboard_path"], project_name))

    # 把当前使用的 cfg 也存一份，便于复现实验
    save_cfg(cfg_train, cfg_sample, dataset, os.path.join(save_path, "cfg.py"))

    # 开始训练
    train(model, trainloader, testloader, save_path, cfg_train, writer)
