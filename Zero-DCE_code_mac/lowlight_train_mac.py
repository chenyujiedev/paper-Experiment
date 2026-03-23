# -*- coding: utf-8 -*-
# @Time    : 2026/3/7 17:40
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : lowlight_train_mac.py

import os
import argparse

import torch
import dataloader_mac
import model_mac
import Myloss_light_mac

import random
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    DCE_net = model_mac.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)

    if config.load_pretrain and config.pretrain_dir != "":
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
        print("Loaded pretrained weights from:", config.pretrain_dir)

    train_dataset = dataloader_mac.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )

    # 原始 Zero-DCE loss
    L_color = Myloss_light_mac.L_color().to(device)
    L_spa = Myloss_light_mac.L_spa().to(device)
    L_exp = Myloss_light_mac.L_exp(16, 0.6).to(device)
    L_TV = Myloss_light_mac.L_TV().to(device)

    # 新增：暗区域曝光损失
    L_dark_exp = Myloss_light_mac.DarkRegionExposureLoss(
        patch_size=config.dark_patch_size,
        target_mean=config.dark_target_mean,
        tau=config.dark_tau,
        gamma=config.dark_gamma
    ).to(device)

    #新增：光源保护
    # 新增：光源保护
    L_highlight = Myloss_light_mac.HighlightPreservationLoss(
        patch_size=config.highlight_patch_size,
        tau=config.highlight_tau,
        beta=config.highlight_beta,
        margin=config.highlight_margin
    ).to(device)

    optimizer = torch.optim.Adam(
        DCE_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    DCE_net.train()

    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}]")

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device)

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            # 原始 loss
            loss_tv = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            # 新增暗区 loss
            loss_dark = config.dark_loss_weight * torch.mean(
                L_dark_exp(img_lowlight, enhanced_image)
            )
            #光源保护
            loss_high = config.highlight_loss_weight * torch.mean(
                L_highlight(img_lowlight, enhanced_image)
            )
            loss = loss_tv + loss_spa + loss_col + loss_exp + loss_dark + loss_high


            # 总 loss
            # loss = loss_tv + loss_spa + loss_col + loss_exp + loss_dark

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print(
                    f"Epoch {epoch + 1}, Iteration {iteration + 1}, "
                    f"Loss: {loss.item():.6f}, "
                    f"TV: {loss_tv.item():.6f}, "
                    f"Spa: {loss_spa.item():.6f}, "
                    f"Col: {loss_col.item():.6f}, "
                    f"Exp: {loss_exp.item():.6f}, "
                    f"DarkExp: {loss_dark.item():.6f}"
                    f"HighPreserve: {loss_high.item():.6f}"
                )

        save_path = os.path.join(config.snapshots_folder, f"Epoch{epoch + 1}.pth")
        torch.save(DCE_net.state_dict(), save_path)
        print("Saved model to:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lowlight_images_path", type=str, default="data/train_data/")
    # parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--display_iter", type=int, default=5)
    parser.add_argument("--snapshot_iter", type=int, default=10)

    # 这次默认改成权重实验专用目录
    # parser.add_argument("--snapshots_folder", type=str, default="snapshots_tau04_weight/")
    parser.add_argument("--snapshots_folder", type=str, default="snapshots_darkexp_highprotect/")

    # parser.add_argument("--load_pretrain", type=bool, default=False)
    # parser.add_argument("--pretrain_dir", type=str, default="")
    parser.add_argument("--load_pretrain", type=bool, default=True)
    parser.add_argument("--pretrain_dir", type=str, default="snapshots_w3/Epoch8.pth")

    # =========================================================
    # 暗区实验参数
    # 这次固定 tau=0.4, gamma=2.0
    # 只调整 dark_loss_weight
    # =========================================================
    parser.add_argument("--dark_loss_weight", type=float, default=3.0)
    parser.add_argument("--dark_patch_size", type=int, default=16)
    parser.add_argument("--dark_target_mean", type=float, default=0.6)

    parser.add_argument("--dark_tau", type=float, default=0.4)
    parser.add_argument("--dark_gamma", type=float, default=2.0)


    parser.add_argument("--highlight_loss_weight", type=float, default=2.0)
    parser.add_argument("--highlight_patch_size", type=int, default=16)
    parser.add_argument("--highlight_tau", type=float, default=0.3)
    parser.add_argument("--highlight_beta", type=float, default=0.5)
    parser.add_argument("--highlight_margin", type=float, default=0.07)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    train(config)