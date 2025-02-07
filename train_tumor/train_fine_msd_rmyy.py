import argparse
import os
import sys
import traceback

from pytorch_metric_learning import losses

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from models.RandConv import RandConvModule
from data.dataloader import Generalize_Tumor_Skeleton_in_Tumor_MSD_RMYY
from utils.get_patches import (
    get_pos_patch_from_skeleton_multi_batch,
    get_neg_patch_from_skeleton_multi_batch,
)

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.util import (
    adjust_learning_rate_D,
    seed_torch,
    save_arg,
    join,
    exists,
    get_time,
)


def get_args():
    parser = argparse.ArgumentParser(description="3D segmentation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        metavar="N",
        help="random seed (default: 2022)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        metavar="N",
        help="number of epochs to train_tumor (default: 500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=int,
        default=0.0005,
        metavar="N",
        help="weight-decay (default: 0.0001)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="N",
        help="input visible devices for training (default: 0)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        metavar="str",
        help="Optimizer (default: Adam)",
    )
    parser.add_argument(
        "--time", type=str, default="time", metavar="str", help="cur_time"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="msd,rmyy",
        metavar="str",
        help="dataset for training",
    )
    parser.add_argument(
        "--workers", type=int, default=16, metavar="N", help="num_worker (default: 0)"
    )
    parser.add_argument("--log", action="store_false", help="write logs or not")
    return parser.parse_args()


args = get_args()
seed_torch(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj_DT_CCLS_RC
from models.loss import get_contrastive_loss_v2, DSC_loss, BCELoss


def generalize_train_pool_u_conresnet(args):
    log_file_name = "train_logs.txt"

    checkpoint_path = join("../checkpoints", args.time)
    if args.log:
        if not exists(join("../checkpoints", args.time)):
            os.mkdir(join("../checkpoints", args.time))
        save_arg(args, join("../checkpoints", args.time, log_file_name))

    model = U_CorResNet_Fix_Contrastive_Proj_DT_CCLS_RC().cuda().train()
    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    dset_train = Generalize_Tumor_Skeleton_in_Tumor_MSD_RMYY(
        data=args.data, augment=True, sdf=True
    )
    train_loader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    print("#################################")
    print("Training Data : ", len(train_loader.dataset))
    print("dataset:", args.data)
    print("epoch:", args.epochs)
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("optimizer:", args.optimizer)
    print("seed:", args.seed)
    print("#################################")

    # Defining Loss Function
    criterion_dsc = DSC_loss()
    criterion_bce = BCELoss()
    criterion_cls = nn.BCELoss(size_average=True)
    bgrTh = nn.Threshold(0, 1)
    tarTh = nn.Threshold(0.999999, 0)
    criterion_mse = nn.MSELoss()
    model.train()

    rand_conv = RandConvModule(
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 5, 7],
        mixing=True,
        identity_prob=0.0,
        rand_bias=True,
        distribution="kaiming_normal",
        data_mean=0.0,
        data_std=1.0,
        clamp_output=True,
    )
    rand_conv = rand_conv.cuda()

    # train_tumor for some epochs
    for i in range(args.epochs):

        loss_dsc_list = []
        loss_bce_list = []
        loss_con_list = []
        loss_cls_list = []
        loss_dist_list = []
        loss_mse_list = []
        loss_dsc_dist_list = []

        with tqdm(train_loader, ncols=120) as t:
            for (
                    batch_idx,
                    (image, mask, mask_pool, mask_dilation, skeleton_lee, gt_dis),
            ) in enumerate(t):

                t.set_description("%s" % i)
                image, mask, mask_pool = (
                    image.cuda().float(),
                    mask.cuda().float(),
                    mask_pool.cuda().float(),
                )
                mask_dilation, skeleton_lee, gt_dis = (
                    mask_dilation.cuda().float(),
                    skeleton_lee.cuda().float(),
                    gt_dis.cuda().float(),
                )

                assert mask_dilation.shape == skeleton_lee.shape
                assert mask_pool.shape == skeleton_lee.shape

                with torch.no_grad():
                    rand_conv.randomize()
                    image1 = rand_conv(image)
                    rand_conv.randomize()
                    image2 = rand_conv(image)

                    for bt in range(image.size(0)):
                        image1[bt] = (image1[bt] - image1[bt].min()) / (
                                image1[bt].max() - image1[bt].min()
                        )
                        image2[bt] = (image2[bt] - image2[bt].min()) / (
                                image2[bt].max() - image2[bt].min()
                        )

                images = torch.cat((image1, image2), dim=0)
                masks = torch.cat((mask, mask), dim=0)
                mask_pools = torch.cat((mask_pool, mask_pool), dim=0)
                mask_dilations = torch.cat((mask_dilation, mask_dilation), dim=0)
                skeleton_lees = torch.cat((skeleton_lee, skeleton_lee), dim=0)
                gt_diss = torch.cat((gt_dis, gt_dis), dim=0)

                if torch.isnan(images).any():
                    images = image
                    masks = mask
                    mask_pools = mask_pool
                    mask_dilations = mask_dilation
                    skeleton_lees = skeleton_lee
                    gt_diss = gt_dis

                try:
                    optimizer.zero_grad()

                    cls_label = torch.squeeze(masks).view(
                        images.size(0) * images.size(2), -1
                    )
                    cls_label = torch.sum(cls_label, dim=-1)
                    cls_label = cls_label > 10

                    output, emb, out_dis, cls = model(images)
                    out_put_sig = torch.sigmoid(output)
                    loss_dsc = criterion_dsc(out_put_sig, masks)
                    loss_bce = criterion_bce(output, masks)

                    loss_cls = criterion_cls(
                        torch.sigmoid(cls).squeeze(), cls_label.float()
                    )
                    loss = loss_dsc + loss_bce + loss_cls
                    loss_dsc_list.append(loss_dsc.item())
                    loss_bce_list.append(loss_bce.item())
                    loss_cls_list.append(loss_cls.item())

                    pos_patch = get_pos_patch_from_skeleton_multi_batch(
                        emb, mask_pools, skeleton_lees, k=0, sum_pixel=5
                    )
                    neg_patch = get_neg_patch_from_skeleton_multi_batch(
                        mask_dilations, emb, mask_pools, k=0, sum_pixel=5
                    )

                    tt = 1
                    while pos_patch is None and tt < 5:
                        pos_patch = get_pos_patch_from_skeleton_multi_batch(
                            emb,
                            mask_pools,
                            skeleton_lees,
                            k=0,
                            sum_pixel=max(1, 5 - tt),
                        )
                        tt += 1

                    tt = 1
                    while pos_patch is None and tt < 5:
                        neg_patch = get_neg_patch_from_skeleton_multi_batch(
                            mask_dilations,
                            emb,
                            mask_pools,
                            k=0,
                            sum_pixel=max(1, 5 - tt),
                        )
                        tt += 1

                    if pos_patch is not None and neg_patch is not None:
                        loss_con = get_contrastive_loss_v2(
                            pos_patch, neg_patch, cont_loss_func, k=0
                        )
                        loss += loss_con * 0.1
                        loss_con_list.append(loss_con.item())

                    out_dis = torch.tanh(out_dis)

                    if gt_diss.sum() != 0:
                        # compute L1 Loss
                        loss_dist = torch.norm(out_dis - gt_diss, 1) / torch.numel(
                            out_dis
                        )
                        loss_dist_list.append(loss_dist.item())
                        loss += loss_dist

                    if len(masks) > len(mask):
                        pred1 = out_put_sig[0: len(mask)]
                        pred2 = out_put_sig[len(mask):]
                        loss_cos = criterion_mse(pred1, pred2)

                        pred1 = out_dis[0: len(mask)]
                        pred2 = out_dis[len(mask):]
                        loss_cos += criterion_mse(pred1, pred2)
                        loss += loss_cos
                        loss_mse_list.append(loss_cos.item())

                    out_dis = bgrTh(out_dis)
                    out_dis = tarTh(out_dis)
                    loss_dis_dsc = criterion_dsc(out_dis, masks)
                    loss_dsc_dist_list.append(loss_dis_dsc.item())
                    loss += loss_dis_dsc

                    loss.backward()
                    optimizer.step()

                except RuntimeError or ValueError or OverflowError:
                    print(images.size())
                    traceback.print_exc()

                torch.cuda.empty_cache()

                t.set_postfix(
                    dsc=np.mean(loss_dsc_list),
                    bce=np.mean(loss_bce_list),
                    con=np.mean(loss_con_list),
                    mse=np.mean(loss_mse_list),
                    dist=np.mean(loss_dist_list),
                    cls=np.mean(loss_cls_list),
                    ddsc=np.mean(loss_dsc_dist_list),
                )

                # adjust learning rate
                adjust_learning_rate_D(
                    optimizer,
                    args.lr,
                    i * int(len(train_loader)) + batch_idx,
                    args.epochs * int(len(train_loader)),
                )
        if args.log:
            with open(
                    join("../checkpoints", args.time, "%s" % log_file_name), "a"
            ) as f:
                f.write(
                    "E %s, ave_bce_loss=%s, ave_dsc_loss=%s, ave_con_loss=%s, ave_mse_loss=%s, ave_cls_loss=%s, ave_dist_loss=%s, ave_dsc_dist_loss=%s \n"
                    % (
                        i,
                        np.mean(loss_bce_list),
                        np.mean(loss_dsc_list),
                        np.mean(loss_con_list),
                        np.mean(loss_mse_list),
                        np.mean(loss_cls_list),
                        np.mean(loss_dist_list),
                        np.mean(loss_dsc_dist_list),
                    )
                )

        if (i + 1) % 10 == 0 or i >= (args.epochs - 10):
            torch.save(
                model.state_dict(),
                join(
                    checkpoint_path,
                    "Tumor_3D_Fine_Fine_%s.pth" % args.data,
                ),
            )


if __name__ == "__main__":

    if args.time == "time":
        args.time = get_time()
    args.time = "Tumor_3D_Fine_Fine_On_%s_%s" % (args.data, args.time)
    generalize_train_pool_u_conresnet(args)
