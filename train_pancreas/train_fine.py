import argparse
import os
import sys
import traceback

from pytorch_metric_learning import losses

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from train_coarse import Generalize_Pancreas_Dataset_Skeleton

from utils.get_patches import (
    get_pos_patch_from_skeleton_none,
    get_neg_patch_from_skeleton_none,
)
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.util import (
    adjust_learning_rate_D,
    seed_torch,
    get_time,
    save_arg,
    join,
    exists,
)


def get_args():
    parser = argparse.ArgumentParser(description="3D Res UNet for segmentation")
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
        default=150,
        metavar="N",
        help="number of epochs to train_tumor (default: 150)",
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
        "--data", type=str, default="rmyy", metavar="str", help="dataset for training"
    )
    parser.add_argument(
        "--workers", type=int, default=6, metavar="N", help="num_worker (default: 0)"
    )
    return parser.parse_args()


args = get_args()
seed_torch(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj
from models.loss import BCELoss, DSC_loss, get_contrastive_loss_v2


def generalize_train_pool_u_conresnet(args):
    log_file_name = "train_logs.txt"

    if not exists(join("../checkpoints", args.time)):
        os.mkdir(join("../checkpoints", args.time))
    checkpoint_path = join("../checkpoints", args.time)
    save_arg(args, join("../checkpoints", args.time, log_file_name))

    model = U_CorResNet_Fix_Contrastive_Proj().cuda().train()
    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    dset_train = Generalize_Pancreas_Dataset_Skeleton(
        data=args.data, augment=True, resize=True, crop=True, pool=False
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
    model.train()

    # train_tumor for some epochs
    for i in range(args.epochs):

        loss_dsc_list = []
        loss_bce_list = []
        loss_con_list = []

        with tqdm(train_loader, ncols=120) as t:
            for (
                    batch_idx,
                    (image, mask, mask_pool, mask_dilation, skeleton_lee),
            ) in enumerate(t):

                t.set_description("%s" % i)
                image, mask, mask_pool = (
                    image.cuda().float(),
                    mask.cuda().float(),
                    mask_pool.cuda().float(),
                )
                mask_dilation, skeleton_lee = (
                    mask_dilation.cuda().float(),
                    skeleton_lee.cuda().float(),
                )

                assert mask_dilation.shape == skeleton_lee.shape
                assert mask_pool.shape == skeleton_lee.shape

                try:
                    optimizer.zero_grad()
                    output, emb = model(image)
                    out_put_sig = torch.sigmoid(output)
                    loss_dsc = criterion_dsc(out_put_sig, mask)
                    loss_bce = criterion_bce(output, mask)
                    loss = loss_dsc + loss_bce
                    loss_dsc_list.append(loss_dsc.item())
                    loss_bce_list.append(loss_bce.item())

                    pos_patch = get_pos_patch_from_skeleton_none(
                        emb, mask_pool, skeleton_lee, k=0, sum_pixel=10
                    )
                    neg_patch = get_neg_patch_from_skeleton_none(
                        mask_dilation, emb, mask_pool, k=0, sum_pixel=5
                    )

                    if pos_patch is not None and neg_patch is not None:
                        loss_con = get_contrastive_loss_v2(
                            pos_patch, neg_patch, cont_loss_func, k=0
                        )
                        loss += loss_con * 0.1
                        loss_con_list.append(loss_con.item())

                    loss.backward()
                    optimizer.step()

                except RuntimeError:
                    print(image.size())
                    traceback.print_exc()

                torch.cuda.empty_cache()

                t.set_postfix(
                    dsc=np.mean(loss_dsc_list),
                    bce=np.mean(loss_bce_list),
                    con=np.mean(loss_con_list),
                )

                # adjust learning rate
                adjust_learning_rate_D(
                    optimizer,
                    args.lr,
                    i * int(len(train_loader)) + batch_idx,
                    args.epochs * int(len(train_loader)),
                )

        with open(join("../checkpoints", args.time, "%s" % log_file_name), "a") as f:
            f.write(
                "E %s, ave_bce_loss=%s, ave_dsc_loss=%s, ave_con_loss=%s \n"
                % (
                    i,
                    np.mean(loss_bce_list),
                    np.mean(loss_dsc_list),
                    np.mean(loss_con_list),
                )
            )

        if (i + 1) % 10 == 0 or i >= (args.epochs - 10):
            torch.save(
                model.state_dict(),
                join(
                    checkpoint_path,
                    "Generalize_Pancreas_Fine_%s.pth" % args.data,
                ),
            )


if __name__ == "__main__":

    datas = list(map(str, args.data.split(",")))
    if args.time == "time":
        args.time = get_time()
    for data in datas:
        args.data = data
        args.time = "Fine_Pancreas_Generalize_%s_%s" % (args.data, args.time)
        generalize_train_pool_u_conresnet(args)
