import argparse
import itertools
import os
import sys
import time

import matplotlib
import nibabel
import nrrd
import numpy as np
import torch
from medpy import metric

matplotlib.use("AGG")

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def get_args():
    parser = argparse.ArgumentParser(
        description="Test for pancreatic cancer segmentation"
    )
    parser.add_argument(
        "--gpu", type=str, default="0", metavar="N", help="visible devices (default: 0)"
    )
    parser.add_argument(
        "--coarse_thresh",
        type=float,
        default=0.5,
        metavar="N",
        help="coarse thresh (0.5)",
    )
    parser.add_argument(
        "--coarse_ensemble", type=str, default="Mean", metavar="N", help="ensemble (No)"
    )
    parser.add_argument(
        "--ensemble", type=str, default="Mean", metavar="N", help="ensemble (No)"
    )
    parser.add_argument(
        "--log", type=bool, default=True, help="write logs (default: true)"
    )
    parser.add_argument(
        "--save", type=bool, default=True, help="save preds (default: false)"
    )
    parser.add_argument(
        "--check", action="store_true", default=False, help="check box (default: false)"
    )
    parser.add_argument("--data", type=str, default="msd", metavar="N", help="nih")
    parser.add_argument(
        "--coarse_path",
        type=str,
        default="Baseline_Coarse_Pancreas_Generalize_nih_2022_09_09_00_09",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--thresh", type=float, default=0.5, help="thresh for binaries."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="Tumor_3D_Fine_CL_DT_CLS_clsseg_On_rmyy,renji_2022_11_30_11_38",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Tumor_3D_Fine_CL_DT_CLS_clsseg_rmyy,renji.pth",
        metavar="N",
        help="model_name",
    )
    return parser.parse_args()


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from config.config import (
    get_tumor_data_mask_path,
    LOW_RANGE,
    HIGH_RANGE,
    rmyy_panc_tumor_label_path,
    renji_panc_tumor_label_path,
    msd_mix_list,
)
from utils.util import (
    post_processing,
    seed_torch,
    cal_recall,
    cal_precision,
    join,
    save_arg,
    crop_by_pancreas,
    cal_cover,
    IOStream,
    pred_result_ensemble_mean,
)

seed_torch(2022)

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj_DT


def evaluate_on_test_list(args):
    dice_t = []
    recall_t = []
    precision_t = []
    cover_t = []
    cover_margin_t = []
    time_list = []

    for i in range(len(args.test_list)):
        images, mask, _ = get_data_mask_coarse(
            args, args.test_list[i], use_coarse=False
        )

        image, mask_crop_temp, bbox = crop_by_pancreas(images, mask, mask)

        mask[mask != 2] = 0
        mask[mask == 2] = 1

        time_start = time.time()
        pred = pred_result_ensemble_mean(image, args)
        pred = pred >= args.thresh
        temp_pred = post_processing(pred)
        pred_3D = np.zeros_like(mask)
        pred_3D[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]] = temp_pred
        time_end = time.time()
        time_sum = time_end - time_start

        dice = metric.dc(pred_3D, mask)
        recall = cal_recall(pred_3D, mask)
        precision = cal_precision(pred_3D, mask)
        if pred_3D.sum() > 10:
            f1 = cal_cover(pred_3D, mask, margin_z=0, margin_xy=0)
            f2 = cal_cover(pred_3D, mask, margin_z=5, margin_xy=20)
        else:
            f1, f2 = 0.0, 0.0

        dice_t.append(dice)
        recall_t.append(recall)
        precision_t.append(precision)
        cover_t.append(f1)
        cover_margin_t.append(f2)
        time_list.append(time_sum)

        args.io.pwrite(
            "Processing {} Tumor in {} s: Dice = {:.4}, Recall = {:.4}, Precision = {:.4}, Cover = {:.4}, Cover Margin = "
            "{:.4}".format(args.test_list[i], time_sum, dice, recall, precision, f1, f2)
        )

        if args.save:
            if not os.path.exists(
                    join(
                        "../checkpoints/",
                        args.path,
                        "pred_{}_{}_{}".format(args.data, args.ensemble, args.thresh),
                    )
            ):
                os.mkdir(
                    join(
                        "../checkpoints/",
                        args.path,
                        "pred_{}_{}_{}".format(args.data, args.ensemble, args.thresh),
                    )
                )

            if args.data == "msd":
                pred_3D = np.flip(pred_3D, axis=0)
                pred_3D = np.flip(pred_3D, axis=1)
                temp = pred_3D.transpose((1, 2, 0))
            else:
                temp = pred_3D.transpose((2, 1, 0))

            nrrd.write(
                join(
                    "../checkpoints/",
                    args.path,
                    "pred_{}_{}_{}".format(args.data, args.ensemble, args.thresh),
                    "{}.nrrd".format(args.test_list[i]),
                ),
                temp,
            )

    dice_avg_t = np.mean(dice_t)
    recall_avg_t = np.mean(recall_t)
    precision_avg_t = np.mean(precision_t)
    cover_avg_t = np.mean(cover_t)
    cover_margin_avg_t = np.mean(cover_margin_t)
    time_mean = np.mean(time_list)

    args.io.pwrite(
        "Average: Dice = {:.4}, Recall = {:.4}, Precision = {:.4}, Cover = {:.4}, Cover Margin = "
        "{:.4}, Time = {}".format(
            dice_avg_t,
            recall_avg_t,
            precision_avg_t,
            cover_avg_t,
            cover_margin_avg_t,
            time_mean,
        )
    )


def get_data_mask_coarse(args, index, use_coarse=True):
    HAS_FILE = os.path.exists(
        join(
            args.coarse_path,
            "true_pred_{}_No_{}".format(args.data, args.coarse_thresh),
            "{}.nrrd".format(index),
        )
    )

    if use_coarse and HAS_FILE:
        coarse = nrrd.read(
            join(
                args.coarse_path,
                "true_pred_{}_No_{}".format(args.data, args.coarse_thresh),
                "{}.nrrd".format(index),
            )
        )[0]
        if coarse.sum() < 5:
            coarse = None
            HAS_FILE = False
    elif not use_coarse:
        coarse = None

    if args.data.lower() in ["nih", "msd"]:
        image = (
            np.load(join(args.data_path, "{}.npy".format(index)))
            .transpose((2, 0, 1))
            .astype(np.float)
        )
        mask = np.load(join(args.mask_path, "{}.npy".format(index))).transpose(
            (2, 0, 1)
        )

        image = np.flip(image, axis=1)
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=1)
        mask = np.flip(mask, axis=0)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        if use_coarse:
            if HAS_FILE:
                coarse = coarse.transpose((2, 0, 1))
                coarse = np.flip(coarse, axis=1)
                coarse = np.flip(coarse, axis=0)
                coarse = np.ascontiguousarray(coarse)
            else:
                coarse = mask.copy()

    elif args.data.lower() in ["renji"]:
        image = nibabel.load(
            join(args.data_path, "{}.nii.gz".format(index))
        ).get_fdata()

        if len(image.shape) == 5:
            image = image[:, :, :, 0, 0]

        image = image.transpose((2, 1, 0)).astype(np.float)
        mask = (
            nrrd.read(join(args.mask_path, "{}.nrrd".format(index)))[0]
            .transpose((2, 1, 0))
            .astype(np.float)
        )
        if use_coarse:
            if HAS_FILE:
                coarse = coarse.transpose((2, 1, 0))
            else:
                coarse = mask.copy()

    elif args.data.lower() in ["rmyy"]:
        image = (
            nrrd.read(join(args.data_path, "{:0>3}.nrrd".format(index)))[0]
            .transpose(2, 1, 0)
            .astype(np.float)
        )
        mask = (
            nrrd.read(join(args.mask_path, "{:0>3}.nrrd".format(index)))[0]
            .transpose(2, 1, 0)
            .astype(np.float)
        )
        if use_coarse:
            if HAS_FILE:
                coarse = coarse.transpose((2, 1, 0))
            else:
                coarse = mask.copy()

    else:
        image, mask = None, None

    if np.max(image) > 1:
        np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
        image -= LOW_RANGE
        image /= HIGH_RANGE - LOW_RANGE

    return image, mask, coarse


def evaluate(args):
    if args.log:
        save_arg(args, join(args.path, args.log_file))
    args.io = IOStream(join(args.path, args.log_file), log=args.log)
    model_path = join(args.path, args.model_name)
    args.model.load_state_dict(torch.load(model_path))
    evaluate_on_test_list(args)


def get_tumor_test_list(args):
    if args.data.lower() == "rmyy":
        test_list = sorted(os.listdir(rmyy_panc_tumor_label_path))
        test_list = [int(i.replace(".nrrd", "")) for i in test_list]
    elif args.data.lower() == "msd":
        a = []
        a.append(msd_mix_list[-1])
        a.append(msd_mix_list[-2])
        a.append(msd_mix_list[-3])
        a.append(msd_mix_list[-4])
        test_list = list(itertools.chain(*a))
    elif args.data.lower() == "renji":
        test_list = sorted(os.listdir(renji_panc_tumor_label_path))
        test_list = [int(i.replace(".nrrd", "")) for i in test_list]
    else:
        return None
    return test_list


if __name__ == "__main__":
    args.test_list = get_tumor_test_list(args)
    args.data_path, args.mask_path = get_tumor_data_mask_path(args)
    args.model = U_CorResNet_Fix_Contrastive_Proj_DT().cuda().eval()
    args.log_file = "cal_time_ensemble_%s_on_%s_%s_based_pancreas_All.txt" % (
        args.ensemble,
        args.data,
        args.thresh,
    )
    args.path = "../checkpoints/" + args.path
    args.coarse_path = "../checkpoints/" + args.coarse_path
    evaluate(args)
