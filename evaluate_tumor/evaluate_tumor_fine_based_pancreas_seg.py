
import argparse
import itertools
import os
import sys
import time
import traceback

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
        "--thresh", type=float, default=0.5, help="thresh for binaries."
    )
    parser.add_argument(
        "--coarse_thresh",
        type=float,
        default=0.0,
        metavar="N",
        help="coarse thresh (0.5)",
    )
    parser.add_argument(
        "--coarse_ensemble",
        type=str,
        default="Mean",
        metavar="N",
        help="ensemble (Mean)",
    )
    parser.add_argument(
        "--ensemble", type=str, default="Mean", metavar="N", help="ensemble (Mean)"
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
        default="Tumor_3D_In_Pancreas_On_renji,rmyy_2022_11_15_17_21",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="Tumor_3D_In_Pancreas_On_renji,rmyy_2022_11_15_17_21",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Tumor_3D_In_Pancreas_On_renji,rmyy.pth",
        metavar="N",
        help="model_name",
    )
    parser.add_argument(
        "--mode", type=str, default="replace", metavar="N", help="merge function"
    )
    return parser.parse_args()


args = get_args()
if args.coarse_thresh == 0:
    args.coarse_thresh = args.thresh
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from config.config import (
    get_tumor_data_mask_path,
    HIGH_RANGE,
    LOW_RANGE,
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
    pred_result_ensemble_mean_res,
)

seed_torch(2022)

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj_DT_CCLS_RC


def get_data_mask_coarse(args, index):

    coarse = nrrd.read(
        join(
            args.coarse_path,
            "pred_{}_{}_{}_based_pancreas_segmentation".format(
                args.data, args.coarse_ensemble, args.coarse_thresh
            ),
            "{}.nrrd".format(index),
        )
    )[0]
    if coarse.sum() < 10:
        coarse = None

    if coarse is None:
        if args.data.lower() in ["msd"]:
            coarse = nrrd.read(
                join(
                    "../checkpoints/CL_Fine_Pancreas_Generalize_New_msd_2023_04_10_22_36",
                    "thresh_0.5_ensemble_Mean_based_0.5_Mean",
                    "{}.nrrd".format(index),
                )
            )[0]
        elif args.data.lower() in ["rmyy"]:
            coarse = nrrd.read(
                join(
                    "../checkpoints/CL_Fine_Pancreas_Generalize_New_rmyy_2023_04_09_19_50",
                    "thresh_0.5_ensemble_Mean_based_0.5_Mean",
                    "{}.nrrd".format(index),
                )
            )[0]
        elif args.data.lower() in ["renji"]:
            coarse = nrrd.read(
                join(
                    "../checkpoints/CL_Fine_Pancreas_Generalize_New_renji_2023_04_11_19_33",
                    "thresh_0.5_ensemble_Mean_based_0.5_Mean",
                    "{}.nrrd".format(index),
                )
            )[0]

    if args.data.lower() in ["nih", "isicdm", "msd"]:
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
        coarse = coarse.transpose((2, 0, 1))
        coarse = np.flip(coarse, axis=1)
        coarse = np.flip(coarse, axis=0)
        coarse = np.ascontiguousarray(coarse)

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
        coarse = coarse.transpose((2, 1, 0))

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
        coarse = coarse.transpose((2, 1, 0))

    if np.max(image) > 1:
        np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
        image -= LOW_RANGE
        image /= HIGH_RANGE - LOW_RANGE

    return image, mask, coarse


def evaluate_on_test_list(args):

    dsc_list = np.zeros((len(args.test_list), args.round + 1))
    DSC_90 = np.zeros((len(args.test_list))) + 10000
    DSC_91 = np.zeros((len(args.test_list))) + 10000
    DSC_92 = np.zeros((len(args.test_list))) + 10000
    DSC_93 = np.zeros((len(args.test_list))) + 10000
    DSC_94 = np.zeros((len(args.test_list))) + 10000
    DSC_95 = np.zeros((len(args.test_list))) + 10000

    recall_list = np.zeros((len(args.test_list), args.round + 1))
    recall_90 = np.zeros((len(args.test_list))) + 10000
    recall_91 = np.zeros((len(args.test_list))) + 10000
    recall_92 = np.zeros((len(args.test_list))) + 10000
    recall_93 = np.zeros((len(args.test_list))) + 10000
    recall_94 = np.zeros((len(args.test_list))) + 10000
    recall_95 = np.zeros((len(args.test_list))) + 10000

    precision_list = np.zeros((len(args.test_list), args.round + 1))
    precision_90 = np.zeros((len(args.test_list))) + 10000
    precision_91 = np.zeros((len(args.test_list))) + 10000
    precision_92 = np.zeros((len(args.test_list))) + 10000
    precision_93 = np.zeros((len(args.test_list))) + 10000
    precision_94 = np.zeros((len(args.test_list))) + 10000
    precision_95 = np.zeros((len(args.test_list))) + 10000

    cover_list = np.zeros((len(args.test_list), args.round + 1))
    cover_90 = np.zeros((len(args.test_list))) + 10000
    cover_91 = np.zeros((len(args.test_list))) + 10000
    cover_92 = np.zeros((len(args.test_list))) + 10000
    cover_93 = np.zeros((len(args.test_list))) + 10000
    cover_94 = np.zeros((len(args.test_list))) + 10000
    cover_95 = np.zeros((len(args.test_list))) + 10000

    cover_margin_list = np.zeros((len(args.test_list), args.round + 1))
    cover_margin_90 = np.zeros((len(args.test_list))) + 10000
    cover_margin_91 = np.zeros((len(args.test_list))) + 10000
    cover_margin_92 = np.zeros((len(args.test_list))) + 10000
    cover_margin_93 = np.zeros((len(args.test_list))) + 10000
    cover_margin_94 = np.zeros((len(args.test_list))) + 10000
    cover_margin_95 = np.zeros((len(args.test_list))) + 10000

    time_list = []

    for i in range(len(args.test_list)):
        try:

            images, mask, coarse = get_data_mask_coarse(args, args.test_list[i])
            mask[mask != 2] = 0
            mask[mask == 2] = 1
            coarse[coarse > 0] = 1
            prob_f_threshed_temp = None
            prob_temp = None

            dice = metric.dc(coarse, mask)
            recall = cal_recall(coarse, mask)
            precision = cal_precision(coarse, mask)

            if coarse.sum() > 5:
                f1 = cal_cover(coarse, mask, margin_z=0, margin_xy=0)
                f2 = cal_cover(coarse, mask, margin_z=5, margin_xy=20)
            else:
                f1, f2 = 0.0, 0.0

            dsc_list[i, 0] = dice
            recall_list[i, 0] = recall
            precision_list[i, 0] = precision
            cover_list[i, 0] = f1
            cover_margin_list[i, 0] = f2

            args.io.pwrite(
                "Processing {} Coarse: Dice = {:.4}, Recall = {:.4}, Precision = {:.4}, Cover = {:.4}, Cover Margin = "
                "{:.4}".format(args.test_list[i], dice, recall, precision, f1, f2)
            )
            start_time = time.time()
            time_sum = -1
            for r in range(args.round):

                if r == 0:
                    coarse_temp = coarse
                elif np.sum(pred_3D) > 80:
                    coarse_temp = pred_3D

                image, mask_crop_temp, bbox = crop_by_pancreas(
                    images, mask, coarse_temp
                )

                if "Res_Weight_std_False" in args.model_name:
                    pred_prob = pred_result_ensemble_mean_res(image, args)
                else:
                    pred_prob = pred_result_ensemble_mean(image, args)

                pred = pred_prob >= args.thresh
                temp_pred = post_processing(pred)
                pred_3D = np.zeros_like(mask)
                prob_3D = np.zeros_like(mask)
                pred_3D[
                    bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
                ] = temp_pred
                prob_3D[
                    bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
                ] = pred_prob
                dice = metric.dc(pred_3D, mask)
                recall = cal_recall(pred_3D, mask)
                precision = cal_precision(pred_3D, mask)

                if pred_3D.sum() > 5:
                    f1 = cal_cover(pred_3D, mask, margin_z=0, margin_xy=0)
                    f2 = cal_cover(pred_3D, mask, margin_z=5, margin_xy=20)
                else:
                    f1, f2 = 0.0, 0.0

                dsc_list[i, r + 1] = dice
                recall_list[i, r + 1] = recall
                precision_list[i, r + 1] = precision
                cover_list[i, r + 1] = f1
                cover_margin_list[i, r + 1] = f2

                if prob_f_threshed_temp is not None and prob_temp is not None:
                    dsc_recurrent = metric.dc(pred_3D, prob_f_threshed_temp)
                    if DSC_95[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.95
                    ):
                        DSC_95[i] = dice
                        recall_95[i] = recall
                        precision_95[i] = precision
                        cover_95[i] = f1
                        cover_margin_95 = f2
                    if DSC_94[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.94
                    ):
                        DSC_94[i] = dice
                        recall_94[i] = recall
                        precision_94[i] = precision
                        cover_94[i] = f1
                        cover_margin_94 = f2
                    if DSC_93[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.93
                    ):
                        DSC_93[i] = dice
                        recall_93[i] = recall
                        precision_93[i] = precision
                        cover_93[i] = f1
                        cover_margin_93 = f2
                    if DSC_92[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.92
                    ):
                        DSC_92[i] = dice
                        recall_92[i] = recall
                        precision_92[i] = precision
                        cover_92[i] = f1
                        cover_margin_92 = f2

                        end_time = time.time()
                        time_sum = end_time - start_time
                        time_list.append(time_sum)

                        if args.save:
                            if not os.path.exists(
                                join(
                                    args.path,
                                    "pred_{}_{}_{}_{}_tumor_based_pancreas_segmentation".format(
                                        args.coarse_thresh,
                                        args.thresh,
                                        args.data,
                                        args.ensemble,
                                    ),
                                )
                            ):
                                os.mkdir(
                                    join(
                                        args.path,
                                        "pred_{}_{}_{}_{}_tumor_based_pancreas_segmentation".format(
                                            args.coarse_thresh,
                                            args.thresh,
                                            args.data,
                                            args.ensemble,
                                        ),
                                    )
                                )

                            if args.data == "msd":
                                temp = pred_3D.copy()
                                temp = np.flip(temp, axis=0)
                                temp = np.flip(temp, axis=1)
                                temp = temp.transpose((1, 2, 0))
                            else:
                                temp = pred_3D.transpose((2, 1, 0))

                            nrrd.write(
                                join(
                                    args.path,
                                    "pred_{}_{}_{}_{}_tumor_based_pancreas_segmentation".format(
                                        args.coarse_thresh,
                                        args.thresh,
                                        args.data,
                                        args.ensemble,
                                    ),
                                    "{}.nrrd".format(args.test_list[i]),
                                ),
                                temp,
                            )

                    if DSC_91[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.91
                    ):
                        DSC_91[i] = dice
                        recall_91[i] = recall
                        precision_91[i] = precision
                        cover_91[i] = f1
                        cover_margin_91 = f2
                    if DSC_90[i] == 10000 and (
                        r == args.round - 1 or dsc_recurrent >= 0.90
                    ):
                        DSC_90[i] = dice
                        recall_90[i] = recall
                        precision_90[i] = precision
                        cover_90[i] = f1
                        cover_margin_90 = f2

                if prob_f_threshed_temp is None:
                    prob_f_threshed_temp = pred_3D
                    prob_temp = prob_3D
                elif args.mode == "replace":
                    prob_f_threshed_temp = pred_3D
                elif args.mode == "or":
                    prob_f_threshed_temp = np.logical_or(pred_3D, prob_f_threshed_temp)
                elif args.mode == "mean":
                    prob_temp = (prob_3D + prob_temp) / 2
                    prob_f_threshed_temp = prob_temp >= 0.5
                    prob_f_threshed_temp = post_processing(prob_f_threshed_temp)

                args.io.pwrite(
                    "Processing {} Round {} in {}s: Dice = {:.4}, Recall = {:.4}, Precision = {:.4}, Cover = {:.4}, Cover Margin = "
                    "{:.4}".format(
                        args.test_list[i], r, time_sum, dice, recall, precision, f1, f2
                    )
                )

        except RuntimeError or ValueError:
            print(images.shape)
            traceback.print_exc()

    for r in range(args.round + 1):
        args.io.pwrite(
            "Round {}: DSC = {}  Recall = {}  Precision = {}  Cover = {}  Cover Margin = {}.".format(
                r,
                np.mean(dsc_list[:, r]),
                np.mean(recall_list[:, r]),
                np.mean(precision_list[:, r]),
                np.mean(cover_list[:, r]),
                np.mean(cover_margin_list[:, r]),
            )
        )

    args.io.pwrite(
        "0.95:  DSC = "
        + str(np.mean(DSC_95))
        + " Recall = "
        + str(np.mean(recall_95))
        + " Precision = "
        + str(np.mean(precision_95))
        + " Cover = "
        + str(np.mean(cover_95))
        + " Cover Margin = "
        + str(np.mean(cover_margin_95))
        + " ."
    )
    args.io.pwrite(
        "0.94:  DSC = "
        + str(np.mean(DSC_94))
        + " Recall = "
        + str(np.mean(recall_94))
        + " Precision = "
        + str(np.mean(precision_94))
        + " Cover = "
        + str(np.mean(cover_94))
        + " Cover Margin = "
        + str(np.mean(cover_margin_94))
        + " ."
    )
    args.io.pwrite(
        "0.93:  DSC = "
        + str(np.mean(DSC_93))
        + " Recall = "
        + str(np.mean(recall_93))
        + " Precision = "
        + str(np.mean(precision_93))
        + " Cover = "
        + str(np.mean(cover_93))
        + " Cover Margin = "
        + str(np.mean(cover_margin_93))
        + " ."
    )
    args.io.pwrite(
        "0.92:  DSC = "
        + str(np.mean(DSC_92))
        + " Recall = "
        + str(np.mean(recall_92))
        + " Precision = "
        + str(np.mean(precision_92))
        + " Cover = "
        + str(np.mean(cover_92))
        + " Cover Margin = "
        + str(np.mean(cover_margin_92))
        + " ."
    )
    args.io.pwrite(
        "0.91:  DSC = "
        + str(np.mean(DSC_91))
        + " Recall = "
        + str(np.mean(recall_91))
        + " Precision = "
        + str(np.mean(precision_91))
        + " Cover = "
        + str(np.mean(cover_91))
        + " Cover Margin = "
        + str(np.mean(cover_margin_91))
        + " ."
    )
    args.io.pwrite(
        "0.90:  DSC = "
        + str(np.mean(DSC_90))
        + " Recall = "
        + str(np.mean(recall_90))
        + " Precision = "
        + str(np.mean(precision_90))
        + " Cover = "
        + str(np.mean(cover_90))
        + " Cover Margin = "
        + str(np.mean(cover_margin_90))
        + " ."
    )
    args.io.pwrite(
        "Mean Time: {} of {} cases.".format(np.mean(time_list), len(time_list))
    )
    args.io.pwrite("####" * 20)


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
    args.model = U_CorResNet_Fix_Contrastive_Proj_DT_CCLS_RC().cuda().eval()

    args.log_file = (
        "save_true_%s_%s_on_%s_based_%s_%s_based_pancreas_segmentation.txt"
        % (args.ensemble, args.thresh, args.data, args.coarse_thresh, args.mode)
    )
    args.round = 4
    args.path = "../checkpoints/" + args.path
    args.coarse_path = "../checkpoints/" + args.coarse_path
    evaluate(args)
