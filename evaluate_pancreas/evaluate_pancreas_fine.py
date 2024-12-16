import argparse
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
        "--thresh", type=float, default=0.5, metavar="N", help="thresh (0.5)"
    )
    parser.add_argument(
        "--coarse_thresh",
        type=float,
        default=0.5,
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
        "--log", action="store_false", default=True, help="write logs (default: false)"
    )
    parser.add_argument(
        "--save", action="store_false", default=True, help="write logs (default: false)"
    )
    parser.add_argument("--data", type=str, default="msd", metavar="N", help="msd")
    parser.add_argument(
        "--coarse_path",
        type=str,
        default="Baseline_Coarse_Pancreas_Generalize_nih_2022_09_09_00_09",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="Baseline_Fine_Pancreas_Generalize_nih_2022_09_08_14_19",
        metavar="N",
        help="checkpoint path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Generalize_Pancreas_Baseline_Coarse_nih.pth",
        metavar="N",
        help="model_name",
    )
    return parser.parse_args()


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from config.config import get_test_list, get_data_mask_path, LOW_RANGE, HIGH_RANGE
from utils.util import (
    post_processing,
    seed_torch,
    cal_recall,
    cal_precision,
    join,
    save_arg,
    crop_by_pancreas,
    pred_result_ensemble_mean,
    save_pred_fine,
    cal_cover,
)

seed_torch(2022)

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj


def get_data_mask_coarse(args, index, use_coarse=True):
    HAS_FILE = os.path.exists(
        join(args.coarse_path, "pred_thresh_0.5_ensemble_Mean", "{}.nrrd".format(index))
    )

    if use_coarse and HAS_FILE:
        coarse = nrrd.read(
            join(
                args.coarse_path,
                "pred_thresh_0.5_ensemble_Mean",
                "{}.nrrd".format(index),
            )
        )[0]
        if coarse.sum() < 5:
            coarse = None
            HAS_FILE = False
    elif not use_coarse:
        print("No Coarse!!!!!")
        coarse = None

    elif use_coarse and not HAS_FILE:
        sys.exit("No Coarse Segmentations!")

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
            .transpose(2, 1, 0)
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


def evaluate_on_test_list(args):
    patient_dsc_full = []
    patient_recall = []
    patient_precision = []
    patient_box_recover_nomargin = []
    patient_box_recover_margin = []
    time_list = []

    for i in range(len(args.test_list)):
        images, mask, coarse = get_data_mask_coarse(args, args.test_list[i])
        mask[mask > 0] = 1
        coarse[coarse > 0] = 1
        pred_3D = np.zeros_like(mask)

        image, _, bbox = crop_by_pancreas(images, mask, coarse)
        time_start = time.time()

        pred = pred_result_ensemble_mean(image, args)
        pred = pred >= args.thresh

        pred_post = post_processing(pred)

        pred_3D[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]] = pred_post
        time_end = time.time()
        time_sum = time_end - time_start

        temp_patient_dsc_full = metric.dc(pred_3D, mask)
        temp_recall = cal_recall(pred_3D, mask)
        temp_precision = cal_precision(pred_3D, mask)
        temp_recover_nomargin = cal_cover(pred_3D, mask, margin_z=0, margin_xy=0)
        temp_recover_margin = cal_cover(pred_3D, mask, margin_z=5, margin_xy=20)

        if args.save:
            save_pred_fine(args, args.test_list[i], pred_3D)

        print(
            "# %s in %s s: DSC %s, Recall %s, Precision %s, Recover %s, Recover Margin %s"
            % (
                args.test_list[i],
                time_sum,
                temp_patient_dsc_full,
                temp_recall,
                temp_precision,
                temp_recover_nomargin,
                temp_recover_margin,
            )
        )

        if args.log:
            with open(join(args.path, args.log_file), "a") as f:
                f.write(
                    "# %s in %s s: DSC %s, Recall %s, Precision %s, Recover %s, Recover Margin %s \n"
                    % (
                        args.test_list[i],
                        time_sum,
                        temp_patient_dsc_full,
                        temp_recall,
                        temp_precision,
                        temp_recover_nomargin,
                        temp_recover_margin,
                    )
                )

        patient_dsc_full.append(temp_patient_dsc_full)
        patient_recall.append(temp_recall)
        patient_precision.append(temp_precision)
        patient_box_recover_nomargin.append(temp_recover_nomargin)
        patient_box_recover_margin.append(temp_recover_margin)
        time_list.append(time_sum)

    print("Len time: ", len(time_list))
    if args.log:
        with open(join(args.path, args.log_file), "a") as f:
            f.write("Len time: %s \n" % len(time_list))

    return (
        np.mean(patient_dsc_full),
        np.mean(patient_recall),
        np.mean(patient_precision),
        np.mean(patient_box_recover_nomargin),
        np.mean(patient_box_recover_margin),
        np.mean(time_list),
    )


def evaluate(args):
    if args.log:
        save_arg(args, join(args.path, args.log_file))

    model_path = join(args.path, args.model_name)
    args.model.load_state_dict(torch.load(model_path))

    dsc_mean, recall_mean, precision_mean, recover_nomarin_mean, recover_margin_mean, time_mean = evaluate_on_test_list(
        args
    )

    print(
        "DSC in :",
        time_mean,
        " s,",
        dsc_mean,
        "Recall Post:",
        recall_mean,
        "Precision Post:",
        precision_mean,
        "Recover nomargin Post:",
        recover_nomarin_mean,
        "Recover margin Post:",
        recover_margin_mean,
    )
    print("####" * 20)

    if args.log:
        with open(join(args.path, args.log_file), "a") as f:
            f.write(
                "DSC in %s s, %s, Recall %s, Precision %s, Recover %s, Recover Margin %s \n"
                % (
                    time_mean,
                    dsc_mean,
                    recall_mean,
                    precision_mean,
                    recover_nomarin_mean,
                    recover_margin_mean,
                )
            )
            f.write("####" * 20 + "\n")


if __name__ == "__main__":

    args.test_list = get_test_list(args)
    args.data_path, args.mask_path = get_data_mask_path(args)
    args.model = U_CorResNet_Fix_Contrastive_Proj().cuda().eval()
    args.log_file = "all_thresh_%s_ensemble_%s_based_%s_%s.txt" % (
        args.thresh,
        args.ensemble,
        args.coarse_thresh,
        args.coarse_ensemble,
    )

    args.path = "../checkpoints/" + args.path
    args.coarse_path = "../checkpoints/" + args.coarse_path
    if None not in [args.test_list, args.data_path, args.mask_path]:
        evaluate(args)
