
import argparse
import os
import sys
import time

import matplotlib
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
        "--ensemble", type=str, default="Mean", metavar="N", help="ensemble (No)"
    )
    parser.add_argument(
        "--log", action="store_false", default=True, help="write logs (default: false)"
    )
    parser.add_argument(
        "--save", action="store_false", default=True, help="write logs (default: false)"
    )
    parser.add_argument(
        "--check", action="store_true", default=False, help="check box (default: false)"
    )
    parser.add_argument("--data", type=str, default="nih", metavar="N", help="nih")
    parser.add_argument(
        "--path",
        type=str,
        default="Baseline_Coarse_Pancreas_Generalize_nih_2022_09_09_00_09",
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

from config.config import get_test_list, get_data_mask_path
from utils.util import (
    center_crop,
    post_processing,
    seed_torch,
    cal_recall,
    cal_precision,
    join,
    padding_z,
    save_arg,
    get_data_mask,
    pred_result_ensemble_mean_up,
    center_uncrop,
    unpadding_z,
    save_pred_coarse,
)

seed_torch(2022)

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj
H_NIH_3D = 320
W_NIH_3D = 368
MARGIN = 25


def evaluate_on_test_list(args):

    patient_dsc_full = []
    patient_recall = []
    patient_precision = []
    time_list = []

    for i in range(len(args.test_list)):
        images, mask = get_data_mask(args, args.test_list[i])
        mask[mask > 0] = 1
        time_start = time.time()
        image, mask_pad = padding_z(images, mask, mode="up")
        image, mask_pad_crop = (
            center_crop(image, H_NIH_3D, W_NIH_3D),
            center_crop(mask_pad, H_NIH_3D, W_NIH_3D),
        )
        image, mask_pad_crop_pool, mask_pad_crop = (
            image[::2, ::2, ::2],
            mask_pad_crop[::2, ::2, ::2],
            mask_pad_crop,
        )

        pred, pred_pool = pred_result_ensemble_mean_up(image, args)
        pred_upsample_post = post_processing(pred)

        pred_uncrop = center_uncrop(
            np.squeeze(pred_upsample_post), np.squeeze(mask_pad)
        )
        pred_3D = unpadding_z(pred_uncrop, mask)
        time_end = time.time()
        time_sum = time_end - time_start

        temp_patient_dsc_full = metric.dc(pred_3D, mask)
        temp_recall = cal_recall(pred_3D, mask)
        temp_precision = cal_precision(pred_3D, mask)

        if args.save:
            save_pred_coarse(args, args.test_list[i], pred_3D)

        print(
            "# %s in %s s: DSC %s, Recall %s, Precision %s"
            % (
                args.test_list[i],
                time_sum,
                temp_patient_dsc_full,
                temp_recall,
                temp_precision,
            )
        )

        if args.log:
            with open(join(args.path, args.log_file), "a") as f:
                f.write(
                    "# %s in %s s: DSC %s, Recall %s, Precision %s \n"
                    % (
                        args.test_list[i],
                        time_sum,
                        temp_patient_dsc_full,
                        temp_recall,
                        temp_precision,
                    )
                )

        patient_dsc_full.append(temp_patient_dsc_full)
        patient_recall.append(temp_recall)
        patient_precision.append(temp_precision)
        time_list.append(time_sum)

    print("Len time: ", len(time_list))
    if args.log:
        with open(join(args.path, args.log_file), "a") as f:
            f.write("Len time: %s \n" % len(time_list))

    return (
        np.mean(patient_dsc_full),
        np.mean(patient_recall),
        np.mean(patient_precision),
        np.mean(time_list),
    )


def evaluate(args):

    if args.log:
        save_arg(args, join(args.path, args.log_file))

    model_path = join(args.path, args.model_name)
    args.model.load_state_dict(torch.load(model_path))

    dsc_mean, recall_mean, precision_mean, time_mean = evaluate_on_test_list(args)

    print(
        "DSC in :",
        time_mean,
        " s,",
        dsc_mean,
        "Recall Post:",
        recall_mean,
        "Precision Post:",
        precision_mean,
    )
    print("####" * 20)

    if args.log:
        with open(join(args.path, args.log_file), "a") as f:
            f.write(
                "DSC in %s s, %s, Recall %s, Precision %s \n"
                % (time_mean, dsc_mean, recall_mean, precision_mean)
            )
            f.write("####" * 20 + "\n")


if __name__ == "__main__":

    args.test_list = get_test_list(args)
    args.data_path, args.mask_path = get_data_mask_path(args)
    args.model = U_CorResNet_Fix_Contrastive_Proj().cuda().eval()
    args.log_file = "all_thresh_%s_ensemble_%s.txt" % (args.thresh, args.ensemble)
    args.path = "../checkpoints/" + args.path
    evaluate(args)
