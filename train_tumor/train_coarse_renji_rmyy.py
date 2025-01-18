
import argparse
import itertools
import math
import os
import random
import sys
import time
import traceback

import nibabel
import nrrd
from pytorch_metric_learning import losses
from skimage import morphology
from skimage.morphology import skeletonize_3d
from skimage.transform import resize


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config.config import (
    msd_resample_data_path,
    rmyy_data_path,
    msd_resample_multi_label_path,
    rmyy_panc_tumor_label_path,
    renji_panc_tumor_label_path,
    renji_data_path,
    LOW_RANGE,
    HIGH_RANGE,
    msd_mix_list,
)
from models.RandConv import RandConvModule
from utils.get_patches import (
    get_neg_patch_from_skeleton_multi_batch,
    get_pos_patch_from_skeleton_multi_batch,
)
from augmentation.volumentations import (
    Compose,
    RandomScale,
    Resize,
    ElasticTransform,
    RandomRotate,
    RandomFlip,
    RandomGamma,
    RandomGaussianNoise,
    Normalize,
)
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.device import func_device
from utils.util import (
    adjust_learning_rate_D,
    seed_torch,
    get_time,
    save_arg,
    join,
    exists,
    compute_sdf,
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
        default=800,
        metavar="N",
        help="number of epochs to train_tumor (default: 800)",
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
        default="0,1",
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
        default="renji,rmyy",
        metavar="str",
        help="dataset for training",
    )
    parser.add_argument(
        "--workers", type=int, default=12, metavar="N", help="num_worker (default: 16)"
    )
    parser.add_argument("--log", action="store_false", help="write logs or not")
    return parser.parse_args()


args = get_args()
seed_torch(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from models.U_CorResNet_fix import U_CorResNet_Fix_Contrastive_Proj_DT
from models.loss import get_contrastive_loss_v2, DSC_loss, BCELoss

rmyy_list = os.listdir(rmyy_panc_tumor_label_path)
renji_list = os.listdir(renji_panc_tumor_label_path)
rmyy_list = [int(i[0:3]) for i in rmyy_list]
renji_list = [int(i[:-5]) for i in renji_list]


class Generalize_Tumor_Skeleton(Dataset):
    def __init__(self, data, augment=True, sdf=False, gray=False, size=6000000):

        self.aug = self.get_augmentation()
        self.is_augment = augment
        self.dataset = data
        self.sdf = sdf
        self.gray = gray
        self.size = size

        self.patient_index = []
        self.data_path = []
        self.label_path = []

        if "msd" in self.dataset.lower():
            self.patient_index += list(itertools.chain(*msd_mix_list))
            for i in range(len(self.patient_index)):
                self.data_path.append(msd_resample_data_path)
                self.label_path.append(msd_resample_multi_label_path)

        if "rmyy" in self.dataset.lower():
            self.patient_index += rmyy_list
            for i in range(len(rmyy_list)):
                self.data_path.append(rmyy_data_path)
                self.label_path.append(rmyy_panc_tumor_label_path)

        if "renji" in self.dataset.lower():
            self.patient_index += renji_list
            for i in range(len(renji_list)):
                self.data_path.append(renji_data_path)
                self.label_path.append(renji_panc_tumor_label_path)

    def get_augmentation(self):
        return Compose(
            [
                RandomScale((0.8, 1.2)),
                Resize(always_apply=True),
                ElasticTransform((0, 0.1), p=0.1),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                RandomGamma(),
                RandomGaussianNoise(),
                Normalize(always_apply=True),
            ],
            p=1,
        )

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        background = label == 0
        pancreas = label == 1
        tumors = label == 2

        results_map[0, :, :, :] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(pancreas, 1, 0)
        results_map[2, :, :, :] = np.where(tumors, 1, 0)

        return results_map

    def crop_by_pancreas(self, image, mask, pancreas):
        arr = np.nonzero(pancreas)
        minA = max(0, min(arr[0]) - int(len(pancreas) * 0.01 * random.randint(1, 10)))
        maxA = min(
            len(pancreas),
            max(arr[0]) + int(len(pancreas) * 0.01 * random.randint(1, 10)),
        )
        MARGIN = random.randint(0, 30)
        minB = max(0, min(arr[1]) - MARGIN)
        MARGIN = random.randint(0, 30)
        maxB = min(512, max(arr[1]) + MARGIN)
        MARGIN = random.randint(0, 30)
        minC = max(0, min(arr[2]) - MARGIN)
        MARGIN = random.randint(0, 30)
        maxC = min(512, max(arr[2]) + MARGIN)

        if (maxA - minA) % 8 != 0:
            max_A = 8 * (int((maxA - minA) / 8) + 1)
            gap = int((max_A - (maxA - minA)) / 2)
            minA = max(0, minA - gap)
            maxA = min(len(mask), minA + max_A)
            if maxA == len(mask):
                minA = maxA - max_A

        if (maxB - minB) % 8 != 0:
            max_B = 8 * (int((maxB - minB) / 8) + 1)
            gap = int((max_B - (maxB - minB)) / 2)
            minB = max(0, minB - gap)
            maxB = min(512, minB + max_B)
            if maxB == 512:
                minB = maxB - max_B

        if (maxC - minC) % 8 != 0:
            max_C = 8 * (int((maxC - minC) / 8) + 1)
            gap = int((max_C - (maxC - minC)) / 2)
            minC = max(0, minC - gap)
            maxC = min(512, minC + max_C)
            if maxC == 512:
                minC = maxC - max_C

        image, mask = (
            image[minA:maxA, minB:maxB, minC:maxC],
            mask[minA:maxA, minB:maxB, minC:maxC],
        )

        box = [minA, maxA, minB, maxB, minC, maxC]

        return image.copy(), mask.copy(), box

    def adjust_size(self, z, h, w):
        if z % 8 != 0:
            z = int(z / 8) * 8

        if h != 0:
            h = int(h / 8) * 8

        if w != 0:
            w = int(w / 8) * 8

        return z, h, w

    def __getitem__(self, index):

        if (
            "msd" in self.data_path[index].lower()
            or "task07" in self.data_path[index].lower()
        ):
            image = (
                np.load(
                    join(
                        self.data_path[index],
                        "{}.npy".format(self.patient_index[index]),
                    )
                )
                .transpose(2, 0, 1)
                .astype(np.float64)
            )
            mask = (
                np.load(
                    join(
                        self.label_path[index],
                        "{}.npy".format(self.patient_index[index]),
                    )
                )
                .transpose(2, 0, 1)
                .astype(np.float64)
            )

        elif (
            "rmyy" in self.data_path[index].lower()
            or "renming" in self.data_path[index].lower()
        ):
            image = (
                nrrd.read(
                    join(
                        self.data_path[index],
                        "{:0>3}.nrrd".format(self.patient_index[index]),
                    )
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float64)
            )
            mask = (
                nrrd.read(
                    join(
                        self.label_path[index],
                        "{:0>3}.nrrd".format(self.patient_index[index]),
                    )
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float64)
            )

        elif "renji" in self.data_path[index].lower():
            image = nibabel.load(
                join(
                    self.data_path[index], "{}.nii.gz".format(self.patient_index[index])
                )
            ).get_fdata()

            if len(image.shape) == 5:
                image = image[:, :, :, 0, 0]

            image = image.transpose((2, 1, 0)).astype(np.float64)

            mask = (
                nrrd.read(
                    join(
                        self.label_path[index],
                        "{}.nrrd".format(self.patient_index[index]),
                    )
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float64)
            )

        if np.max(image) > 1:
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        image, mask, _ = self.crop_by_pancreas(image, mask, mask)

        if mask.shape[-3] * mask.shape[-2] * mask.shape[-1] > self.size:
            # print("POOL...")
            spc_ratio = self.size / (mask.shape[-3] * mask.shape[-2] * mask.shape[-1])
            z, h, w = self.adjust_size(
                math.ceil(mask.shape[-3] * spc_ratio ** (1 / 3)),
                math.ceil(mask.shape[-2] * spc_ratio ** (1 / 3)),
                math.ceil(mask.shape[-1] * spc_ratio ** (1 / 3)),
            )
            image = resize(
                image,
                (z, h, w),
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                preserve_range=True,
            )
            mask = resize(
                mask,
                (z, h, w),
                order=0,
                mode="edge",
                cval=0,
                clip=True,
                preserve_range=True,
            )
            mask = np.round(mask)

        else:
            if random.random() < 0.2:
                sum_p = mask.shape[-3] * mask.shape[-2] * mask.shape[-1]
                new_sum_p = int(sum_p * random.uniform(0.8, 1.2))
                if new_sum_p < self.size:
                    spc_ratio = new_sum_p / (
                        mask.shape[-3] * mask.shape[-2] * mask.shape[-1]
                    )
                    z, h, w = self.adjust_size(
                        math.ceil(mask.shape[-3] * spc_ratio ** (1 / 3)),
                        math.ceil(mask.shape[-2] * spc_ratio ** (1 / 3)),
                        math.ceil(mask.shape[-1] * spc_ratio ** (1 / 3)),
                    )
                    image = resize(
                        image,
                        (z, h, w),
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        preserve_range=True,
                    )
                    mask = resize(
                        mask,
                        (z, h, w),
                        order=0,
                        mode="edge",
                        cval=0,
                        clip=True,
                        preserve_range=True,
                    )
                    mask = np.round(mask)

        image, mask = image.astype(np.float64), mask.astype(np.float64)

        if self.is_augment:
            temp_data = {
                "image": image.transpose(1, 2, 0),
                "mask": mask.transpose(1, 2, 0),
                "size": mask.transpose(1, 2, 0).shape,
            }

            aug_data = self.aug(**temp_data)
            image, mask = (
                aug_data["image"].transpose(2, 0, 1),
                aug_data["mask"].transpose(2, 0, 1),
            )

        tumor = mask.copy()
        tumor[tumor < 2] = 0
        tumor[tumor >= 2] = 1

        if self.gray and random.random() < 0.3:
            random_array = random.uniform(-0.3, 0.3) * np.ones(mask.shape)
            k = random.randint(5, 20)
            mask_temp = morphology.dilation(
                tumor.copy().astype(np.uint8), np.ones([k - 3, k, k])
            )
            image = image + random_array * mask_temp
            image = np.clip(image, 0, 1)
            image = image.astype(np.float64)

        if self.sdf:
            gt_dis = compute_sdf(
                np.array([[tumor]]),
                np.array([[tumor]]).shape,
                self.label_path[index],
                self.patient_index[index],
            )
            gt_dis = gt_dis.astype(np.float32)

        k = random.randint(10, 20)
        mask_dilation = morphology.dilation(
            tumor.astype(np.uint8), np.ones([k - 3, k, k])
        )[::8, ::8, ::8]

        skeleton_lee_tumor = skeletonize_3d(tumor[::8, ::8, ::8].astype(np.uint8))
        tumor_mask_pool = tumor[::8, ::8, ::8]

        image = np.array([image]).astype(np.float32)
        label = np.array([tumor]).astype(np.float32)
        mask_dilation = np.array([mask_dilation]).astype(np.float32)
        skeleton_lee_tumor = np.array([skeleton_lee_tumor]).astype(np.float32)
        tumor_mask_pool = np.array([tumor_mask_pool]).astype(np.float32)

        if self.sdf:
            return (
                image.copy(),
                label.copy(),
                tumor_mask_pool.copy(),
                mask_dilation.copy(),
                skeleton_lee_tumor.copy(),
                np.squeeze(gt_dis, axis=0).copy(),
            )
        else:
            return (
                image.copy(),
                label.copy(),
                tumor_mask_pool.copy(),
                mask_dilation.copy(),
                skeleton_lee_tumor.copy(),
            )

    def __len__(self):
        return len(self.patient_index)


def generalize_train_pool_u_conresnet(args):
    log_file_name = "train_logs.txt"

    checkpoint_path = join("../checkpoints", args.time)
    if args.log:
        if not exists(join("../checkpoints", args.time)):
            os.mkdir(join("../checkpoints", args.time))
        save_arg(args, join("../checkpoints", args.time, log_file_name))

    model = U_CorResNet_Fix_Contrastive_Proj_DT().train()
    model = nn.DataParallel(model).cuda()
    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

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

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    dset_train = Generalize_Tumor_Skeleton(
        data=args.data, augment=True, sdf=True, size=5000000
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
    bgrTh = nn.Threshold(0, 1)
    tarTh = nn.Threshold(0.999999999, 0)
    criterion_mse = nn.MSELoss()
    model.train()

    # train_tumor for some epochs
    for i in range(args.epochs):

        loss_dsc_list = []
        loss_bce_list = []
        loss_con_list = []
        loss_dist_list = []
        loss_dsc_dist_list = []
        loss_mse_list = []

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

                assert mask_dilation.shape == skeleton_lee.shape
                assert mask_pool.shape == skeleton_lee.shape

                try:
                    optimizer.zero_grad()
                    # print(images.shape)
                    output, emb, out_dis = model(images)
                    out_put_sig = torch.sigmoid(output)
                    loss_dsc = criterion_dsc(out_put_sig, masks)
                    loss_bce = criterion_bce(output, masks)
                    loss = loss_dsc + loss_bce
                    loss_dsc_list.append(loss_dsc.item())
                    loss_bce_list.append(loss_bce.item())

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
                    while neg_patch is None and tt < 5:
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
                    else:
                        print(
                            "Distance Error!",
                            mask.sum(),
                            gt_diss.sum(),
                            gt_diss.max(),
                            gt_diss.min(),
                        )

                    if len(masks) > len(mask):
                        pred1 = out_put_sig[0 : len(mask)]
                        pred2 = out_put_sig[len(mask) :]
                        loss_cos = criterion_mse(pred1, pred2)
                        loss += loss_cos
                        loss_mse_list.append(loss_cos.item())

                    out_dis = bgrTh(out_dis)
                    out_dis = tarTh(out_dis)
                    loss_dis_dsc = criterion_dsc(out_dis, masks)
                    loss_dsc_dist_list.append(loss_dis_dsc.item())
                    loss += loss_dis_dsc

                    loss.backward()
                    optimizer.step()

                except RuntimeError or ValueError:
                    print(images.size())
                    traceback.print_exc()

                torch.cuda.empty_cache()

                t.set_postfix(
                    dsc=np.mean(loss_dsc_list),
                    bce=np.mean(loss_bce_list),
                    con=np.mean(loss_con_list),
                    mse=np.mean(loss_mse_list),
                    dist=np.mean(loss_dist_list),
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
                    "E %s, ave_bce_loss=%s, ave_dsc_loss=%s, ave_con_loss=%s, ave_mse_loss=%s, ave_dist_loss=%s, ave_dsc_dist_loss=%s \n"
                    % (
                        i,
                        np.mean(loss_bce_list),
                        np.mean(loss_dsc_list),
                        np.mean(loss_con_list),
                        np.mean(loss_mse_list),
                        np.mean(loss_dist_list),
                        np.mean(loss_dsc_dist_list),
                    )
                )

        if (i + 1) % 10 == 0 or i >= (args.epochs - 10):
            torch.save(
                model.module.state_dict(),
                join(checkpoint_path, "Tumor_3D_Fine_%s.pth" % args.data),
            )


if __name__ == "__main__":

    if args.time == "time":
        args.time = get_time()
    args.time = "Tumor_3D_Fine_On_%s_%s" % (args.data, args.time)
    generalize_train_pool_u_conresnet(args)
