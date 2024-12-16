import argparse
import os
import sys
import traceback

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.util import (
    adjust_learning_rate_D,
    seed_torch,
    get_time,
    save_arg,
    join,
    exists,
)
from models.RandConv import RandConvModule
from utils.get_patches import (
    get_pos_patch_from_skeleton_none,
    get_neg_patch_from_skeleton_none,
    get_pos_patch_from_skeleton_dual,
    get_neg_patch_from_skeleton_dual,
)
from config.config import (
    nih_train_list,
    nih_data_path,
    nih_label_path,
    msd_mix_list,
    msd_resample_data_path,
    msd_resample_multi_label_path,
    renji_data_path,
    LOW_RANGE,
    HIGH_RANGE,
    rmyy_data_path,
    rmyy_panc_tumor_label_path,
    renji_panc_tumor_label_path,
)
from utils.util import padding_z, center_crop

from torch import optim
from tqdm import tqdm
from pytorch_metric_learning import losses
import itertools
import random
from os.path import join
import nibabel
import nrrd
import numpy as np
import torch
from skimage import morphology
from skimage.morphology import skeletonize_3d
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from augmentation.volumentations import (
    Compose,
    RandomScale,
    Resize,
    ElasticTransform,
    RandomRotate,
    RandomFlip,
    RandomGamma,
    RandomGaussianNoise,
)

Is_Flip = [0, 1, 0, 1, 0, 1]
rmyy_list = os.listdir(rmyy_panc_tumor_label_path)
renji_list = os.listdir(renji_panc_tumor_label_path)
rmyy_list = [int(i[0:3]) for i in rmyy_list]
renji_list = [int(i[:-5]) for i in renji_list]


class Generalize_Pancreas_Dataset_Skeleton(Dataset):
    def __init__(
            self,
            data,
            augment=True,
            resize=True,
            crop=True,
            pool=False,
            check=False,
            return_source=False,
            size=6000000,
    ):

        self.aug = self.get_augmentation()
        self.is_augment = augment
        self.dataset = data
        self.resize = resize
        self.crop = crop
        self.size = size
        self.pool = pool
        self.check = check
        self.return_source = return_source
        self.patient_index = []
        self.data_path = []
        self.label_path = []

        if "msd" not in self.dataset.lower():
            msd_list = list(itertools.chain(*msd_mix_list))
            self.patient_index += msd_list
            for i in range(len(msd_list)):
                self.data_path.append(msd_resample_data_path)
                self.label_path.append(msd_resample_multi_label_path)

        if "rmyy" not in self.dataset.lower():
            self.patient_index += rmyy_list
            for i in range(len(rmyy_list)):
                self.data_path.append(rmyy_data_path)
                self.label_path.append(rmyy_panc_tumor_label_path)

        if "renji" not in self.dataset.lower():
            self.patient_index += renji_list
            for i in range(len(renji_list)):
                self.data_path.append(renji_data_path)
                self.label_path.append(renji_panc_tumor_label_path)

        if "nih" not in self.dataset.lower():
            self.patient_index += nih_train_list
            for i in range(len(nih_train_list)):
                self.data_path.append(nih_data_path)
                self.label_path.append(nih_label_path)

    def crop_by_pancreas(self, image, mask, pancreas):
        arr = np.nonzero(pancreas)
        minA = max(0, min(arr[0]) - int(len(mask) * 0.01 * random.randint(-5, 5)))
        maxA = min(
            len(mask), max(arr[0]) + int(len(mask) * 0.01 * random.randint(-5, 5))
        )
        if minA >= maxA:
            minA = max(0, min(arr[0]) - int(len(mask) * 0.01 * random.randint(0, 10)))
            maxA = min(
                len(mask), max(arr[0]) + int(len(mask) * 0.01 * random.randint(0, 10))
            )
        MARGIN = random.randint(-20, 20)
        minB = max(0, min(arr[1]) - MARGIN)
        MARGIN = random.randint(-20, 20)
        maxB = min(512, max(arr[1]) + MARGIN)
        if minB >= maxB:
            MARGIN = random.randint(0, 20)
            minB = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 20)
            maxB = min(512, max(arr[1]) + MARGIN)
        MARGIN = random.randint(-20, 20)
        minC = max(0, min(arr[2]) - MARGIN)
        MARGIN = random.randint(-20, 20)
        maxC = min(512, max(arr[2]) + MARGIN)
        if minC >= maxC:
            MARGIN = random.randint(0, 20)
            minC = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 20)
            maxC = min(512, max(arr[1]) + MARGIN)

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

        return image, mask

    def get_augmentation(self):
        return Compose(
            [
                RandomScale((0.8, 1.2)),
                Resize(always_apply=True),
                ElasticTransform((0, 0.25)),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                RandomGamma(),
                RandomGaussianNoise(),
            ],
            p=1,
        )

    def adjust_size(self, z, h, w):
        if z % 8 != 0:
            z = int(z / 8) * 8

        if h != 0:
            h = int(h / 8) * 8

        if w != 0:
            w = int(w / 8) * 8

        return z, h, w

    def __getitem__(self, index):

        if "nih" in self.data_path[index] or "NIH" in self.data_path[index]:
            data_source = "nih"
            image = (
                np.load(
                    join(
                        self.data_path[index],
                        "{:0>4}.npy".format(self.patient_index[index]),
                    )
                )
                .transpose(2, 0, 1)
                .astype(np.float64)
            )
            mask = np.load(
                join(
                    self.label_path[index],
                    "{:0>4}.npy".format(self.patient_index[index]),
                )
            ).transpose(2, 0, 1)

            image = np.flip(image, axis=1)
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=1)
            mask = np.flip(mask, axis=0)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

        elif "msd" in self.data_path[index] or "Task07" in self.data_path[index]:
            data_source = "msd"
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
            mask = np.load(
                join(self.label_path[index], "{}.npy".format(self.patient_index[index]))
            ).transpose(2, 0, 1)

            image = np.flip(image, axis=1)
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=1)
            mask = np.flip(mask, axis=0)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

        elif "renji" in self.data_path[index].lower():
            data_source = "renji"
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

        elif (
                "rmyy" in self.data_path[index].lower()
                or "renming" in self.data_path[index].lower()
        ):
            data_source = "rmyy"
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

        mask[mask > 0] = 1
        image, mask = image.astype(np.float64), mask.astype(np.float64)

        if self.crop:
            image, mask = self.crop_by_pancreas(image, mask, mask)

        if np.max(image) > 1:
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        if self.pool:
            image, mask = padding_z(image, mask)
            image, mask = (center_crop(image, 320, 368), center_crop(mask, 320, 368))
            image, mask = image[::2, ::2, ::2], mask[::2, ::2, ::2]

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

            mn = image.min()
            mx = image.max()
            image = (image - mn) / (mx - mn)

        if self.resize:
            if mask.shape[-3] * mask.shape[-2] * mask.shape[-1] > self.size:
                spc_ratio = self.size / (
                        mask.shape[-3] * mask.shape[-2] * mask.shape[-1]
                )
                z, h, w = self.adjust_size(
                    int(mask.shape[-3] * spc_ratio),
                    int(mask.shape[-2] * spc_ratio),
                    int(mask.shape[-1] * spc_ratio),
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
                mask[mask > 0] = 1

        if self.check:
            if image.shape[0] > 144:
                image = image[
                        int((image.shape[0] - 144) / 2): int((image.shape[0] - 144) / 2)
                                                         + 144,
                        ]
                mask = mask[
                       int((image.shape[0] - 144) / 2): int((image.shape[0] - 144) / 2)
                                                        + 144,
                       ]

        k = random.randint(10, 20)
        mask_dilation = morphology.dilation(
            mask.astype(np.uint8), np.ones([k, k, k - 3])
        )[::8, ::8, ::8]
        skeleton_lee = skeletonize_3d(mask[::8, ::8, ::8].astype(np.uint8))
        mask_pool = mask[::8, ::8, ::8]

        image, mask, mask_pool, mask_dilation, skeleton_lee = (
            np.expand_dims(image, axis=0),
            np.expand_dims(mask, axis=0),
            np.expand_dims(mask_pool, axis=0),
            np.expand_dims(mask_dilation, axis=0),
            np.expand_dims(skeleton_lee, axis=0),
        )

        data = torch.from_numpy(image).clone()
        label = torch.from_numpy(mask).clone()
        mask_pool = torch.from_numpy(mask_pool).clone()
        mask_dilation = torch.from_numpy(mask_dilation).clone()
        skeleton_lee = torch.from_numpy(skeleton_lee).clone()

        if not self.return_source:
            return data, label, mask_pool, mask_dilation, skeleton_lee
        else:
            return data, label, mask_pool, mask_dilation, skeleton_lee, data_source

    def __len__(self):
        return len(self.patient_index)


def get_args():
    parser = argparse.ArgumentParser(description="3D Pancreas segmentation")
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
    dset_train = Generalize_Pancreas_Dataset_Skeleton(
        data=args.data, augment=True, resize=False, crop=False, pool=True, check=True
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
        loss_cos_list = []

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

                with torch.no_grad():
                    rand_conv.randomize()
                    image1 = rand_conv(image)
                    rand_conv.randomize()
                    image2 = rand_conv(image)

                image1 = (image1 - image1.min()) / (image1.max() - image1.min())
                image2 = (image2 - image2.min()) / (image2.max() - image2.min())

                images = torch.cat((image1, image2), dim=0)
                masks = torch.cat((mask, mask), dim=0)

                if torch.isnan(images).any():
                    images = image
                    masks = mask

                assert mask_dilation.shape == skeleton_lee.shape
                assert mask_pool.shape == skeleton_lee.shape

                try:
                    optimizer.zero_grad()
                    output, emb = model(images)
                    out_put_sig = torch.sigmoid(output)
                    loss_dsc = criterion_dsc(out_put_sig, masks)
                    loss_bce = criterion_bce(output, masks)
                    loss = loss_dsc + loss_bce

                    loss_dsc_list.append(loss_dsc.item())
                    loss_bce_list.append(loss_bce.item())

                    if len(masks) == 1:
                        pos_patch = get_pos_patch_from_skeleton_none(
                            emb, mask_pool, skeleton_lee, k=0, sum_pixel=10
                        )
                        neg_patch = get_neg_patch_from_skeleton_none(
                            mask_dilation, emb, mask_pool, k=0, sum_pixel=5
                        )
                    elif len(masks) > 1:
                        pos_patch = get_pos_patch_from_skeleton_dual(
                            emb, mask_pool, skeleton_lee, k=0, sum_pixel=10
                        )
                        neg_patch = get_neg_patch_from_skeleton_dual(
                            mask_dilation, emb, mask_pool, k=0, sum_pixel=5
                        )

                    if pos_patch is not None and neg_patch is not None:
                        loss_con = get_contrastive_loss_v2(
                            pos_patch, neg_patch, cont_loss_func, k=0
                        )
                        loss += loss_con * 0.1
                        loss_con_list.append(loss_con.item())

                    if len(masks) > 1:
                        pred1 = out_put_sig[0:1]
                        pred2 = out_put_sig[1:2]

                        cos_sim = torch.cosine_similarity(
                            pred1.view(1, -1).squeeze(),
                            pred2.view(1, -1).squeeze(),
                            dim=0,
                        )

                        loss_cos = 1 - cos_sim
                        loss += loss_cos
                        loss_cos_list.append(loss_cos.item())

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
                    cos=np.mean(loss_cos_list),
                )

                adjust_learning_rate_D(
                    optimizer,
                    args.lr,
                    i * int(len(train_loader)) + batch_idx,
                    args.epochs * int(len(train_loader)),
                )

        with open(join("../checkpoints", args.time, "%s" % log_file_name), "a") as f:
            f.write(
                "E %s, ave_bce_loss=%s, ave_dsc_loss=%s, ave_con_loss=%s, avg_cos_loss=%s \n"
                % (
                    i,
                    np.mean(loss_bce_list),
                    np.mean(loss_dsc_list),
                    np.mean(loss_con_list),
                    np.mean(loss_cos_list),
                )
            )

        if (i + 1) % 10 == 0 or i >= (args.epochs - 10):
            torch.save(
                model.state_dict(),
                join(
                    checkpoint_path,
                    "Generalize_Pancreas_Coarse_%s.pth" % args.data,
                ),
            )


if __name__ == "__main__":

    datas = list(map(str, args.data.split(",")))
    if args.time == "time":
        args.time = get_time()
    for data in datas:
        args.data = data
        args.time = "Coarse_Pancreas_Generalize_%s_%s" % (
            args.data,
            args.time,
        )
        generalize_train_pool_u_conresnet(args)
