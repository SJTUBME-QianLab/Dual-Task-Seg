import math
import os
import random
from os.path import join

import nibabel
import nrrd
import numpy as np
from skimage import morphology
from skimage.morphology import skeletonize_3d
from skimage.transform import resize
from torch.utils.data import Dataset

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
from config.config import (
    msd_resample_data_path,
    msd_resample_multi_label_path,
    renji_data_path,
    LOW_RANGE,
    HIGH_RANGE,
    rmyy_data_path,
    rmyy_panc_tumor_label_path,
    renji_panc_tumor_label_path,
)
from utils.util import compute_sdf

Is_Flip = [0, 1, 0, 1, 0, 1]


class Generalize_Tumor_Skeleton_in_Tumor_MSD_RMYY(Dataset):
    def __init__(self, data, augment=True, sdf=False):

        self.aug = self.get_augmentation()
        self.is_augment = augment
        self.dataset = data
        self.sdf = sdf

        self.patient_index = []
        self.data_path = []
        self.label_path = []

        if "msd" in self.dataset.lower():
            self.patient_index += os.listdir(msd_resample_multi_label_path)
            for i in range(len(self.patient_index)):
                self.data_path.append(msd_resample_data_path)
                self.label_path.append(msd_resample_multi_label_path)

        if "rmyy" in self.dataset.lower():
            self.patient_index += os.listdir(rmyy_panc_tumor_label_path)
            for i in range(len(os.listdir(rmyy_panc_tumor_label_path))):
                self.data_path.append(rmyy_data_path)
                self.label_path.append(rmyy_panc_tumor_label_path)

        if "renji" in self.dataset.lower():
            self.patient_index += os.listdir(renji_panc_tumor_label_path)
            for i in range(len(os.listdir(renji_panc_tumor_label_path))):
                self.data_path.append(renji_data_path)
                self.label_path.append(renji_panc_tumor_label_path)

    def get_augmentation(self):
        return Compose(
            [
                # RemoveEmptyBorder(always_apply=True),
                RandomScale((0.8, 1.2)),
                # PadIfNeeded(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                # CenterCrop(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                Resize(always_apply=True),
                # CropNonEmptyMaskIfExists(patch_size, always_apply=True),
                ElasticTransform((0, 0.1), p=0.1),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                # Transpose((1,0,2)), # only if patch.height = patch.width
                # RandomRotate90((0,1)),
                RandomGamma(),
                RandomGaussianNoise(),
                Normalize(always_apply=True),
            ],
            p=1,
        )

    def crop(self, image, mask):
        arr = np.nonzero(mask)
        minA = max(0, min(arr[0]) - int(len(mask) * 0.01 * random.randint(-1, 5)))
        maxA = min(
            len(mask), max(arr[0]) + int(len(mask) * 0.01 * random.randint(-1, 5))
        )
        if minA >= maxA:
            minA = max(0, min(arr[0]) - int(len(mask) * 0.01 * random.randint(0, 10)))
            maxA = min(
                len(mask), max(arr[0]) + int(len(mask) * 0.01 * random.randint(0, 10))
            )
        MARGIN = random.randint(-10, 40)
        minB = max(0, min(arr[1]) - MARGIN)
        MARGIN = random.randint(-10, 40)
        maxB = min(512, max(arr[1]) + MARGIN)
        if minB >= maxB:
            MARGIN = random.randint(0, 40)
            minB = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxB = min(512, max(arr[1]) + MARGIN)
        MARGIN = random.randint(-10, 40)
        minC = max(0, min(arr[2]) - MARGIN)
        MARGIN = random.randint(-10, 40)
        maxC = min(512, max(arr[2]) + MARGIN)
        if minC >= maxC:
            MARGIN = random.randint(0, 40)
            minC = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxC = min(512, max(arr[1]) + MARGIN)

        if (maxA - minA) % 8 != 0:
            max_A = 8 * (int((maxA - minA) / 8) + 1)
            gap = int((max_A - (maxA - minA)) / 2)
            minA = max(0, minA - gap)
            maxA = min(len(mask), minA + max_A)
            if maxA == len(mask):
                minA = maxA - max_A

        if (maxB - minB) % 8 != 0 or (maxB - minB) <= 8:
            max_B = 8 * (int((maxB - minB) / 8) + 1)
            gap = int((max_B - (maxB - minB)) / 2)
            minB = max(0, minB - gap)
            maxB = min(512, minB + max_B)
            if maxB == 512:
                minB = maxB - max_B

        if (maxC - minC) % 8 != 0 or (maxC - minC) <= 8:
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
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )
            mask = (
                np.load(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )

        elif (
            "rmyy" in self.data_path[index].lower()
            or "renming" in self.data_path[index].lower()
        ):
            image = (
                nrrd.read(
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )
            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        elif "renji" in self.data_path[index].lower():
            image = nibabel.load(
                join(
                    self.data_path[index],
                    "{}".format(self.patient_index[index].replace(".nrrd", ".nii.gz")),
                )
            ).get_fdata()

            if len(image.shape) == 5:
                image = image[:, :, :, 0, 0]

            image = image.transpose((2, 1, 0)).astype(np.float)

            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        # if np.max(image) > 1:
        #     np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
        #     image -= LOW_RANGE
        #     image /= HIGH_RANGE - LOW_RANGE

        mask[mask != 2] = 0
        mask[mask == 2] = 1
        # print("ori image", np.min(image), np.max(image))
        image, mask, _ = self.crop(image, mask)

        if np.max(image) > 1:
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        # print("cropped image", np.min(image), np.max(image))
        image, mask = image.astype(np.float), mask.astype(np.float)

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

        image, mask = image.astype(np.float), mask.astype(np.float)
        # print(np.max(image), np.max(mask))

        # if (
        #     mask.shape[-3] % 8 != 0
        #     or mask.shape[-2] % 8 != 0
        #     or mask.shape[-1] % 8 != 0
        # ):
        #
        #     z, h, w = self.adjust_size(mask.shape[-3], mask.shape[-2], mask.shape[-1])
        #     image = resize(
        #         image,
        #         (z, h, w),
        #         order=3,
        #         mode="edge",
        #         cval=0,
        #         clip=True,
        #         preserve_range=True,
        #     )
        #     mask = resize(
        #         mask,
        #         (z, h, w),
        #         order=0,
        #         mode="edge",
        #         cval=0,
        #         clip=True,
        #         preserve_range=True,
        #     )
        #     mask = np.round(mask)
        #     mask[mask > 0] = 1

        # print("resized image", np.min(image), np.max(image))

        tumor = mask.copy()

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


class Generalize_Tumor_Skeleton_in_Tumor_Renji_RMYY(Dataset):
    def __init__(self, data, augment=True, sdf=False, size=5000000):

        self.aug = self.get_augmentation()
        self.is_augment = augment
        self.dataset = data
        self.sdf = sdf
        self.size = size

        self.patient_index = []
        self.data_path = []
        self.label_path = []

        if "msd" in self.dataset.lower():
            self.patient_index += os.listdir(msd_resample_multi_label_path)
            for i in range(len(self.patient_index)):
                self.data_path.append(msd_resample_data_path)
                self.label_path.append(msd_resample_multi_label_path)

        if "rmyy" in self.dataset.lower():
            self.patient_index += os.listdir(rmyy_panc_tumor_label_path)
            for i in range(len(os.listdir(rmyy_panc_tumor_label_path))):
                self.data_path.append(rmyy_data_path)
                self.label_path.append(rmyy_panc_tumor_label_path)

        if "renji" in self.dataset.lower():
            self.patient_index += os.listdir(renji_panc_tumor_label_path)
            for i in range(len(os.listdir(renji_panc_tumor_label_path))):
                self.data_path.append(renji_data_path)
                self.label_path.append(renji_panc_tumor_label_path)

    def get_augmentation(self):
        return Compose(
            [
                # RemoveEmptyBorder(always_apply=True),
                RandomScale((0.8, 1.2)),
                # PadIfNeeded(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                # CenterCrop(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                Resize(always_apply=True),
                # CropNonEmptyMaskIfExists(patch_size, always_apply=True),
                ElasticTransform((0, 0.1), p=0.1),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                # Transpose((1,0,2)), # only if patch.height = patch.width
                # RandomRotate90((0,1)),
                RandomGamma(),
                RandomGaussianNoise(),
                Normalize(always_apply=True),
            ],
            p=1,
        )

    def crop(self, image, mask):
        arr = np.nonzero(mask)
        minA = max(0, min(arr[0]) - math.ceil(len(mask) * 0.01 * random.randint(1, 8)))
        maxA = min(
            len(mask), max(arr[0]) + math.ceil(len(mask) * 0.01 * random.randint(1, 8))
        )
        MARGIN = random.randint(-5, 40)
        minB = max(0, min(arr[1]) - MARGIN)
        MARGIN = random.randint(-5, 40)
        maxB = min(512, max(arr[1]) + MARGIN)
        if minB >= maxB:
            MARGIN = random.randint(0, 40)
            minB = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxB = min(512, max(arr[1]) + MARGIN)
        MARGIN = random.randint(-5, 40)
        minC = max(0, min(arr[2]) - MARGIN)
        MARGIN = random.randint(-5, 40)
        maxC = min(512, max(arr[2]) + MARGIN)
        if minC >= maxC:
            MARGIN = random.randint(0, 40)
            minC = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxC = min(512, max(arr[1]) + MARGIN)

        if (maxA - minA) % 8 != 0:
            max_A = 8 * (int((maxA - minA) / 8) + 1)
            gap = int((max_A - (maxA - minA)) / 2)
            minA = max(0, minA - gap)
            maxA = min(len(mask), minA + max_A)
            if maxA == len(mask):
                minA = maxA - max_A

        if (maxB - minB) % 8 != 0 or (maxB - minB) <= 8:
            max_B = 8 * (int((maxB - minB) / 8) + 1)
            gap = int((max_B - (maxB - minB)) / 2)
            minB = max(0, minB - gap)
            maxB = min(512, minB + max_B)
            if maxB == 512:
                minB = maxB - max_B

        if (maxC - minC) % 8 != 0 or (maxC - minC) <= 8:
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
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )
            mask = (
                np.load(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )

        elif (
            "rmyy" in self.data_path[index].lower()
            or "renming" in self.data_path[index].lower()
        ):
            image = (
                nrrd.read(
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )
            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        elif "renji" in self.data_path[index].lower():
            image = nibabel.load(
                join(
                    self.data_path[index],
                    "{}".format(self.patient_index[index].replace(".nrrd", ".nii.gz")),
                )
            ).get_fdata()

            if len(image.shape) == 5:
                image = image[:, :, :, 0, 0]

            image = image.transpose((2, 1, 0)).astype(np.float)

            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        mask[mask != 2] = 0
        mask[mask == 2] = 1
        image, mask, _ = self.crop(image, mask)

        if np.max(image) > 1:
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        if mask.shape[-3] * mask.shape[-2] * mask.shape[-1] > self.size:
            # print("pool...")
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
            if random.random() < 0.3:
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

        image, mask = image.astype(np.float64), mask.astype(np.float64)
        tumor = mask.copy()

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


class Generalize_Tumor_Skeleton_in_Tumor_MSD_Renji(Dataset):
    def __init__(self, data, augment=True, sdf=False, size=5000000):

        self.aug = self.get_augmentation()
        self.is_augment = augment
        self.dataset = data
        self.sdf = sdf
        self.size = size

        self.patient_index = []
        self.data_path = []
        self.label_path = []

        if "msd" in self.dataset.lower():
            self.patient_index += os.listdir(msd_resample_multi_label_path)
            for i in range(len(self.patient_index)):
                self.data_path.append(msd_resample_data_path)
                self.label_path.append(msd_resample_multi_label_path)

        if "rmyy" in self.dataset.lower():
            self.patient_index += os.listdir(rmyy_panc_tumor_label_path)
            for i in range(len(os.listdir(rmyy_panc_tumor_label_path))):
                self.data_path.append(rmyy_data_path)
                self.label_path.append(rmyy_panc_tumor_label_path)

        if "renji" in self.dataset.lower():
            self.patient_index += os.listdir(renji_panc_tumor_label_path)
            for i in range(len(os.listdir(renji_panc_tumor_label_path))):
                self.data_path.append(renji_data_path)
                self.label_path.append(renji_panc_tumor_label_path)

    def get_augmentation(self):
        return Compose(
            [
                # RemoveEmptyBorder(always_apply=True),
                RandomScale((0.8, 1.2)),
                # PadIfNeeded(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                # CenterCrop(patch_size, always_apply=True),
                # RandomCrop(patch_size, always_apply=True),
                Resize(always_apply=True),
                # CropNonEmptyMaskIfExists(patch_size, always_apply=True),
                ElasticTransform((0, 0.1), p=0.1),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                # Transpose((1,0,2)), # only if patch.height = patch.width
                # RandomRotate90((0,1)),
                RandomGamma(),
                RandomGaussianNoise(),
                Normalize(always_apply=True),
            ],
            p=1,
        )

    def crop(self, image, mask):
        arr = np.nonzero(mask)
        minA = max(0, min(arr[0]) - math.ceil(len(mask) * 0.01 * random.randint(1, 8)))
        maxA = min(
            len(mask), max(arr[0]) + math.ceil(len(mask) * 0.01 * random.randint(1, 8))
        )
        MARGIN = random.randint(-5, 40)
        minB = max(0, min(arr[1]) - MARGIN)
        MARGIN = random.randint(-5, 40)
        maxB = min(512, max(arr[1]) + MARGIN)
        if minB >= maxB:
            MARGIN = random.randint(0, 40)
            minB = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxB = min(512, max(arr[1]) + MARGIN)
        MARGIN = random.randint(-5, 40)
        minC = max(0, min(arr[2]) - MARGIN)
        MARGIN = random.randint(-5, 40)
        maxC = min(512, max(arr[2]) + MARGIN)
        if minC >= maxC:
            MARGIN = random.randint(0, 40)
            minC = max(0, min(arr[1]) - MARGIN)
            MARGIN = random.randint(0, 40)
            maxC = min(512, max(arr[1]) + MARGIN)

        if (maxA - minA) % 8 != 0:
            max_A = 8 * (int((maxA - minA) / 8) + 1)
            gap = int((max_A - (maxA - minA)) / 2)
            minA = max(0, minA - gap)
            maxA = min(len(mask), minA + max_A)
            if maxA == len(mask):
                minA = maxA - max_A

        if (maxB - minB) % 8 != 0 or (maxB - minB) <= 8:
            max_B = 8 * (int((maxB - minB) / 8) + 1)
            gap = int((max_B - (maxB - minB)) / 2)
            minB = max(0, minB - gap)
            maxB = min(512, minB + max_B)
            if maxB == 512:
                minB = maxB - max_B

        if (maxC - minC) % 8 != 0 or (maxC - minC) <= 8:
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
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )
            mask = (
                np.load(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )
                .transpose(2, 0, 1)
                .astype(np.float)
            )

        elif (
            "rmyy" in self.data_path[index].lower()
            or "renming" in self.data_path[index].lower()
        ):
            image = (
                nrrd.read(
                    join(self.data_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )
            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        elif "renji" in self.data_path[index].lower():
            image = nibabel.load(
                join(
                    self.data_path[index],
                    "{}".format(self.patient_index[index].replace(".nrrd", ".nii.gz")),
                )
            ).get_fdata()

            if len(image.shape) == 5:
                image = image[:, :, :, 0, 0]

            image = image.transpose((2, 1, 0)).astype(np.float)

            mask = (
                nrrd.read(
                    join(self.label_path[index], "{}".format(self.patient_index[index]))
                )[0]
                .transpose(2, 1, 0)
                .astype(np.float)
            )

        mask[mask != 2] = 0
        mask[mask == 2] = 1
        image, mask, _ = self.crop(image, mask)

        if np.max(image) > 1:
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        if mask.shape[-3] * mask.shape[-2] * mask.shape[-1] > self.size:
            # print("pool...")
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
            if random.random() < 0.3:
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

        image, mask = image.astype(np.float64), mask.astype(np.float64)
        tumor = mask.copy()

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

