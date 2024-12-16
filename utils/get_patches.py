import random
from os.path import join

import nibabel
import nrrd
import numpy as np
import torch
from skimage import morphology
from skimage.morphology import skeletonize_3d, skeletonize

from config.config import renji_panc_tumor_label_path, renji_data_path


# plt.imshow(mask[100], "gray")
# plt.show()

# skeleton_lee = skeletonize_3d(mask)
# print(skeleton_lee.shape)


# skeleton_lee = morphology.dilation(skeleton_lee, np.ones([5, 5, 2]))

# skeleton_lee[120:130] = np.zeros_like(skeleton_lee[120:130])
# plt.imshow(skeleton_lee[150], "gray")
# plt.show()

# nrrd.write("skeleton_lee.nrrd", skeleton_lee.transpose((2, 1, 0)))
# labels = measure.label(skeleton_lee[150])
# print(labels.max(), labels.min())

# arr = np.nonzero(skeleton_lee[150])
# random_select = np.random.randint(0, len(arr[0]))
# center_point = [arr[0][random_select], arr[1][random_select]]


def get_tumor_patch_from_skeleton_none(prob, mask, skeleton_lee, k=1, sum_pixel=5):
    assert skeleton_lee.shape == mask.shape
    assert len(prob.shape) == 5
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    skeleton_lee = torch.squeeze(skeleton_lee)
    if len(skeleton_lee.size()) == 2:
        skeleton_lee = torch.unsqueeze(skeleton_lee, dim=0)
    if len(skeleton_lee.size()) < 3 or len(mask.size()) < 3:
        return None
    pos_patch_list = []
    for sli in range(mask.size(0)):
        if mask[sli].sum() < sum_pixel or skeleton_lee[sli].sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee[sli])  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            # print(arr, skeleton_lee.size())
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(mask.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(mask.size(2) - 1, center_point[1] + k + 1)
            if mask[sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (2 * k + 1):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            pos_patch_list.append(prob[:, :, sli, minA:maxA, minB:maxB])
            break
    if len(pos_patch_list) == 0:
        return None
    random.shuffle(pos_patch_list)
    pos_patches = torch.cat(pos_patch_list, dim=0).cuda().float()
    return pos_patches


def get_pos_patch_from_skeleton_multi_batch(prob, mask, skeleton_lee, k=0, sum_pixel=5):
    assert skeleton_lee.shape == mask.shape
    assert len(prob.shape) == 5
    assert len(mask.shape) == 5

    pos_patch_list = []
    for bt in range(len(mask)):
        for sli in range(mask.size(-3)):
            if (
                    mask[bt, 0, sli].sum() < sum_pixel
                    or skeleton_lee[bt, 0, sli].sum() == 0
            ):
                continue
            arr = torch.nonzero(skeleton_lee[bt, 0, sli])  # torch.Size([18, 2])
            while arr.sum() > 1 and arr.size(-1) == 2:
                point = np.random.randint(0, arr.size(0))
                center_point = [arr[point, 0].item(), arr[point, 1].item()]
                minA = max(0, center_point[0] - k)
                maxA = min(mask.size(-2) - 1, center_point[0] + k + 1)
                minB = max(0, center_point[1] - k)
                maxB = min(mask.size(-1) - 1, center_point[1] + k + 1)
                if mask[bt, 0, sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (
                        2 * k + 1
                ):
                    arr[point, 0], arr[point, 1] = 0, 0
                    continue
                pos_patch_list.append(prob[bt: bt + 1, :, sli, minA:maxA, minB:maxB])
                break
    if len(pos_patch_list) == 0:
        return None
    random.shuffle(pos_patch_list)
    pos_patches = torch.cat(pos_patch_list, dim=0).cuda().float()
    return pos_patches


def get_neg_patch_from_skeleton_multi_batch(dilation, prob, mask, k=2, sum_pixel=5):
    assert len(prob.shape) == 5
    assert len(mask.shape) == 5

    error = (dilation - mask) * (1 - mask)
    neg_patch_list = []
    for bt in range(len(mask)):
        for sli in range(error.size(-3)):
            if error[bt, 0, sli].sum() < sum_pixel:
                continue
            skeleton_lee = torch.from_numpy(
                skeletonize(error[bt, 0, sli].cpu().numpy())
            ).float()
            if skeleton_lee.sum() == 0:
                continue
            arr = torch.nonzero(skeleton_lee)  # torch.Size([18, 2])
            while arr.sum() > 1 and arr.size(-1) == 2:
                point = np.random.randint(0, arr.size(0))
                center_point = [arr[point, 0].item(), arr[point, 1].item()]
                minA = max(0, center_point[0] - k)
                maxA = min(error.size(-2) - 1, center_point[0] + k + 1)
                minB = max(0, center_point[1] - k)
                maxB = min(error.size(-3) - 1, center_point[1] + k + 1)
                if (
                        mask[bt, 0, sli, minA:maxA, minB:maxB].sum() != 0
                        or (maxA - minA) != 2 * k + 1
                        or (maxB - minB) != 2 * k + 1
                ):
                    arr[point, 0], arr[point, 1] = 0, 0
                    continue
                neg_patch_list.append(prob[bt: bt + 1, :, sli, minA:maxA, minB:maxB])
                break
    if len(neg_patch_list) == 0:
        return None
    random.shuffle(neg_patch_list)
    neg_patches = torch.cat(neg_patch_list, dim=0).cuda().float()
    return neg_patches


def get_pos_patch_from_skeleton_none(prob, mask, skeleton_lee, k=2, sum_pixel=10):
    assert skeleton_lee.shape == mask.shape
    assert len(prob.shape) == 5

    mask = torch.squeeze(mask)
    skeleton_lee = torch.squeeze(skeleton_lee)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)

    if len(skeleton_lee.size()) == 2:
        skeleton_lee = torch.unsqueeze(skeleton_lee, dim=0)
    if len(skeleton_lee.size()) < 3 or len(mask.size()) < 3:
        return None
    pos_patch_list = []
    for sli in range(mask.size(0)):
        if mask[sli].sum() < sum_pixel or skeleton_lee[sli].sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee[sli])  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            # print(arr, skeleton_lee.size())
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(mask.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(mask.size(2) - 1, center_point[1] + k + 1)
            if mask[sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (2 * k + 1):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            pos_patch_list.append(prob[:, :, sli, minA:maxA, minB:maxB])
            break
    if len(pos_patch_list) == 0:
        return None
    random.shuffle(pos_patch_list)
    pos_patches = torch.cat(pos_patch_list, dim=0).cuda().float()
    return pos_patches


def get_neg_patch_from_skeleton_none(mask_dilation, prob, mask, k=2, sum_pixel=50):
    # assert prob.shape[:, 0] == mask_dilation.shape[:, 0]
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    mask_dilation = torch.squeeze(mask_dilation)
    if len(mask_dilation.size()) == 2:
        mask_dilation = torch.unsqueeze(mask_dilation, dim=0)

    if len(mask_dilation.size()) < 3 or len(mask.size()) < 3:
        return None

    error = (mask_dilation - mask) * (1 - mask)
    neg_patch_list = []

    for sli in range(error.size(0)):
        if error[sli].sum() < sum_pixel:
            continue
        skeleton_lee = torch.from_numpy(skeletonize(error[sli].cpu().numpy())).float()
        if skeleton_lee.sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee)  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(error.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(error.size(2) - 1, center_point[1] + k + 1)
            if (
                    mask[sli, minA:maxA, minB:maxB].sum() != 0
                    or (maxA - minA) != 2 * k + 1
                    or (maxB - minB) != 2 * k + 1
            ):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            neg_patch_list.append(prob[:, :, sli, minA:maxA, minB:maxB])
            break
    if len(neg_patch_list) == 0:
        return None
    random.shuffle(neg_patch_list)
    neg_patches = torch.cat(neg_patch_list, dim=0).cuda().float()
    return neg_patches


def get_pos_patch_from_skeleton(prob, mask, skeleton_lee, k=2, sum_pixel=100):
    assert skeleton_lee.shape == mask.shape
    assert len(prob.shape) == 5
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    skeleton_lee = torch.squeeze(skeleton_lee)
    if len(skeleton_lee.size()) == 2:
        skeleton_lee = torch.unsqueeze(skeleton_lee, dim=0)
    pos_patch_list = []
    for sli in range(mask.size(0)):
        if mask[sli].sum() < sum_pixel or skeleton_lee[sli].sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee[sli])  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(mask.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(mask.size(2) - 1, center_point[1] + k + 1)
            if mask[sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (2 * k + 1):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            pos_patch_list.append(prob[:, :, sli, minA:maxA, minB:maxB])
            break
    random.shuffle(pos_patch_list)
    pos_patches = torch.cat(pos_patch_list, dim=0).cuda().float()
    return pos_patches


def get_pos_patch_from_skeleton_dual(prob, mask, skeleton_lee, k=2, sum_pixel=100):
    assert skeleton_lee.shape == mask.shape
    assert len(prob.shape) == 5
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    skeleton_lee = torch.squeeze(skeleton_lee)
    if len(skeleton_lee.size()) == 2:
        skeleton_lee = torch.unsqueeze(skeleton_lee, dim=0)
    pos_patch_list = []
    for sli in range(mask.size(0)):
        if mask[sli].sum() < sum_pixel or skeleton_lee[sli].sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee[sli])  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(mask.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(mask.size(2) - 1, center_point[1] + k + 1)
            if mask[sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (2 * k + 1):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            pos_patch_list.append(prob[0:1, :, sli, minA:maxA, minB:maxB])
            break
    for sli in range(mask.size(0)):
        if mask[sli].sum() < sum_pixel or skeleton_lee[sli].sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee[sli])  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(mask.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(mask.size(2) - 1, center_point[1] + k + 1)
            if mask[sli, minA:maxA, minB:maxB].sum() != (2 * k + 1) * (2 * k + 1):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            pos_patch_list.append(prob[1:2, :, sli, minA:maxA, minB:maxB])
            break
    if len(pos_patch_list) == 0:
        return None
    random.shuffle(pos_patch_list)
    pos_patches = torch.cat(pos_patch_list, dim=0).cuda().float()
    return pos_patches


def get_neg_patch_from_skeleton(mask_dilation, prob, mask, k=2, sum_pixel=50):
    # assert prob.shape[:, 0] == mask_dilation.shape[:, 0]
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    mask_dilation = torch.squeeze(mask_dilation)
    if len(mask_dilation.size()) == 2:
        mask_dilation = torch.unsqueeze(mask_dilation, dim=0)
    error = (mask_dilation - mask) * (1 - mask)
    neg_patch_list = []
    for sli in range(error.size(0)):
        if error[sli].sum() < sum_pixel:
            continue
        skeleton_lee = torch.from_numpy(skeletonize(error[sli].cpu().numpy())).float()
        if skeleton_lee.sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee)  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(error.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(error.size(2) - 1, center_point[1] + k + 1)
            if (
                    mask[sli, minA:maxA, minB:maxB].sum() != 0
                    or (maxA - minA) != 2 * k + 1
                    or (maxB - minB) != 2 * k + 1
            ):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            neg_patch_list.append(prob[:, :, sli, minA:maxA, minB:maxB])
            break
    if len(neg_patch_list) == 0:
        return None
    random.shuffle(neg_patch_list)
    neg_patches = torch.cat(neg_patch_list, dim=0).cuda().float()
    return neg_patches


def get_neg_patch_from_skeleton_dual(mask_dilation, prob, mask, k=2, sum_pixel=50):
    # assert prob.shape[:, 0] == mask_dilation.shape[:, 0]
    mask = torch.squeeze(mask)
    if len(mask.size()) == 2:
        mask = torch.unsqueeze(mask, dim=0)
    mask_dilation = torch.squeeze(mask_dilation)
    if len(mask_dilation.size()) == 2:
        mask_dilation = torch.unsqueeze(mask_dilation, dim=0)

    error = (mask_dilation - mask) * (1 - mask)

    neg_patch_list = []
    for sli in range(error.size(0)):
        if error[sli].sum() < sum_pixel:
            continue
        skeleton_lee = torch.from_numpy(skeletonize(error[sli].cpu().numpy())).float()
        if skeleton_lee.sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee)  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(error.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(error.size(2) - 1, center_point[1] + k + 1)
            if (
                    mask[sli, minA:maxA, minB:maxB].sum() != 0
                    or (maxA - minA) != 2 * k + 1
                    or (maxB - minB) != 2 * k + 1
            ):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            neg_patch_list.append(prob[0:1, :, sli, minA:maxA, minB:maxB])
            break
    for sli in range(error.size(0)):
        if error[sli].sum() < sum_pixel:
            continue
        skeleton_lee = torch.from_numpy(skeletonize(error[sli].cpu().numpy())).float()
        if skeleton_lee.sum() == 0:
            continue
        arr = torch.nonzero(skeleton_lee)  # torch.Size([18, 2])
        while arr.sum() > 1 and arr.size(-1) == 2:
            point = np.random.randint(0, arr.size(0))
            center_point = [arr[point, 0].item(), arr[point, 1].item()]
            minA = max(0, center_point[0] - k)
            maxA = min(error.size(1) - 1, center_point[0] + k + 1)
            minB = max(0, center_point[1] - k)
            maxB = min(error.size(2) - 1, center_point[1] + k + 1)
            if (
                    mask[sli, minA:maxA, minB:maxB].sum() != 0
                    or (maxA - minA) != 2 * k + 1
                    or (maxB - minB) != 2 * k + 1
            ):
                arr[point, 0], arr[point, 1] = 0, 0
                continue
            neg_patch_list.append(prob[1:2, :, sli, minA:maxA, minB:maxB])
            break
    if len(neg_patch_list) == 0:
        return None
    random.shuffle(neg_patch_list)
    neg_patches = torch.cat(neg_patch_list, dim=0).cuda().float()
    return neg_patches


if __name__ == "__main__":
    image = (
        nibabel.load(join(renji_data_path, "{}.nii.gz".format(1)))
        .get_fdata()
        .transpose(2, 1, 0)
    )
    mask = (
        nrrd.read(join(renji_panc_tumor_label_path, "{}.nrrd".format(1)))[0]
        .transpose(2, 1, 0)
        .astype(np.uint8)
    )
    mask[mask > 0] = 1

    mask_dilation = morphology.dilation(mask, np.ones([5, 5, 2]))
    mask_dilation = torch.from_numpy(mask_dilation).float().cuda()

    # skeleton_lee = skeletonize(mask[160])
    # plt.imshow(skeleton_lee)
    # plt.show()

    skeleton_lee = skeletonize_3d(mask)
    skeleton_lee = torch.from_numpy(skeleton_lee).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()

    pos_patches = get_pos_patch_from_skeleton(
        torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0), mask, skeleton_lee
    )

    print(pos_patches.size(), pos_patches[20].sum())

    neg_patches = get_neg_patch_from_skeleton(
        torch.unsqueeze(torch.unsqueeze(mask_dilation, dim=0), dim=0),
        torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0),
        mask,
    )

    print(neg_patches.size(), neg_patches.sum())
