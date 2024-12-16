
import copy
import itertools
import os
import random
import sys
import time

import nibabel
import nrrd
import numpy as np
import torch
import yaml
from skimage import measure
from skimage.transform import resize

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from models.Upsample import UpsampleDeterministicP3D

from config.config import HIGH_RANGE, LOW_RANGE
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from medpy.metric.binary import __surface_distances
join = os.path.join
exists = os.path.exists


class IOStream:
    def __init__(self, path, log=True):
        self.f = open(path, "a")
        self.log = log

    def pwrite(self, text):
        print(text)
        if self.log:
            self.f.write(text + "\n")
            self.f.flush()

    def write(self, text):
        if self.log:
            self.f.write(text + "\n")
            self.f.flush()
        else:
            print("log is False!")

    def close(self):
        self.f.close()

def crop(pred, label, margin_z=0, margin_xy=0):
    arr = np.nonzero(pred)
    minA = max(0, min(arr[0]) - margin_z)
    maxA = min(len(label), max(arr[0]) + margin_z)
    minB = max(0, min(arr[-2]) - margin_xy)
    maxB = min(label.shape[1], max(arr[1]) + margin_xy)
    minC = max(0, min(arr[2]) - margin_xy)
    maxC = min(label.shape[-1], max(arr[2]) + margin_xy)
    label = label[minA:maxA, minB:maxB, minC:maxC]
    return label


def cal_cover(pred, mask, margin_z=0, margin_xy=0):
    if pred.sum() == 0:
        return 0
    label_cropped = crop(pred, mask, margin_z=margin_z, margin_xy=margin_xy)
    ratio = np.sum(label_cropped) / np.sum(mask)
    return ratio


def crop_by_pancreas(image, mask, pancreas, MARGIN=25, MARGIN_Z=5):

    arr = np.nonzero(pancreas)
    minA = max(0, min(arr[0]) - MARGIN_Z)
    maxA = min(len(mask), max(arr[0]) + MARGIN_Z)

    minB = max(0, min(arr[1]) - MARGIN)
    maxB = min(512, max(arr[1]) + MARGIN)
    minC = max(0, min(arr[2]) - MARGIN)
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
        image[minA:maxA, minB:maxB, minC:maxC].copy(),
        mask[minA:maxA, minB:maxB, minC:maxC].copy(),
    )

    bbox = [minA, maxA, minB, maxB, minC, maxC]

    return image, mask, bbox


def unpadding_z(pred, label):
    slice_label = len(label)
    slice_pred = len(pred)

    new_pred = np.zeros_like(label)

    if slice_pred >= slice_label:
        up = int((slice_pred - slice_label) / 2)
        new_pred[:] = pred[up : up + slice_label]
        return new_pred
    else:
        up = int((slice_label - slice_pred) / 2)
        new_pred[up : up + slice_pred] = pred
        return new_pred


def center_uncrop(pred, label):
    assert (
        label.shape[1] >= pred.shape[1]
    ), "Error! The pred-height should be smaller than label!"
    assert (
        label.shape[2] >= pred.shape[2]
    ), "Error! The pred-width should be smaller than label!"

    height, width = pred.shape[1], pred.shape[2]
    height_label, width_label = label.shape[1], label.shape[2]

    s_h = int((height_label - height) / 2)
    s_w = int((width_label - width) / 2)

    new_pred = np.zeros_like(label)
    new_pred[:, s_h : s_h + height, s_w : s_w + width] = pred

    return new_pred


def pred_result_ensemble_mean(tests, args):
    if hasattr(args, "model"):
        model = args.model
    elif hasattr(args, "fine_model"):
        model = args.fine_model
    Is_Flip = [0, 1, 0, 1, 0, 1]
    Is_Flip_LIST = list(set(list(itertools.permutations(Is_Flip, 3))))

    with torch.no_grad():
        pred_list = []

        for flips in Is_Flip_LIST:
            test = tests.copy()
            if flips[0] == 1:
                test = np.flip(test, axis=0)
            if flips[1] == 1:
                test = np.flip(test, axis=1)
            if flips[2] == 1:
                test = np.flip(test, axis=2)
            test = np.expand_dims(test, axis=0)
            test = np.expand_dims(test, axis=0)
            test = np.ascontiguousarray(test)
            test = torch.from_numpy(test)
            test = test.float().cuda()
            pred = list(model(test))[0]

            if len(pred.shape) < 5:
                pred = torch.unsqueeze(pred, dim=0)

            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)

            if flips[0] == 1:
                pred = np.flip(pred, axis=0)
            if flips[1] == 1:
                pred = np.flip(pred, axis=1)
            if flips[2] == 1:
                pred = np.flip(pred, axis=2)

            pred = np.ascontiguousarray(pred)
            pred_list.append(pred)

    pred = pred_list[0]

    for i in range(1, len(pred_list)):
        pred += pred_list[i]

    pred /= len(pred_list)

    return pred

def pred_result_ensemble_mean_res(tests, args):
    if hasattr(args, "model"):
        model = args.model
    elif hasattr(args, "fine_model"):
        model = args.fine_model
    Is_Flip = [0, 1, 0, 1, 0, 1]
    Is_Flip_LIST = list(set(list(itertools.permutations(Is_Flip, 3))))

    with torch.no_grad():
        pred_list = []

        for flips in Is_Flip_LIST:
            test = tests.copy()
            if flips[0] == 1:
                test = np.flip(test, axis=0)
            if flips[1] == 1:
                test = np.flip(test, axis=1)
            if flips[2] == 1:
                test = np.flip(test, axis=2)
            test = np.array([test])

            images = test
            # image -> res
            image_copy = np.zeros_like(images).astype(np.float32)
            image_copy[:, 1:, :, :] = images[:, 0 : images.shape[-3] - 1, :, :]
            image_res = images - image_copy
            image_res[:, 0, :, :] = 0
            image_res = np.abs(image_res)
            image_res_ = resize(
                image_res[0],
                (
                    int(image_res.shape[-3] / 4),
                    int(image_res.shape[-2] / 4),
                    int(image_res.shape[-1] / 4),
                ),
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                preserve_range=True,
            )
            image_res_ = np.array([image_res_])
            image_res = np.array([image_res_]).astype(np.float32)
            test = np.array([test])

            test = torch.from_numpy(test).float().cuda()
            image_res = torch.from_numpy(image_res).float().cuda()
            pred = list(model(test, image_res))[0]

            if len(pred.shape) < 5:
                pred = torch.unsqueeze(pred, dim=0)

            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)

            if flips[0] == 1:
                pred = np.flip(pred, axis=0)
            if flips[1] == 1:
                pred = np.flip(pred, axis=1)
            if flips[2] == 1:
                pred = np.flip(pred, axis=2)

            pred = np.ascontiguousarray(pred)
            pred_list.append(pred)

    pred = pred_list[0]

    for i in range(1, len(pred_list)):
        pred += pred_list[i]

    pred /= len(pred_list)

    return pred


def pred_result_ensemble_mean_up(tests, args):
    if hasattr(args, "model"):
        model = args.model
    elif hasattr(args, "coarse_model"):
        model = args.coarse_model
    Is_Flip = [0, 1, 0, 1, 0, 1]
    Is_Flip_LIST = list(set(list(itertools.permutations(Is_Flip, 3))))

    with torch.no_grad():
        pred_list = []
        pred_pool_list = []

        for flips in Is_Flip_LIST:
            test = tests.copy()
            if flips[0] == 1:
                test = np.flip(test, axis=0)
            if flips[1] == 1:
                test = np.flip(test, axis=1)
            if flips[2] == 1:
                test = np.flip(test, axis=2)
            test = np.expand_dims(test, axis=0)
            test = np.expand_dims(test, axis=0)
            test = np.ascontiguousarray(test)
            test = torch.from_numpy(test)
            test = test.float().cuda()
            pred_pool = list(model(test))[0]

            if len(pred_pool.shape) < 5:
                pred_pool = torch.unsqueeze(pred_pool, dim=0)

            pred = UpsampleDeterministicP3D(upscale=2)(pred_pool)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()
            pred_pool = torch.sigmoid(pred_pool)
            pred_pool = pred_pool.cpu().detach().numpy()
            pred = np.squeeze(pred)
            pred_pool = np.squeeze(pred_pool)

            if flips[0] == 1:
                pred = np.flip(pred, axis=0)
                pred_pool = np.flip(pred_pool, axis=0)
            if flips[1] == 1:
                pred = np.flip(pred, axis=1)
                pred_pool = np.flip(pred_pool, axis=1)
            if flips[2] == 1:
                pred = np.flip(pred, axis=2)
                pred_pool = np.flip(pred_pool, axis=2)

            pred = np.ascontiguousarray(pred)
            pred_pool = np.ascontiguousarray(pred_pool)

            pred_list.append(pred)
            pred_pool_list.append(pred_pool)

    pred = pred_list[0]
    pred_pool = pred_pool_list[0]

    for i in range(1, len(pred_list)):
        pred += pred_list[i]
        pred_pool += pred_pool_list[i]

    pred /= len(pred_list)
    pred_pool /= len(pred_pool_list)

    pred = pred >= args.thresh
    pred_pool = pred_pool >= args.thresh

    return pred, pred_pool


def save_pred_coarse(args, index, pred):

    if not os.path.exists(
        join(args.path, "pred_thresh_{}_ensemble_{}".format(args.thresh, args.ensemble))
    ):
        os.mkdir(
            join(
                args.path,
                "pred_thresh_{}_ensemble_{}".format(args.thresh, args.ensemble),
            )
        )

    if args.data.lower() in ["nih", "isicdm", "msd"]:
        pred = np.flip(pred, axis=1)
        pred = np.flip(pred, axis=0)
        pred = pred.transpose((1, 2, 0))
    else:
        pred = pred.transpose((2, 1, 0))

    nrrd.write(
        os.path.join(
            args.path,
            "pred_thresh_%s_ensemble_%s" % (args.thresh, args.ensemble),
            "{}.nrrd".format(index),
        ),
        pred.astype(np.uint8),
    )


def save_pred_fine(args, index, pred):

    if not os.path.exists(
        join(
            args.path,
            "thresh_%s_ensemble_%s_based_%s_%s"
            % (args.thresh, args.ensemble, args.coarse_thresh, args.coarse_ensemble),
        )
    ):
        os.mkdir(
            join(
                args.path,
                "thresh_%s_ensemble_%s_based_%s_%s"
                % (
                    args.thresh,
                    args.ensemble,
                    args.coarse_thresh,
                    args.coarse_ensemble,
                ),
            )
        )

    if args.data.lower() in ["nih", "isicdm", "msd"]:
        pred = np.flip(pred, axis=1)
        pred = np.flip(pred, axis=0)
        pred = pred.transpose((1, 2, 0))
    else:
        pred = pred.transpose((2, 1, 0))

    nrrd.write(
        os.path.join(
            args.path,
            "thresh_%s_ensemble_%s_based_%s_%s"
            % (args.thresh, args.ensemble, args.coarse_thresh, args.coarse_ensemble),
            "{}.nrrd".format(index),
        ),
        pred.astype(np.uint8),
    )


def get_data_mask(args, index):
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
    else:
        image, mask = None, None

    if np.max(image) > 1:
        np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
        image -= LOW_RANGE
        image /= HIGH_RANGE - LOW_RANGE

    return image, mask


def center_crop(data, height, width):
    assert (
        data.shape[1] >= height
    ), "Error! The height %s should be smaller than data %s!" % (height, data.shape)
    assert (
        data.shape[2] >= width
    ), "Error! The width %s should be smaller than data %s!" % (height, data.shape)

    height_data, width_data = data.shape[1], data.shape[2]

    s_h = int((height_data - height) / 2)
    s_w = int((width_data - width) / 2)

    return data[:, s_h : s_h + height, s_w : s_w + width]


def padding_z(data, label=None, mode="down"):
    if mode == "down":
        if label is not None:
            slices = len(data)
            new_slice = int(slices / 16) * 16
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            new_label = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                new_label[up : up + slices] = label
                return new_data, new_label
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                new_label = label[up : up + new_slice]
                return new_data, new_label
        else:
            slices = len(data)
            new_slice = int(slices / 16) * 16
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                return new_data
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                return new_data

    elif mode == "up":
        if label is not None:
            slices = len(data)
            new_slice = int(slices / 16) * 16 + 16
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            new_label = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                new_label[up : up + slices] = label
                return new_data, new_label
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                new_label = label[up : up + new_slice]
                return new_data, new_label
        else:
            slices = len(data)
            new_slice = int(slices / 16) * 16 + 16
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                return new_data
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                return new_data


def adjust_size(z, h, w):
    if z % 8 != 0:
        z = int(z / 8) * 8

    if h != 0:
        h = int(h / 8) * 8

    if w != 0:
        w = int(w / 8) * 8

    return z, h, w


def cal_recall(pred, label):
    return np.logical_and(pred, label).sum() / label.sum()


def cal_precision(pred, label):
    return np.logical_and(pred, label).sum() / pred.sum()


def normalize(new_img):
    Min = np.min(new_img)
    Max = np.max(new_img)
    new_img = (new_img - Min) / (Max - Min)
    return new_img


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_iter, power=0.9):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]["lr"] = lr


def get_time():
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    return cur_time


def save_arg(args, path):
    args_temp = copy.deepcopy(args)
    if hasattr(args_temp, "model"):
        args_temp.model = None
    if hasattr(args_temp, "seg_model"):
        args_temp.seg_model = None
    if hasattr(args_temp, "AE_models"):
        args_temp.AE_models = None
    if hasattr(args_temp, "coarse_model"):
        args_temp.coarse_model = None
    if hasattr(args_temp, "fine_model"):
        args_temp.fine_model = None
    if hasattr(args_temp, "test_list"):
        args_temp.test_list = list(args_temp.test_list)
    arg_dict = vars(args_temp)
    with open(path, "a", encoding="utf-8") as f:
        yaml.dump(arg_dict, f, encoding="utf-8", allow_unicode=True)


def seed_torch(seed=2022):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_processing(labels, r=0.5):
    labels = measure.label(labels, connectivity=3)

    max_num = 0
    max_pixel = 1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j

    # If only the largest volume is to be retained, the following three lines can be commented out.
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > r * np.sum(labels == max_pixel):
            labels[labels == j] = max_pixel

    labels[labels != max_pixel] = 0
    labels[labels == max_pixel] = 1

    return labels


def compute_sdf(img_gt, out_shape, a, b):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode="inner").astype(
                    np.uint8
                )
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (
                    posdis - np.min(posdis)
                ) / (np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf
                if np.min(sdf) != -1.0 or np.max(sdf) != 1.0:
                    # print("Error in sdf")
                    return np.zeros(out_shape)
                assert np.min(sdf) == -1.0, print(
                    "np.min(sdf) != -1.0",
                    a,
                    b,
                    np.min(posdis),
                    np.max(posdis),
                    np.min(negdis),
                    np.max(negdis),
                )
                assert np.max(sdf) == 1.0, print(
                    "np.max(sdf) != 1.0",
                    a,
                    b,
                    np.min(posdis),
                    np.min(negdis),
                    np.max(posdis),
                    np.max(negdis),
                )

    return normalized_sdf


def normalized_surface_dice(b: np.ndarray, a: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc


