import itertools
import os

from utils.device import func_device

LOW_RANGE = -100
HIGH_RANGE = 240
DEVICE = func_device()

# NIH
nih_train_list = [i for i in range(1, 83)]
# MSD
msd_train_list = [i for i in range(1, 282)]
msd_mix_list = []
fold_0 = [1 + 4 * i for i in range(80) if (1 + 4 * i) < 282]
fold_1 = [2 + 4 * i for i in range(80) if (2 + 4 * i) < 282]
fold_2 = [3 + 4 * i for i in range(80) if (3 + 4 * i) < 282]
fold_3 = [4 + 4 * i for i in range(80) if (4 + 4 * i) < 282]

msd_mix_list.append(fold_0)
msd_mix_list.append(fold_1)
msd_mix_list.append(fold_2)
msd_mix_list.append(fold_3)

renji_mix_list = [
    [
        1,
        2,
        3,
        5,
        10,
        11,
        12,
        13,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        26,
        27,
        29,
        30,
        33,
        36,
        40,
        42,
        43,
        44,
    ],
    [
        46,
        47,
        49,
        51,
        52,
        54,
        55,
        56,
        57,
        59,
        60,
        61,
        62,
        63,
        65,
        66,
        68,
        69,
        71,
        73,
        75,
        76,
        79,
        80,
        81,
        84,
    ],
    [
        86,
        89,
        90,
        93,
        95,
        96,
        98,
        99,
        100,
        102,
        103,
        104,
        105,
        106,
        108,
        109,
        111,
        112,
        113,
        116,
        119,
        120,
        121,
        122,
        123,
        126,
    ],
    [
        128,
        129,
        130,
        131,
        132,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        147,
        149,
        151,
        153,
        154,
        157,
        158,
        159,
        161,
        162,
        163,
    ],
]
rmyy_mix_list = [
    [
        93,
        94,
        16,
        114,
        163,
        135,
        105,
        47,
        12,
        119,
        83,
        40,
        134,
        37,
        75,
        117,
        98,
        140,
        124,
        102,
        53,
        24,
        137,
        112,
        157,
        103,
        51,
    ],
    [
        164,
        153,
        130,
        22,
        123,
        121,
        89,
        118,
        131,
        106,
        126,
        95,
        161,
        111,
        11,
        81,
        23,
        108,
        138,
        78,
        45,
        109,
        96,
        141,
        92,
        77,
        74,
    ],
    [
        59,
        158,
        33,
        58,
        151,
        9,
        57,
        21,
        52,
        35,
        132,
        136,
        19,
        48,
        129,
        36,
        120,
        46,
        142,
        90,
        91,
        128,
        65,
        13,
        85,
        38,
        79,
    ],
    [
        99,
        146,
        32,
        39,
        18,
        10,
        3,
        143,
        55,
        63,
        152,
        17,
        113,
        27,
        7,
        150,
        44,
        97,
        156,
        104,
        29,
        100,
        115,
        25,
        56,
        60,
        43,
    ],
]

if DEVICE == "PC":
    nih_data_path = "G:/OrganSegRSTN_PyTorch-master/OrganSegRSTN_Multi/nih/images"
    nih_label_path = "G:/OrganSegRSTN_PyTorch-master/OrganSegRSTN_Multi/nih/labels"

    msd_resample_slice_path = (
        "F:/public_dataset/Task07_Pancreas/Task07_Pancreas_Re_Spacing"
    )
    msd_data_path = "F:/public_dataset/Task07_Pancreas/Task07_Pancreas/imagesTr"
    msd_label_path = "F:/public_dataset/Task07_Pancreas/Task07_Pancreas/labelsTr"
    msd_multi_label_path = (
        "F:/public_dataset/Task07_Pancreas/Task07_Pancreas/npy/label_multi"
    )
    msd_resample_data_path = (
        "F:/public_dataset/Task07_Pancreas/Task07_Pancreas_Re_Spacing/npy/data"
    )
    msd_resample_multi_label_path = (
        "F:/public_dataset/Task07_Pancreas/Task07_Pancreas_Re_Spacing/npy/multi_label"
    )

    renji_slice_path = r"F:\renji-data\pancreas_only\V"
    renji_label_path = r"F:\renji-data\pancreas_only\V\label"

    renji_data_path = r"F:\renji-data\pancreas_only\V\data"
    renji_panc_tumor_label_path = r"F:\renji-data\npy\panc_tumor_label"

    rmyy_data_path = (
        r"F:\Pancreas_cancer\CT-pancreas-cancer-renming\patient_npy_norm\nrrd\data"
    )
    rmyy_panc_tumor_label_path = (
        r"F:\Pancreas_cancer\CT-pancreas-cancer-renming\panc_tumor_label"
    )


elif DEVICE == "13000":

    nih_data_path = "/home/data/dirk/OrganSegRSTN/nih/images"
    nih_label_path = "/home/data/dirk/OrganSegRSTN/nih/labels"

    msd_data_path = "/home/dirk/pancreas/MSD/Task07_Pancreas/imagesTr"
    msd_label_path = "/home/dirk/pancreas/MSD/Task07_Pancreas/labelsTr"
    msd_multi_label_path = "/home/data/dirk/pancreas/MSD/Full/label_multi/"
    msd_resample_data_path = (
        "/home/data/dirk/pancreas/MSD/Task07_Pancreas_Re_Spacing/npy/data/"
    )
    msd_resample_multi_label_path = (
        "/home/data/dirk/pancreas/MSD/Task07_Pancreas_Re_Spacing/npy/multi_label/"
    )

    rmyy_data_path = "/home/data/dirk/pancreas/rmyy/data"
    rmyy_label_path = "/home/data/dirk/pancreas/rmyy/label"
    rmyy_tumor_label_path = "/home/data/dirk/pancreas/rmyy/tumor_label"
    rmyy_panc_tumor_label_path = "/home/data/dirk/pancreas/rmyy/panc_tumor_label"
    renji_data_path = "/home/data/dirk/pancreas/renji/data"
    renji_label_path = "/home/data/dirk/pancreas/renji/label"
    renji_tumor_label_path = "/home/data/dirk/pancreas/renji/tumor_label_filled_hole"
    renji_panc_tumor_label_path = "/home/data/dirk/pancreas/renji/panc_tumor_label"

    msd_resample_data_path = (
        "/home/dirk/pancreas/MSD/Task07_Pancreas_Re_Spacing/npy/data/"
    )
    msd_resample_multi_label_path = (
        "/home/dirk/pancreas/MSD/Task07_Pancreas_Re_Spacing/npy/multi_label/"
    )
    rmyy_data_path = "/home/dirk/pancreas/rmyy/data"
    rmyy_panc_tumor_label_path = "/home/dirk/pancreas/rmyy/panc_tumor_label"
    renji_data_path = "/home/dirk/pancreas/renji/data"
    renji_panc_tumor_label_path = "/home/dirk/pancreas/renji/panc_tumor_label"

elif DEVICE == "10886":
    nih_data_path = "../pancreas_data/nih/images"
    nih_label_path = "../pancreas_data/nih/labels"
    msd_resample_data_path = "../pancreas_data/msd_resized/data/"
    msd_resample_multi_label_path = "../pancreas_data/msd_resized/multi_label/"
    rmyy_data_path = "../pancreas_data/rmyy/data"
    rmyy_panc_tumor_label_path = "../pancreas_data/rmyy/panc_tumor_label"
    rmyy_nii_label_path = "../pancreas_data/rmyy/nii_label"
    renji_data_path = "../pancreas_data/renji/data"
    renji_panc_tumor_label_path = "../pancreas_data/renji/panc_tumor_label"


def get_tumor_data_mask_path(args):
    if args.data.lower() == "msd":
        data_path = msd_resample_data_path
        mask_path = msd_resample_multi_label_path
    elif args.data.lower() == "renji":
        data_path = renji_data_path
        mask_path = renji_panc_tumor_label_path
    elif args.data.lower() == "rmyy":
        data_path = rmyy_data_path
        mask_path = rmyy_panc_tumor_label_path
    else:
        data_path = None
        mask_path = None
    return data_path, mask_path


def get_test_list(args):
    if args.data.lower() == "nih":
        test_list = ["{:0>4}".format(i) for i in range(1, 83)]
    elif args.data.lower() == "msd":
        test_list = list(itertools.chain(*msd_mix_list))
    elif args.data.lower() == "renji":
        test_list = sorted(os.listdir(renji_panc_tumor_label_path))
        test_list = [int(i.replace(".nrrd", "")) for i in test_list]
        # test_list = list(itertools.chain(*renji_mix_list))
    elif args.data.lower() == "rmyy":
        # test_list = list(itertools.chain(*rmyy_mix_list))
        test_list = sorted(os.listdir(rmyy_panc_tumor_label_path))
        test_list = [int(i.replace(".nrrd", "")) for i in test_list]
    else:
        return None
    return test_list


def get_data_mask_path(args):
    if args.data.lower() == "nih":
        data_path = nih_data_path
        mask_path = nih_label_path
    elif args.data.lower() == "msd":
        data_path = msd_resample_data_path
        mask_path = msd_resample_multi_label_path
    elif args.data.lower() == "renji":
        data_path = renji_data_path
        mask_path = renji_panc_tumor_label_path
    elif args.data.lower() == "rmyy":
        data_path = rmyy_data_path
        mask_path = rmyy_panc_tumor_label_path
    else:
        data_path = None
        mask_path = None
    return data_path, mask_path



if __name__ == "__main__":
    print(len(list(itertools.chain(*nih_train_list))))
    temp_list = nih_train_list.copy()
    del temp_list[0]
    renji_list = list(itertools.chain(*temp_list))
    print(len(renji_list), renji_list)

