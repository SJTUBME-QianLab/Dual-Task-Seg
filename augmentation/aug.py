from augmentation.volumentations import (
    Compose,
    ElasticTransform,
    RandomRotate,
    RandomFlip,
    RandomGamma,
    RandomGaussianNoise,
    RandomScale,
    Resize,
)


def get_augmentation():
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
            # Normalize(always_apply=True),
            ElasticTransform((0, 0.25)),
            RandomRotate((-15, 15), (-15, 15), (-15, 15)),
            RandomFlip(0),
            RandomFlip(1),
            RandomFlip(2),
            # Transpose((1,0,2)), # only if patch.height = patch.width
            # RandomRotate90((0,1)),
            RandomGamma(),
            RandomGaussianNoise(),
        ],
        p=1,
    )
