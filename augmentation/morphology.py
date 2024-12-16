
"""
Module for fast morphological operations for binary images
"""

__all__ = ["binary_dilation", "binary_erosion", "binary_closing", "binary_opening"]

import numpy as np


def binary_dilation(image, selem=None, out=None):
    """Apply a binary-dilation operation to an image using a structuring element

    This function removes pixels to objects' perimeter in the image using
    a structuring element.

    Args:
        image (array-like):
            Image data as an array. If the input is not numpy.bool array,
            the data is converted to this type.
        selem (array-like):
            Structuring element as an boolean image of the same dimension of `image`
        out (numpy.bool array):
            Array to store the result of this operation. The length of the array
            must be the same as the input image.

    Returns:
        numpy.bool array: Dilated image (when `out` is `None`)
    """
    dim = image.ndim
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if not image.dtype == np.bool:
        image = image.astype(np.bool)
    if selem is None:
        if dim == 1:
            selem = np.ones(shape=[3], dtype=np.bool)
        elif dim == 2:
            selem = np.zeros(shape=[3, 3], dtype=np.bool)
            selem[1, :] = True
            selem[:, 1] = True
        elif dim == 3:
            selem = np.zeros(shape=[3, 3, 3], dtype=np.bool)
            selem[:, 1, 1] = True
            selem[1, :, 1] = True
            selem[1, 1, :] = True
    else:
        if not isinstance(selem, np.ndarray):
            selem = np.asarray(selem, dtype=np.bool)
        if not selem.dtype == np.bool:
            selem = selem.astype(np.bool)
        if any([num_pixels % 2 == 0 for num_pixels in selem.shape]):
            raise ValueError(
                "Only structure element of odd dimension "
                "in each direction is supported."
            )
    perimeter_image = _get_perimeter_image(image)
    perimeter_coords = np.where(perimeter_image)
    if out is None:
        return_out = True
        out = image.copy()
    else:
        return_out = False
        out[:] = image[:]

    if dim == 1:
        sx = selem.shape[0]
        rx = sx // 2
        lx = image.shape[0]
        for ix in perimeter_coords[0]:
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            out[jx_b:jx_e] |= selem[kx_b:kx_e]

    if dim == 2:
        rx, ry = [n // 2 for n in selem.shape]
        lx = image.shape
        sx, sy = selem.shape
        lx, ly = image.shape
        for ix, iy in zip(perimeter_coords[0], perimeter_coords[1]):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            out[jx_b:jx_e, jy_b:jy_e] |= selem[kx_b:kx_e, ky_b:ky_e]

    if dim == 3:
        rx, ry, rz = [n // 2 for n in selem.shape]
        sx, sy, sz = selem.shape
        lx, ly, lz = image.shape
        for ix, iy, iz in zip(
            perimeter_coords[0], perimeter_coords[1], perimeter_coords[2]
        ):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            (jz_b, jz_e), (kz_b, kz_e) = _generate_array_indices(iz, rz, sz, lz)
            out[jx_b:jx_e, jy_b:jy_e, jz_b:jz_e] |= selem[
                kx_b:kx_e, ky_b:ky_e, kz_b:kz_e
            ]

    if return_out:
        return out


def binary_erosion(image, selem=None, out=None):
    """Apply a binary-erosion operation to an image using a structuring element

    This function removes pixels around objects' perimeter in an image and returns
    the result as an image.
    See the `binary_dilation` function doc-string for the arguments and retuned value.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if not image.dtype == np.bool:
        image = image.astype(np.bool)

    out_image = binary_dilation(~image, selem, out)

    if out is None:
        return ~out_image
    else:
        out[:] = ~out[:]


def binary_closing(image, selem=None, out=None):
    """Apply a binary-closing operation to an image using a structuring element

    This function dilates an image and then erodes the dilation result.
    See the `binary_dilation` function doc-string for the arguments and retuned value.
    """
    out_image = binary_erosion(binary_dilation(image, selem), selem, out)
    if out is None:
        return out_image


def binary_opening(image, selem=None, out=None):
    """Apply a binary-opening operation to an image using a structuring element

    This function erodes an image and then dilates the eroded result.
    See the `binary_dilation` function doc-string for arguments and retuned value.
    """
    out_image = binary_dilation(binary_erosion(image, selem), selem, out)
    if out is None:
        return out_image


def _get_perimeter_image(image):
    """Return the image of the perimeter structure of the input image

    Args:
        image (Numpy array): Image data as an array

    Returns:
        Numpy array: Perimeter image
    """
    dim = image.ndim
    if dim > 3:
        raise RuntimeError("Binary image in 4D or above is not supported.")
    count = np.zeros_like(image, dtype=np.uint8)
    inner = np.zeros_like(image, dtype=np.bool)

    count[1:] += image[:-1]
    count[:-1] += image[1:]

    if dim == 1:
        inner |= image == 2
        for i in [0, -1]:
            inner[i] |= count[i] == 1
        return image & (~inner)

    count[:, 1:] += image[:, :-1]
    count[:, :-1] += image[:, 1:]
    if dim == 2:
        inner |= count == 4
        for i in [0, -1]:
            inner[i] |= count[i] == 3
            inner[:, i] |= count[:, i] == 3
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 2
        return image & (~inner)

    count[:, :, 1:] += image[:, :, :-1]
    count[:, :, :-1] += image[:, :, 1:]

    if dim == 3:
        inner |= count == 6
        for i in [0, -1]:
            inner[i] |= count[i] == 5
            inner[:, i] |= count[:, i] == 5
            inner[:, :, i] |= count[:, :, i] == 5
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    inner[i, j, k] |= count[i, j, k] == 3
        return image & (~inner)
    raise RuntimeError("This line should not be reached.")


def _generate_array_indices(selem_center, selem_radius, selem_length, result_length):
    """Return the correct indices for slicing considering near-edge regions

    Args:
        selem_center (int): The index of the structuring element's center
        selem_radius (int): The radius of the structuring element
        selem_length (int): The length of the structuring element
        result_length (int): The length of the operating image

    Returns:
        (int, int): The range begin and end indices for the operating image
        (int, int): The range begin and end indices for the structuring element image
    """
    # First index for the result array
    result_begin = selem_center - selem_radius
    # Last index for the result array
    result_end = selem_center + selem_radius + 1
    # First index for the structuring element array
    selem_begin = -result_begin if result_begin < 0 else 0
    result_begin = max(0, result_begin)
    # Last index for the structuring element array
    selem_end = (
        selem_length - (result_end - result_length)
        if result_end > result_length
        else selem_length
    )
    return (result_begin, result_end), (selem_begin, selem_end)
