import torch

from pathlib import Path
import numpy as np
from PIL import Image as im
import imageio.v3 as iio
import torch

def write_exr(exr_path, image):
    image = to_numpy(image).astype(np.float32)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    try:
        iio.imwrite(exr_path, image)
    except OSError:
        imageio.plugins.freeimage.download()
        iio.imwrite(exr_path, image, extension='.exr')


def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)

def linear_to_srgb(l):
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055*(l[~m]**(1.0/2.4))-0.055
    return s


def srgb_to_linear(s):
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m]+0.055)/1.055) ** 2.4
    return l

def to_srgb(image):
    image = to_numpy(image)
    if image.shape[2] == 4:
        image_alpha = image[:, :, 3:4]
        image = linear_to_srgb(image[:, :, 0:3])
        image = np.concatenate([image, image_alpha], axis=2)
    else:
        image = linear_to_srgb(image)
    return np.clip(image, 0, 1)

def write_jpg(jpg_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    rgb_im = im.fromarray(image).convert('RGB')
    rgb_im.save(jpg_path, format='JPEG', quality=95)
    
def read_image(image_path, is_srgb=None, remove_alpha=True):
    image_path = Path(image_path)
    image = iio.imread(image_path)
    image = np.atleast_3d(image)
    if remove_alpha and image.shape[2] == 4:
        image = image[:, :, 0:3]

    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_linear(image)

    return image

def write_image(image_path, image, is_srgb=None):
    image_path = Path(image_path)
    image = to_numpy(image)
    image = np.atleast_3d(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_srgb(image)

    if image_path.suffix == '.exr':
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)

    iio.imwrite(image_path, image)
