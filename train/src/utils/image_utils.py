# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
import gc
import inspect
import math
import os
import shutil
import subprocess
import time
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import imageio
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image


# -----------------------------------------------------------------------------
# 宽高比与 condition 图像缩放（forward_step / extract_vlm_embeds 共用）
# -----------------------------------------------------------------------------

ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}

CONDITION_IMAGE_SIZE = 384 * 384


def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512) -> Tuple[List[float], float]:
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda r: abs(float(r) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


def calculate_dimensions(target_area: float, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def resize_source_images_for_condition(
    source_images_list: List[List[Image.Image]],
    ref_height: int,
    ref_width: int,
    image_sample_size: int,
    condition_image_size: int = CONDITION_IMAGE_SIZE,
) -> List[List[Image.Image]]:
    """
    对送入 text_encoder 的 source 图像做 Resize + CenterCrop，再缩放到 condition_image_size 对应尺寸。
    参考尺寸为 (ref_height, ref_width)。source_images_list: 每样本多张 source 图的 list of list of PIL。
    返回相同结构的 list of list of PIL Image。
    """
    if not source_images_list:
        return source_images_list
    return [
        apply_condition_transform_to_images(sample, ref_height, ref_width, image_sample_size, condition_image_size)
        for sample in source_images_list
    ]


def apply_condition_transform_to_images(
    images: List[Image.Image],
    ref_height: int,
    ref_width: int,
    image_sample_size: int,
    condition_image_size: int = CONDITION_IMAGE_SIZE,
) -> List[Image.Image]:
    """
    对多张图像应用统一的 Resize+CenterCrop+最终缩放。参考尺寸 (ref_height, ref_width)。
    返回与 images 等长的 list of PIL Image。
    """
    if not images:
        return []
    h, w = float(ref_height), float(ref_width)
    aspect_ratio_sample_size = {
        k: [x / 512 * image_sample_size for x in ASPECT_RATIO_512[k]]
        for k in ASPECT_RATIO_512
    }
    closest_size, _ = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
    closest_size = [int(x / 16) * 16 for x in closest_size]
    closest_size = list(map(int, closest_size))
    if closest_size[0] / h > closest_size[1] / w:
        resize_size = (closest_size[0], int(w * closest_size[0] / h))
    else:
        resize_size = (int(h * closest_size[1] / w), closest_size[1])

    source_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(closest_size),
    ])

    result = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        cropped = source_transform(im)
        width_c, height_c = cropped.size
        aspect_ratio = width_c / height_c
        new_width, new_height = calculate_dimensions(condition_image_size, aspect_ratio)
        result.append(cropped.resize((new_width, new_height), Image.Resampling.LANCZOS))
    return result


def load_and_resize_image_for_condition(
    src_image_paths: List[str],
    tgt_image_path: str,
    image_sample_size: int,
    condition_image_size: int = CONDITION_IMAGE_SIZE,
) -> Tuple[List[Image.Image], Image.Image, List[Dict], Dict]:
    """
    从路径加载 source/target 图像，做 Resize+CenterCrop 后缩放到 condition 尺寸。
    返回 (condition_src_images, condition_tgt_image, source_image_size, edit_image_size)。
    """
    src_images = [load_image(p) for p in src_image_paths]
    tgt_image = load_image(tgt_image_path)

    source_image_size = [{"width": im.size[0], "height": im.size[1]} for im in src_images]
    edit_image_size = {"width": tgt_image.size[0], "height": tgt_image.size[1]}

    ref_h, ref_w = tgt_image.size[1], tgt_image.size[0]
    condition_src_images = apply_condition_transform_to_images(
        src_images, ref_h, ref_w, image_sample_size, condition_image_size
    )
    condition_tgt_image = apply_condition_transform_to_images(
        [tgt_image], ref_h, ref_w, image_sample_size, condition_image_size
    )[0]

    return condition_src_images, condition_tgt_image, source_image_size, edit_image_size


def load_image(path: str) -> Image.Image:
    """支持本地路径或 http URL，返回 RGB PIL Image。"""
    if path.startswith('http'):
        response = requests.get(path, timeout=10)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# -----------------------------------------------------------------------------
# 数据加载与 crop / tensor（data_provider 共用）
# -----------------------------------------------------------------------------

def resize_by_short_size(
    image: Image.Image,
    target_size: Tuple[int, int],
    seed: int = None,
) -> Image.Image:
    """按短边缩放到 target_size 并 RandomCrop。target_size = (resolution_h, resolution_w)。"""
    resolution_h, resolution_w = target_size
    ppt_ratio = image.size[0] / image.size[1]
    if ppt_ratio > resolution_w / resolution_h:
        scale_ratio = resolution_h / image.size[1]
        image = image.resize((math.ceil(image.size[0] * scale_ratio), math.ceil(resolution_h)), Image.BICUBIC)
    else:
        scale_ratio = resolution_w / image.size[0]
        image = image.resize((math.ceil(resolution_w), math.ceil(image.size[1] * scale_ratio)), Image.BICUBIC)
    if seed is not None:
        saved_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
    image = transforms.RandomCrop((target_size[0], target_size[1]))(image)
    if seed is not None:
        torch.set_rng_state(saved_rng_state)
    return image


def batch_crop_to_size(
    images: List[Image.Image],
    target_size: int,
    seed: int = None,
) -> List[Image.Image]:
    """将多图按比例裁剪到统一面积（保持 mean ratio）。target_size 为边长（正方形）。"""
    if not images:
        return []
    ratios = [image.size[0] / image.size[1] for image in images]
    ratio_mean = np.mean(ratios)
    width = math.sqrt(target_size * target_size * ratio_mean)
    height = width / ratio_mean
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return [resize_by_short_size(image, (int(height), int(width)), seed) for image in images]


def images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """PIL 图像列表转为 (B, C, H, W) 张量，归一化到 [-1, 1]。"""
    to_tensor_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tensors = [to_tensor_normalize(im) for im in images]
    out = torch.stack(tensors)
    out = out.to(memory_format=torch.contiguous_format).float()
    return out


# -----------------------------------------------------------------------------
# 原有工具
# -----------------------------------------------------------------------------

def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def save_image(image: torch.Tensor, path: str, rescale=False):
    # image = image.squeeze(0)
    # image = rearrange(image, "c t h w -> t c h w")
    image = image.squeeze(0).squeeze(1)
    if rescale:
        image = (image + 1.0) / 2.0  # -1,1 -> 0,1
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


