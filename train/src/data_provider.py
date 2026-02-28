"""
数据提供：从 meta 目录加载 jsonl 标注、按 task/宽高比分桶、构建 DataLoader。
"""
import logging
import time
import io
import torch
import glob
import math
import requests
import traceback
import copy
import numpy as np
import os
import random
import json
from io import BytesIO
from PIL import Image
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from accelerate.utils import set_seed
from .utils.log_utils import get_logger, log_once

logger = get_logger(__name__)

EMPTY_EMB_PATH = os.path.join(os.path.dirname(__file__), 'null_text_embedding.pt')

def _parse_data_weights(s: str | None) -> dict[str, float] | None:
    """解析 train_data_weights 字符串为归一化后的权重字典。"""
    if not s:
        return None
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    out_reweight = {task: weight / sum(out.values()) for task, weight in out.items()}
    log_once(logger, logging.INFO, "Parsed data weights:")
    for task, weight in sorted(out_reweight.items(), key=lambda x: x[1]):
        log_once(logger, logging.INFO, "  %s: %.03f", task[:50].ljust(50), weight)
    return out_reweight

def _get_bucket_key(line, data_name):
    """按 (data_name, source_ratios..., edit_ratio) 生成分桶 key。"""
    def _get_ratio(size, RATIO_STEP=0.1, RATIO_MIN=1.0/4, RATIO_MAX=4):
        ratio = min(max(RATIO_MIN, float(size['width'] / size['height'])), RATIO_MAX)
        ratio = round(ratio / RATIO_STEP) * RATIO_STEP
        return ratio
    buckets = [data_name]
    source_image_size = line.get('source_image_size', [])
    if source_image_size:
        buckets.extend([_get_ratio(size) for size in source_image_size])
    edit_image_size = line['edit_image_size']
    if edit_image_size:
        buckets.append(_get_ratio(edit_image_size))
    return tuple(buckets)


def _load_annos(data_root, seed, is_debug=False, max_frac=1.0):
    """从 data_root 下所有 jsonl 加载标注列表，并写入 bucket/task 信息。"""
    annos_list = []
    files = glob.glob(os.path.join(data_root, '*.jsonl'))
    logger.debug("load_annos seed: %s", seed)
    random.seed(seed)
    random.shuffle(files)
    data_name = os.path.basename(data_root)
    for anno_path in files[:int(len(files) * max_frac)]:
        with open(anno_path) as file:
            data_list = file.readlines()
        if is_debug:
            data_list = data_list[:100]
        for di, line in enumerate(data_list):
            try:
                line = json.loads(line)
                line['task'] = data_name
                line['bucket'] = _get_bucket_key(line, data_name)
                annos_list.append(line)
            except Exception as e:
                logger.warning("Err(load_annos) %s", e)
    return annos_list


class Task_InputCnt_AspectRatio_BucketBatchSampler(Sampler):
    """
    Sampler that buckets samples by aspect ratio (W / H) and yields batches where
    all samples belong to the same aspect-ratio bin.

    Args:
      annos: list of annotations
      batch_size: int
      data_weight: dict of data weights
      drop_last: whether to drop incomplete final batches per bucket
      global_rank: global rank
    """
    def __init__(self, annos: List[dict], batch_size: int, data_weight: dict, input_num_weights: dict, drop_last: bool = False):
        self.data_weight = data_weight
        self.input_num_weights = input_num_weights
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        # buckets: bin_index -> list of indices
        self.buckets = defaultdict(list)
        self.task_counts = defaultdict[Any, int](int)
        for i, anno in enumerate(annos):
            self.buckets[anno['bucket']].append(i)
            self.task_counts[anno['task']] += 1

    def __iter__(self):
        """
        For each bucket: shuffle indices, split into batches.
        Shuffle the list of bucket-batches across buckets before yielding.
        Yields lists of sample indices (DataLoader accepts those as batches).
        """
        global_step = 0
        while(True):
            bucket_batches = defaultdict(list)
            batch_size = self.batch_size
            for bucket_info, idxs in self.buckets.items():
                task_name, bucket_ratios = bucket_info[0], bucket_info[1:]
                source_image_cnt = len(bucket_ratios) - 1
                idxs_copy = idxs[:]  # local copy
                random.shuffle(idxs_copy)
                for i in range(0, len(idxs_copy), batch_size):
                    batch = idxs_copy[i:i + batch_size]
                    if len(batch) < batch_size:
                        if self.drop_last:
                            continue
                        else:
                            batch = (batch * batch_size)[:batch_size]
                    bucket_batches[(source_image_cnt, task_name)].append(batch)

            logger.debug("Bucket batch counts:")
            num_batchs = 0
            for bucket_info, batches in bucket_batches.items():
                random.shuffle(batches)
                num_batchs += len(batches)
                logger.debug("  bucket_info: %s, len(batches): %s", bucket_info, len(batches))

            task_idxs = defaultdict(int)
            for bi in range(num_batchs * 2):
                # Select the number of input images, and ensure this value is completely consistent across all processes.

                rng = random.Random(int(global_step % 1e8))
                input_num = rng.choices(list(self.input_num_weights.keys()), weights=list(self.input_num_weights.values()))[0]


                while(True):
                    #TODO: optimize this
                    task_name = random.choices(list(self.data_weight.keys()), weights=list(self.data_weight.values()))[0]
                    bucket_key = (input_num, task_name)
                    if bucket_key in bucket_batches:
                        break

                batches = bucket_batches.get(bucket_key)
                batch = batches[task_idxs[bucket_key] % len(batches)]
                task_idxs[bucket_key] = task_idxs[bucket_key] + 1
                batch = [(idx, global_step, bucket_key) for idx in batch]
                global_step += 1
                yield batch

    def __len__(self):
        """
        Number of batches (sum over buckets of floor(len(bucket)/batch_size))
        """
        total = 0
        for idxs in self.buckets.values():
            total += len(idxs) // self.batch_size
        return total * 100


class ImgDataset(Dataset):
    def __init__(
        self,
        annos,
        sampler,
        text_drop_ratio=0.05,
        enable_inverse=False,
        get_embedding=True,
        seed=None,
        is_debug=False,
        retry_times=5,
    ):
        self.annos = annos
        self.sampler = sampler

        self.enable_inverse = enable_inverse
        self.get_embedding = get_embedding
        self.retry_times = retry_times
        self.text_drop_ratio = text_drop_ratio
        self.length = len(self.annos)

        log_once(logger, logging.INFO, "Bucket summary (top 20):")
        for bin, vals in sorted(list(self.sampler.buckets.items()), key=lambda x: len(x[1]))[::-1][:20]:
            log_once(logger, logging.INFO, "  bin: %s, length: %s", bin, len(vals))
        log_once(logger, logging.INFO, "Task summary:")
        for task_name, cnt in self.sampler.task_counts.items():
            log_once(logger, logging.INFO, "  task_name: %s, length: %s", task_name, cnt)


    def __len__(self):
        return self.length

    def load_image(self, path):
        if path.startswith('http'):
            response = requests.get(path, timeout=10)
            image_data = BytesIO(response.content)
            return Image.open(image_data).convert('RGB')
        else:
            return Image.open(path).convert('RGB')


    def prepare(self, item):
        text, inverse_text, text_cn, inverse_text_cn = item['instruction'], item['inverse_instruction'], item['instruction_cn'], item['inverse_instruction_cn']
        edit_image_path = item['edit_image']
        source_image_paths = item.get('source_image', [])
        if source_image_paths is None:
            source_image_paths = []
        elif isinstance(source_image_paths, str):
            source_image_paths = [source_image_paths]
        else:
            source_image_paths = list(source_image_paths)

        text = text if text is not None else ''
        text_cn = text_cn if text_cn is not None else ''
        inverse_text = inverse_text if inverse_text is not None else ''
        inverse_text_cn = inverse_text_cn if inverse_text_cn is not None else ''

        text_candidates = []    # (text, lang, is_inverse)
        if text:
            text_candidates.append((text, 'eng', False))
        if text_cn:
            text_candidates.append((text_cn, 'cn', False))
        if self.enable_inverse and inverse_text:
            text_candidates.append((inverse_text, 'eng', True))
        if self.enable_inverse and inverse_text_cn:
            text_candidates.append((inverse_text_cn, 'cn', True))
        text, lang, is_inverse = random.choice(text_candidates)

        if self.get_embedding:
            emb_key_name = 'embeddings_tensor'
            if random.random() < self.text_drop_ratio:
                emb_key_name += '_droptext'
                text = ''
            else:
                if lang == 'eng':
                    emb_key_name += '_en'
                else:
                    emb_key_name += '_cn'
            
            if is_inverse:
                emb_key_name += '_inv'
            
            if text == '' and len(source_image_paths) == 0:
                embedding_path = EMPTY_EMB_PATH
            else:
                embedding_path = item[emb_key_name]
            embedding = torch.load(embedding_path)
        else:
            if random.random() < self.text_drop_ratio:
                text = ''
            embedding = None

        if is_inverse and len(source_image_paths) != 1:
            raise ValueError('The source image list must contain exactly one image when using inverse texts.')
        if is_inverse:
            edit_image_path, source_image_paths = source_image_paths[0], [edit_image_path]
        
        edit_image = self.load_image(edit_image_path)

        source_images = []
        for source_image_path in source_image_paths:
            if source_image_path:
                source_images.append(self.load_image(source_image_path))

        item_msg = f'task: {item['task']}, is_inverse: {is_inverse}, edit_image: {edit_image_path}'
        return {
            'edit_image': edit_image, 
            'source_images': source_images, 
            'text': text, 
            'encoder_hidden_states': embedding,
            'item_msg': item_msg,
        }


    def __getitem__(self, index_step):
        start = time.time()
        index, global_step, bucket_key = index_step
        retry = 0
        while(True):
            try:
                retry += 1
                item = copy.deepcopy(self.annos[index])
                info = self.prepare(item)
                info['global_step'] = global_step
                info['bucket_key'] = bucket_key
                # print('data', time.time() - start)
                return info
            except Exception as e:
                logger.warning("__getitem__ error: %s\n%s", e, traceback.format_exc())
                if retry < self.retry_times:
                    index = random.choice(self.sampler.buckets[item['bucket']])
                else:
                    index = random.choice(self.sampler.buckets[('t2i_0', 1.0)])


def _resize_by_short_size(image, target_size, seed=None):
    """按短边缩放到 target_size 并 RandomCrop。"""
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

def _random_drop(images, drop=0.0):
    if isinstance(images, list):
        return [_random_drop(image) for image in images]
    return Image.new('RGB', images.size) if random.uniform(0, 1) < drop else images
    
def _batch_crop_to_size(images, target_size, seed=None):
    """将多图按比例裁剪到统一面积（保持 mean ratio）。"""
    if not images:
        return []
    ratios = [image.size[0] / image.size[1] for image in images]
    ratio_mean = np.mean(ratios)
    # logger.debug("ratios: %s, ratio_mean: %s, target_size: %s", ratios, ratio_mean, target_size)
    width = math.sqrt(target_size * target_size * ratio_mean)
    height = width / ratio_mean
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return [_resize_by_short_size(image, (height, width), seed) for i, image in enumerate(images)]

def _to_tensor(images):
    to_tensor_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    images = [to_tensor_normalize(_) for _ in images]
    images = torch.stack(images) # (b c h w)
    images = images.to(memory_format=torch.contiguous_format).float()
    return images

def collate_fn(examples, image_sample_size, sync_text_encoder=False):
    size_vae = image_sample_size
    crop_seed = random.randint(0, 1000000) # NOTE: source & target randomcrop seed 一致

    edit_image = [item['edit_image'] for item in examples]
    source_images = [item["source_images"] for item in examples]
    text = [item['text'] for item in examples]
    item_msg = [item['item_msg'] for item in examples]
    global_step = [item['global_step'] for item in examples]
    bucket_key = [item['bucket_key'] for item in examples]

    edit_image = _batch_crop_to_size(edit_image, size_vae, seed=crop_seed)
    source_images_transposed = list(map(list, zip(*source_images))) # [ [img1, img2], [img3, img4] ] → [ [img1, img3], [img2, img4]] # NOTE: 确保batch里每个样本的第N张图shape一致，方便后续 vae encode
    source_images_transposed = [_batch_crop_to_size(source_image, size_vae, seed=crop_seed) for source_image in source_images_transposed]

    prompt_embeds = []
    max_seq_len = 0
    if not sync_text_encoder:
        for example in examples:
            prompt_embeds.append(example["encoder_hidden_states"])
            max_seq_len = max(max_seq_len, example["encoder_hidden_states"].size(0))
    
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long) for e in prompt_embeds]
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        padded_prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        encoder_hidden_states = padded_prompt_embeds
    else:
        encoder_hidden_states = None
        encoder_attention_mask = None

    results = {
        "pixel_values": _to_tensor(edit_image),
        "source_images_transposed": source_images_transposed,
        "text": text,
        "item_msg": item_msg,
        "global_step": global_step,
        "bucket_key": bucket_key,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    pixel_values_shape = results['pixel_values'].shape
    encoder_hidden_states_shape = results['encoder_hidden_states'].shape if results['encoder_hidden_states'] is not None else None
    logger.debug("pixel_values.shape: %s | org_source_ratio:  | encoder_hidden_states.shape: %s | bucket_key: %s | global_step: %s",
                 pixel_values_shape, encoder_hidden_states_shape, bucket_key, global_step)
    return results

def worker_init_fn(worker_id, base_seed):
    """DataLoader worker 进程的 RNG 初始化，保证多进程可复现。"""
    base_seed = base_seed * 256
    seed = base_seed + worker_id
    logger.debug("worker_init_fn worker_id=%s seed=%s", worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def data_provider_impl(extra_args, process_index, device):
    """
    根据 extra_args 与 process_index 构建训练 DataLoader（含 bucket sampler、collate、worker_init）。
    """
    log_once(logger, logging.INFO, "Init RNG with seed %s (process_index=%s).", extra_args.seed + process_index, process_index)

    seed = extra_args.seed + process_index

    ## load data as annos
    task_names = set([os.path.basename(_) for _ in glob.glob(os.path.join(extra_args.train_data_meta_dir, '*')) if os.path.isdir(_)])
    data_weight = _parse_data_weights(extra_args.train_data_weights)
    src_img_num_weights = _parse_data_weights(extra_args.train_src_img_num_weights)
    input_num_weights = {}
    for num, weight in src_img_num_weights.items():
        input_num_weights[int(num)] = weight

    if set(data_weight) - task_names:
        log_once(logger, logging.WARNING, "More weight keys than task dirs: %s", set(data_weight) - task_names)
    elif task_names - set(data_weight):
        log_once(logger, logging.WARNING, "Less weight keys than task dirs: %s", task_names - set(data_weight))

    task_names = task_names & set(data_weight.keys())
    log_once(logger, logging.INFO, "task_names: %s", task_names)

    data_paths = [os.path.join(extra_args.train_data_meta_dir, data_name) for data_name in task_names]
    annos = []
    log_once(logger, logging.INFO, "Loading annos from %s (seed=%s)...", extra_args.train_data_meta_dir, seed)
    load_fn = partial(_load_annos, seed=seed)
    with ThreadPoolExecutor(max_workers=32) as pool:
        for annos_ in pool.map(load_fn, data_paths):
            annos.extend(annos_)
    
    ## init sampler, dataset, and dataloader
    sampler = Task_InputCnt_AspectRatio_BucketBatchSampler(
        annos=annos,
        batch_size=extra_args.train_batch_size,
        data_weight=data_weight,
        input_num_weights=input_num_weights,
        drop_last=True
    )

    train_dataset = ImgDataset(
        annos=annos,
        sampler=sampler,
        enable_inverse=extra_args.enable_inverse,
        get_embedding=not extra_args.sync_text_encoder,
        seed=seed,
        is_debug=extra_args.dataloader_num_workers == 0
    )

    log_once(logger, logging.INFO, "Num examples = %s", len(train_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        # persistent_workers=True if extra_args.dataloader_num_workers != 0 else False,
        num_workers=extra_args.dataloader_num_workers,
        collate_fn=partial(collate_fn, image_sample_size=extra_args.image_sample_size, sync_text_encoder=extra_args.sync_text_encoder),
        worker_init_fn=partial(worker_init_fn, base_seed=extra_args.seed+process_index),
        prefetch_factor=extra_args.prefetch_factor if extra_args.dataloader_num_workers > 0 else None
    )
    return train_dataloader

if __name__ == "__main__":
    class TestArgs:
        """测试用的参数类"""
        def __init__(self):
            self.train_data_meta_dir = "/workspace/"
            self.train_data_weights = "\
group_photo_banana_manual_filtered_score_2_3062_rewrite_renamed=0.5,\
pico_banana_refined_sampled=1.2,\
t2i_0=1.0,\
"
            self.train_src_img_num_weights = "0=1,1=1,2=1,3=1"
            self.train_batch_size = 4
            self.seed = 1996
            self.prefetch_factor = 2
            self.dataloader_num_workers = 2
            self.enable_inverse = False
            self.image_sample_size = 512
            self.sync_text_encoder = False

    args = TestArgs()
    dataloader = data_provider_impl(args, 0, None)

    for step, batch in enumerate(dataloader):
        source_pixel_values_sizes = []
        for items in batch['source_images_transposed']:
            source_pixel_values_sizes.append([img.size for img in items])
        print(f"step: {step},  source_images_transposed.sizes: {source_pixel_values_sizes}")
        if step > 20:
            break
