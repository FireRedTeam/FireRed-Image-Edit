# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""
数据提供：从 meta 目录加载 jsonl 标注、按 task/宽高比分桶、构建 DataLoader。
"""
import logging
import time
import io
import torch
import glob
import math
import traceback
import copy
import numpy as np
import os
import random
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from accelerate.utils import set_seed
from .utils.log_utils import get_logger, log_once
from .utils.image_utils import load_image, resize_by_short_size, batch_crop_to_size, images_to_tensor


logger = get_logger(__name__)

EMPTY_EMB_PATH = os.getenv('EMPTY_EMB_PATH', os.path.join(os.path.dirname(__file__), 'null_text_embedding.pt'))

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
    """按 (data_name, source_ratios..., edit_ratio) 生成分桶 key，用于 aspect ratio 分桶。"""
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


def _load_annos(data_root, max_frac=1.0):
    """
    从 data_root 下所有 jsonl 加载标注列表，为每条写入 task / bucket 信息。
    多机多卡：files 先 sorted 再 shuffle(使用当前 RNG)，保证各进程若 seed 一致则 task 内 jsonl 顺序一致，便于分片可复现。
    """
    annos_list = []
    files = glob.glob(os.path.join(data_root, '*.jsonl'))
    files = sorted(files)  # 多机多卡：同一 task 内 jsonl 顺序一致，再 shuffle 才与 seed 一致
    random.shuffle(files)
    data_name = os.path.basename(data_root)
    for anno_path in files[:int(len(files) * max_frac)]:
        with open(anno_path) as file:
            data_list = file.readlines()
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
    按 (input_num, task) 与宽高比分桶的 batch sampler：同一 batch 内样本来自同一 aspect ratio 桶。
    注意：因分桶与 drop_last，每“轮”不会严格看完所有样本；__iter__ 为无限循环，训练端需按 step 数终止。

    参数:
      buckets: 各 (input_num, task_name, ...) 桶对应的样本索引列表（已按 rank 切分）
      task_counts: 各 task 的样本数（用于 __len__）
      batch_size / data_weight / input_num_weights: batch 大小与采样权重
      drop_last: 是否丢弃每个桶最后不足 batch_size 的 batch
    """
    def __init__(self, buckets: dict, task_counts: dict, batch_size: int, data_weight: dict, input_num_weights: dict, drop_last: bool = False):
        self.buckets = buckets
        self.task_counts = task_counts
        self.data_weight = data_weight
        self.input_num_weights = input_num_weights
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        """
        对每个桶 shuffle 后按 batch_size 组成 batch，再按 (input_num, task) 加权随机选桶逐批 yield。
        多卡：input_num 用 global_step 做 seed，保证同一步各卡一致；task 为各卡独立随机。
        """
        global_step = 0
        while(True):
            logger.info("iter over, re-shuffle bucket_batches ...")
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
            for bi in range(num_batchs):
                # 多卡一致：用 global_step 固定 RNG，保证各进程本 step 的 input_num 一致
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
        """每轮 batch 数（按 task_counts 与 drop_last 计算；多卡下各 rank 一致）。"""
        total = 0
        for task_name, cnt in self.task_counts.items():
            if self.drop_last:
                total += cnt // self.batch_size
            else:
                total += math.ceil(cnt / self.batch_size)
        return total


class TxtImgDataset(Dataset):
    """
    图文对数据集：按 sampler 给出的 (index, global_step, bucket_key) 从 annos 取条并 prepare（文本/embedding/图像）。
    支持 text_drop、inverse 指令、get_embedding 预计算；__getitem__ 内对单条失败有重试与 t2i 兜底。
    """
    def __init__(
        self,
        annos,
        buckets,
        batch_cnt,
        text_drop_ratio=0.05,
        enable_inverse=False,
        get_embedding=True,
        seed=None,
        retry_times=5,
    ):
        self.annos = annos
        self.buckets = buckets
        self.text_drop_ratio = text_drop_ratio

        self.enable_inverse = enable_inverse
        self.get_embedding = get_embedding
        self.retry_times = retry_times

        self.length = batch_cnt

    def __len__(self):
        return self.length

    def load_image(self, path):
        """支持本地路径或 http URL，返回 RGB PIL Image。"""
        return load_image(path)

    def prepare(self, item):
        """从一条 anno 解析 instruction/source/edit、可选 embedding，做 text_drop/inverse 等，返回模型输入 dict。"""
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
        """取一条样本；失败时重试同 bucket 随机样本，仍失败则回退到 t2i_0 桶（需存在该 bucket）。"""
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
                logger.debug("getitem time cost: %s", time.time() - start)
                return info
            except Exception as e:
                logger.warning("__getitem__ error: %s\n%s", e, traceback.format_exc())
                if retry < self.retry_times:
                    try:
                        index = random.choice(self.buckets[item['bucket']])
                    except (NameError, KeyError):
                        bucket = next(iter(self.buckets.keys()))
                        index = random.choice(self.buckets[bucket])
                else:
                    if ('t2i_0', 1.0) in self.buckets:
                        index = random.choice(self.buckets[('t2i_0', 1.0)])
                    else:
                        index = random.choice(next(iter(self.buckets.values())))


def collate_fn(examples, image_sample_size, condition_encoder_mode="offline"):
    """将一批样本拼成 batch：统一 crop 尺寸、padding prompt_embeds；source 按位置转置以便 batch 内每张 source 的 shape 一致。"""
    size_vae = image_sample_size
    crop_seed = random.randint(0, 1000000)  # source 与 target 的 RandomCrop 共用同一 seed，保证对应关系

    edit_image = [item['edit_image'] for item in examples]
    source_images = [item["source_images"] for item in examples]
    text = [item['text'] for item in examples]
    item_msg = [item['item_msg'] for item in examples]
    global_step = [item['global_step'] for item in examples]
    bucket_key = [item['bucket_key'] for item in examples]

    edit_image = batch_crop_to_size(edit_image, size_vae, seed=crop_seed)
    source_images_transposed = list(map(list, zip(*source_images)))  # [样本的图列表] → [按位置的图列表]，使 batch 内第 N 张 source 的 shape 一致便于 VAE
    source_images_transposed = [batch_crop_to_size(source_image, size_vae, seed=crop_seed) for source_image in source_images_transposed]

    prompt_embeds = []
    max_seq_len = 0
    if condition_encoder_mode == "offline":
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
        "pixel_values": images_to_tensor(edit_image),
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


def data_provider_impl(args, process_index, num_processes):
    """
    根据 args 与当前进程 rank/总数构建训练 DataLoader（bucket sampler、collate、worker_init）。
    多卡：对 buckets 按 rank 切分索引（见下方「分片」注释），保证数据不交、每轮步数一致。
    """
    log_once(logger, logging.INFO, "Init RNG with seed %s (process_index=%s).", args.seed + process_index, process_index)

    ## load data as annos
    set_seed(args.seed)
    task_names = set([os.path.basename(_) for _ in glob.glob(os.path.join(args.train_data_meta_dir, '*')) if os.path.isdir(_)])
    data_weight = _parse_data_weights(args.train_data_weights)
    src_img_num_weights = _parse_data_weights(args.train_src_img_num_weights)
    input_num_weights = {}
    for num, weight in src_img_num_weights.items():
        input_num_weights[int(num)] = weight

    if set(data_weight) - task_names:
        log_once(logger, logging.WARNING, "More weight keys than task dirs: %s", set(data_weight) - task_names)
    elif task_names - set(data_weight):
        log_once(logger, logging.WARNING, "Less weight keys than task dirs: %s", task_names - set(data_weight))

    task_names = task_names & set(data_weight.keys())
    log_once(logger, logging.INFO, "task_names: %s", task_names)
    # 对 task 排序，保证多机多卡时 data_paths 顺序一致（glob 顺序与文件系统相关）
    data_paths = sorted([os.path.join(args.train_data_meta_dir, data_name) for data_name in task_names])
    annos = []
    log_once(logger, logging.INFO, "Loading annos from %s ...", args.train_data_meta_dir)
    load_fn = partial(_load_annos, max_frac=1.0)
    with ThreadPoolExecutor(max_workers=32) as pool:
        for annos_ in tqdm(
            pool.map(load_fn, data_paths),
            total=len(data_paths),
            desc="Loading annos",
            unit="task",
            disable=(process_index != 0),
        ):
            annos.extend(annos_)

    buckets = defaultdict(list)
    for i, anno in enumerate(annos):
        buckets[anno['bucket']].append(i)

    # 按 rank 分片：每个 bucket 先截断为 num_processes 的整数倍，再取 arr[process_index::num_processes]，保证各卡数据不交且每卡样本数一致
    task_counts = defaultdict(int)
    for bucket_key in buckets.keys():
        arr = buckets[bucket_key]
        n_keep = (len(arr) // num_processes) * num_processes
        buckets[bucket_key] = arr[:n_keep][process_index::num_processes]
        task_counts[bucket_key[0]] += len(buckets[bucket_key])

    bucket_summary_info_str = "Bucket summary (top 20):\n"
    for bin, vals in sorted(list(buckets.items()), key=lambda x: len(x[1]))[::-1][:20]:
        bucket_summary_info_str += f"  bin: {bin}, length: {len(vals)}\n"
    bucket_summary_info_str += "Task summary:\n"
    for task_name, cnt in task_counts.items():
        bucket_summary_info_str += f"  task_name: {task_name}, length: {cnt}\n"
    logger.info(bucket_summary_info_str)
    
    # 各 rank 使用不同 seed，保证 shuffle 可复现且 worker 内 RNG 独立
    seed = args.seed + process_index

    sampler = Task_InputCnt_AspectRatio_BucketBatchSampler(
        buckets=buckets,
        task_counts=task_counts,
        batch_size=args.train_batch_size,
        data_weight=data_weight,
        input_num_weights=input_num_weights,
        drop_last=True,
    )

    train_dataset = TxtImgDataset(
        annos=annos,
        buckets=buckets,
        batch_cnt=len(sampler),
        enable_inverse=args.enable_inverse,
        get_embedding=args.condition_encoder_mode=="offline",
        seed=seed,
    )

    log_once(logger, logging.INFO, "Num batches = %s", len(train_dataset))

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=1, # batch_size is set in sampler
        batch_sampler=sampler,
        collate_fn=partial(collate_fn, image_sample_size=args.image_sample_size, condition_encoder_mode=args.condition_encoder_mode),
        # persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=partial(worker_init_fn, base_seed=args.seed+process_index),
        prefetch_factor=args.prefetch_factor if args.dataloader_num_workers > 0 else None
    )
    return train_dataloader

# if __name__ == "__main__":
#     class TestArgs:
#         """测试用的参数类"""
#         def __init__(self):
#             self.train_data_meta_dir = "/workspace/"
#             self.train_data_weights = "\
# group_photo_banana_manual_filtered_score_2_3062_rewrite_renamed=0.5,\
# pico_banana_refined_sampled=1.2,\
# t2i_0=1.0,\
# "
#             self.train_src_img_num_weights = "0=1,1=1,2=1,3=1"
#             self.train_batch_size = 4
#             self.seed = 1996
#             self.prefetch_factor = None
#             self.dataloader_num_workers = 1
#             self.enable_inverse = False
#             self.image_sample_size = 512
#             self.condition_encoder_mode = "offline"

#     args = TestArgs()
#     dataloader = data_provider_impl(args, 0, 2)

#     for step, batch in enumerate(dataloader):
#         source_pixel_values_sizes = []
#         for items in batch['source_images_transposed']:
#             source_pixel_values_sizes.append([img.size for img in items])
#         print(f"step: {step},  source_images_transposed.sizes: {source_pixel_values_sizes}")
#         if step > 9:
#             break

    
#     dataloader = data_provider_impl(args, 1, 2)

#     for step, batch in enumerate(dataloader):
#         source_pixel_values_sizes = []
#         for items in batch['source_images_transposed']:
#             source_pixel_values_sizes.append([img.size for img in items])
#         print(f"step: {step},  source_images_transposed.sizes: {source_pixel_values_sizes}")
#         if step > 9:
#             break