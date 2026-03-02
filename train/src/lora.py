"""
LoRA 训练脚本。

与 SFT 相同的流程（数据、前向、优化、checkpoint），区别为：
- 在 model_provider 加载完 Transformer 后注入 LoRA，仅训练 LoRA 参数；
- 保存/恢复时只读写 LoRA adapter 权重。
建议使用 accelerate 且不使用 FSDP 进行 LoRA 训练。
"""
import argparse
import gc
import logging
import math
import os
import pickle
import random
import json
import shutil
import sys
from functools import partial

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from datasets import IterableDataset as HFIterableDataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.parallelism_config import ParallelismConfig
from accelerate.logging import get_logger as accelerate_get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DataLoaderConfiguration
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from glob import glob

from .utils.other import linear_decay
from .utils.image_utils import save_image
from .utils.log_utils import get_logger, DistributedColoredFormatter, get_dist_prefix, get_default_log_level, log_once


# 各 impl_module 默认的 LoRA target 模块名（若未通过 --lora_target_modules 指定）
DEFAULT_LORA_TARGET_MODULES = {
    "qwen_image": ["to_q", "to_v", "to_k", "to_out.0"],
    "z_image": ["to_q", "to_v", "to_k", "to_out.0"],
}


def _get_lora_target_modules(args):
    if getattr(args, "lora_target_modules", None) is not None and len(args.lora_target_modules) > 0:
        return args.lora_target_modules
    return DEFAULT_LORA_TARGET_MODULES.get(args.impl_module, ["to_q", "to_v"])


def lora_train(
    data_provider_func,
    model_provider_func,
    forward_step,
    args,
):
    # ===================== Accelerator 初始化 =====================
    logger = get_logger("REDEdit.lora")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False)
    )

    # 配置根 logger
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DistributedColoredFormatter(dist_prefix=get_dist_prefix()))
    root.addHandler(handler)
    root.setLevel(get_default_log_level())

    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if fsdp_plugin is not None:
        log_once(logger, logging.WARNING, "LoRA 训练建议不使用 FSDP；当前已启用 FSDP，保存时将仅保存 adapter 权重。")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    _accel_logger = accelerate_get_logger(__name__, log_level="INFO")
    _accel_logger.info(str(accelerator.state), main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.seed is not None:
        set_seed(args.seed)

    # ===================== 模型初始化（与 SFT 一致） =====================
    log_once(logger, logging.INFO, "Loading model via model_provider...")
    transformer3d, text_encoder, vae, extra_modules = model_provider_func(args, weight_dtype, accelerator.device)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        log_once(logger, logging.INFO, "Gradient checkpointing enabled.")

    # ===================== 注入 LoRA =====================
    lora_r = getattr(args, "lora_r", 0)
    if lora_r <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")

    from peft import get_peft_model, LoraConfig, TaskType

    lora_alpha = getattr(args, "lora_alpha", None) or lora_r
    lora_dropout = getattr(args, "lora_dropout", 0.0)
    target_modules = _get_lora_target_modules(args)

    # 先冻结整个 Transformer，再注入 LoRA
    transformer3d.requires_grad_(False)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    transformer3d = get_peft_model(transformer3d, lora_config)
    log_once(logger, logging.INFO, "LoRA injected: r=%s, alpha=%s, target_modules=%s", lora_r, lora_alpha, target_modules)
    transformer3d.print_trainable_parameters()

    # ===================== 数据加载 =====================
    log_once(logger, logging.INFO, "Building train dataloader (process_index=%s)...", accelerator.process_index)
    train_dataloader = data_provider_func(args, accelerator.process_index, accelerator.num_processes)

    # ===================== 优化器（仅 LoRA 参数） =====================
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
    elif args.use_came:
        try:
            from came_pytorch import CAME
            optimizer_cls = CAME
        except ImportError:
            raise ImportError("Please install came_pytorch: pip install came_pytorch")
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [p for p in transformer3d.parameters() if p.requires_grad]
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16),
        )
    else:
        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if not args.streaming:
        train_dataloader_len = len(train_dataloader)
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer3d, optimizer, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, lr_scheduler
    )

    if args.streaming:
        args.num_train_epochs = 100
    else:
        train_dataloader_len = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ===================== LoRA 专用保存/加载 hook =====================
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # 只保存 LoRA adapter（PEFT 格式）
            model = models[0]
            if hasattr(model, "save_pretrained"):
                adapter_path = os.path.join(output_dir, "adapter")
                os.makedirs(adapter_path, exist_ok=True)
                model.save_pretrained(adapter_path)
        if hasattr(train_dataloader, "state_dict"):
            with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as f:
                pickle.dump([train_dataloader.state_dict(), epoch], f)
        if weights:
            weights.pop()

    def load_model_hook(models, input_dir):
        adapter_path = os.path.join(input_dir, "adapter")
        if os.path.isdir(adapter_path):
            model = accelerator.unwrap_model(models[0]) if hasattr(accelerator, "unwrap_model") else models[0]
            if hasattr(model, "load_adapter"):
                model.load_adapter(adapter_path)
            elif hasattr(model, "load_pretrained"):
                model.load_pretrained(adapter_path)
            else:
                from safetensors.torch import load_file
                state = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
                (accelerator.unwrap_model(models[0]) if hasattr(accelerator, "unwrap_model") else models[0]).load_state_dict(state, strict=False)
        pkl_path = os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                state_dict, first_epoch = pickle.load(f)
                train_dataloader.load_state_dict(state_dict)
                log_once(logger, logging.INFO, "Load dataloader state and first_epoch=%s from %s", first_epoch, pkl_path)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        for k in ("validation_prompts", "trainable_modules", "trainable_modules_low_learning_rate", "fix_sample_size"):
            tracker_config.pop(k, None)
        tracker_config.pop("lora_target_modules", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ===================== 训练循环（与 SFT 一致） =====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    log_once(logger, logging.INFO, "***** Running LoRA training *****")
    log_once(logger, logging.INFO, "  Num Epochs = %s", args.num_train_epochs)
    log_once(logger, logging.INFO, "  Instantaneous batch size per device = %s", args.train_batch_size)
    log_once(logger, logging.INFO, "  Total train batch size (w. parallel, distributed & accumulation) = %s", total_batch_size)
    log_once(logger, logging.INFO, "  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    log_once(logger, logging.INFO, "  Total optimization steps = %s", args.max_train_steps)

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            accelerator.print("Checkpoint '%s' does not exist. Starting fresh." % args.resume_from_checkpoint)
            args.resume_from_checkpoint = None
        else:
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            pkl_path = os.path.join(args.output_dir, path, f"dataloader_{accelerator.process_index}_state_dict.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    _, first_epoch = pickle.load(f)
            else:
                first_epoch = global_step // num_update_steps_per_epoch if not args.streaming else 0
            accelerator.print("Resuming from checkpoint", path)
            accelerator.load_state(os.path.join(args.output_dir, path))

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        if args.streaming and not (args.resume_from_checkpoint and epoch == first_epoch):
            train_dataloader.dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if isinstance(batch, dict) and batch == {}:
                log_once(logger, logging.WARNING, "Empty batch; skipping.")
                continue
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(accelerator.device)

            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch["pixel_values"].cpu(), batch["text"]
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                if batch.get("source_images_transposed") is not None:
                    source_images_transposed = batch["source_images_transposed"]
                    source_images = list(map(list, zip(*source_images_transposed)))
                    for idx, (pixel_value, source_im, text) in enumerate(zip(pixel_values, source_images, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = "-".join((text or "").replace("/", "").split()[:10]) or f"{global_step}-{idx}"
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:50]}.jpg", rescale=True)
                        for li, _im in enumerate(source_im):
                            Image.fromarray(np.uint8(_im)).save(f"{args.output_dir}/sanity_check/source_{li}_{sanity_name[:50]}.jpg")
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = "-".join((text or "").replace("/", "").split()[:10]) or f"{global_step}-{idx}"
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:50]}.jpg", rescale=True)

            with accelerator.accumulate(transformer3d):
                loss = forward_step(
                    args,
                    accelerator.process_index,
                    transformer3d,
                    vae,
                    text_encoder,
                    extra_modules,
                    batch,
                    weight_dtype,
                    accelerator.device,
                )
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    max_grad_norm = args.max_grad_norm
                    total_norm = None
                    if not getattr(args, "use_fsdp", False):
                        grads = [p.grad for p in trainable_params if p.grad is not None]
                        if grads:
                            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
                        max_grad_norm = linear_decay(
                            args.max_grad_norm * args.initial_grad_norm_ratio,
                            args.max_grad_norm,
                            args.abnormal_norm_clip_start,
                            global_step,
                        )
                        if total_norm is not None and total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            max_grad_norm = max_grad_norm / min((total_norm / max_grad_norm).item(), 10)
                    accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                    if accelerator.is_main_process and getattr(args, "report_model_info", False) and total_norm is not None:
                        writer.add_scalar("gradients/norm_sum", total_norm.item(), global_step=global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    writer.add_scalar("loss/step_loss", loss.detach().item(), global_step=global_step)
                    writer.add_scalar("loss/train_loss", train_loss, global_step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for cp in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, cp), ignore_errors=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    log_once(logger, logging.INFO, "Saved state (adapter) to %s", save_path)

            progress_bar.set_postfix(step_loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    log_once(logger, logging.INFO, "Saved final state (adapter) to %s", save_path)
    accelerator.end_training()
    return


if __name__ == "__main__":
    import importlib
    from .arguments import parse_args
    from .data_provider import data_provider_impl
    from .model_provider import model_provider_impl
    from .forward_step import forward_step_impl

    args = parse_args()
    if getattr(args, "lora_r", 0) <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")

    lora_train(
        data_provider_impl,
        model_provider_impl,
        forward_step_impl,
        args,
    )
