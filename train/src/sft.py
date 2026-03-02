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
from accelerate.utils import ProjectConfiguration, set_seed, DataLoaderConfiguration, DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from glob import glob

from .utils.other import linear_decay
from .utils.image_utils import save_image
from .utils.log_utils import get_logger, DistributedColoredFormatter, get_dist_prefix, get_default_log_level, log_once


def sft(
    data_provider_func,
    model_provider_func,
    forward_step,
    args,
):
    # ===================== Accelerator 初始化 =====================
    logger = get_logger("REDEdit.sft")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    use_peft_lora = getattr(args, "use_peft_lora", False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=use_peft_lora)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        kwargs_handlers=[ddp_kwargs],
    )

    # 配置根 logger 为带分布式前缀的彩色格式，便于多进程/多机阅读
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DistributedColoredFormatter(dist_prefix=get_dist_prefix()))
    root.addHandler(handler)
    root.setLevel(get_default_log_level())

    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None

    if fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD or fsdp_plugin.sharding_strategy is ShardingStrategy.HYBRID_SHARD or fsdp_plugin.reshard_after_forward == True:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:  # FSDP 2 中为 None
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        log_once(logger, logging.INFO, "Using FSDP stage: %s", fsdp_stage)

        args.use_fsdp = True
        if fsdp_stage == 3:
            log_once(logger, logging.INFO, "Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        fsdp_stage = 0
        log_once(logger, logging.INFO, "FSDP/HSDP is not enabled.")

    if use_peft_lora and fsdp_plugin is not None:
        log_once(logger, logging.WARNING, "LoRA 训练建议不使用 FSDP；当前已启用 FSDP，保存时将仅保存 adapter 权重。")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # 各进程打印当前配置（带 rank 前缀）
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 混合精度对应的权重 dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    # ===================== 设置随机种子 =====================
    if args.seed is not None:
        set_seed(args.seed)

    # ===================== 模型初始化 =====================
    log_once(logger, logging.INFO, "Loading model via model_provider...")
    transformer3d, text_encoder, vae, extra_modules = model_provider_func(args, weight_dtype, accelerator.device)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        log_once(logger, logging.INFO, "Gradient checkpointing enabled.")

    if use_peft_lora and getattr(args, "lora_r", 0) <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")

    # ===================== 数据加载 =====================
    log_once(logger, logging.INFO, "Building train dataloader (process_index=%s)...", accelerator.process_index)
    train_dataloader = data_provider_func(args, accelerator.process_index, accelerator.num_processes)

    # ===================== 优化器初始化 =====================
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))

    if use_peft_lora:
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
    else:
        trainable_params_optim = [
            {'params': [], 'lr': args.learning_rate},
            {'params': [], 'lr': args.learning_rate / 2},
        ]
        in_already = []
        for name, param in transformer3d.named_parameters():
            high_lr_flag = False
            if name in in_already:
                continue
            for trainable_module_name in args.trainable_modules:
                if trainable_module_name in name:
                    in_already.append(name)
                    high_lr_flag = True
                    trainable_params_optim[0]['params'].append(param)
                    # if accelerator.is_main_process:
                        # logger.debug("Set %s to lr: %s", name, args.learning_rate)
                    break
            if high_lr_flag:
                continue
            for trainable_module_name in args.trainable_modules_low_learning_rate:
                if trainable_module_name in name:
                    in_already.append(name)
                    trainable_params_optim[1]['params'].append(param)
                    # if accelerator.is_main_process:
                        # logger.debug("Set %s to lr: %s", name, args.learning_rate / 2)
                    break
        if args.use_came:
            optimizer = optimizer_cls(
                trainable_params_optim,
                lr=args.learning_rate,
                # weight_decay=args.adam_weight_decay,
                betas=(0.9, 0.999, 0.9999),
                eps=(1e-30, 1e-16)
            )
        else:
            optimizer = optimizer_cls(
                trainable_params_optim,
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
    
    # 学习率调度器初始化 ==============================================================================
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator` ==============================================================================
    transformer3d, optimizer, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if args.streaming:
        args.num_train_epochs = 100 # set to a large number to avoid early stopping
    else:
        train_dataloader_len = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ===================== 保存/加载 hook（LoRA / FSDP / 普通模式） =====================
    if use_peft_lora:
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                model = accelerator.unwrap_model(models[0]) if hasattr(accelerator, "unwrap_model") else models[0]
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
    elif fsdp_stage != 0:
        def save_model_hook(models, weights, output_dir):
            accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
            if accelerator.is_main_process:
                from safetensors.torch import save_file

                safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
                accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})

            with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as file:
                pickle.dump([train_dataloader.state_dict(), epoch], file)

        def load_model_hook(models, input_dir):
            with open(os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "rb") as file:
                state_dict, first_epoch = pickle.load(file)
                train_dataloader.load_state_dict(state_dict)
                log_once(logger, logging.INFO,
                    "Load dataloader state dict and first_epoch=%s from %s",
                    first_epoch,
                    os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"),
                )
    else:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                weights.pop()

            with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as file:
                pickle.dump([train_dataloader.state_dict(), epoch], file)

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = extra_modules["dit_class"].from_pretrained(
                    input_dir, subfolder="transformer"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

            with open(os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "rb") as file:
                state_dict, first_epoch = pickle.load(file)
                train_dataloader.load_state_dict(state_dict)
                log_once(logger, logging.INFO,
                    "Load dataloader state dict and first_epoch=%s from %s",
                    first_epoch,
                    os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"),
                )
                
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts", None)
        tracker_config.pop("trainable_modules", None)
        tracker_config.pop("trainable_modules_low_learning_rate", None)
        tracker_config.pop("fix_sample_size", None)
        tracker_config.pop("lora_target_modules", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ===================== 训练循环 =====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    log_once(logger, logging.INFO, "***** Running %s *****", "LoRA training" if use_peft_lora else "training")
    log_once(logger, logging.INFO, "  Num Epochs = %s", args.num_train_epochs)
    log_once(logger, logging.INFO, "  Instantaneous batch size per device = %s", args.train_batch_size)
    log_once(logger, logging.INFO, "  Total train batch size (w. parallel, distributed & accumulation) = %s", total_batch_size)
    log_once(logger, logging.INFO, "  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    log_once(logger, logging.INFO, "  Total optimization steps = %s", args.max_train_steps)
    # Potentially load in the weights and states from a previous save
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            
            pkl_path = os.path.join(os.path.join(args.output_dir, path), f"dataloader_{accelerator.process_index}_state_dict.pkl")

            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
                log_once(logger, logging.INFO, "Load pkl from %s, first_epoch=%s.", pkl_path, first_epoch)
            else:
                first_epoch = global_step // num_update_steps_per_epoch if not args.streaming else 0

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        if args.streaming:
            if not (args.resume_from_checkpoint and epoch == first_epoch):
                # Dont set epoch if resuming from checkpoint and epoch is the first epoch
                # because the state of the streaming dataset has already loaded from checkpoint
                train_dataloader.dataset.set_epoch(epoch)
            
        for step, batch in enumerate(train_dataloader):
            if isinstance(batch, dict) and batch == {}:
                log_once(logger, logging.WARNING, "Empty batch encountered; skipping.")
                continue
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(accelerator.device)

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                if batch.get('source_images_transposed', None) is not None:
                    source_images_transposed = batch['source_images_transposed']
                    source_images =  list(map(list, zip(*source_images_transposed)))
                
                    for idx, (pixel_value, source_im, text) in enumerate(zip(pixel_values, source_images, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:10]}.jpg", rescale=True)
                        for local_index, _source_im in enumerate(source_im):
                            _source_im = Image.fromarray(np.uint8(_source_im))
                            _source_im.save(f"{args.output_dir}/sanity_check/source_{local_index}_{sanity_name[:10]}.jpg")
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:10]}.jpg", rescale=True)
            
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

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm
                    
                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # Record step loss to TensorBoard
                if accelerator.is_main_process:
                    step_loss_value = loss.detach().item()
                    writer.add_scalar("loss/step_loss", step_loss_value, global_step=global_step)
                    writer.add_scalar("loss/train_loss", train_loss, global_step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                log_once(logger, logging.INFO,
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                log_once(logger, logging.INFO, f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        log_once(logger, logging.INFO, "Saved state%s to %s", " (adapter)" if use_peft_lora else "", save_path)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        log_once(logger, logging.INFO, "Saved state%s to %s", " (adapter)" if use_peft_lora else "", save_path)

    accelerator.end_training()


    return


if __name__ == "__main__":
    from .forward_step import forward_step_impl
    from .model_provider import model_provider_impl
    from .data_provider import data_provider_impl
    from .arguments import parse_args

    args = parse_args()
    if getattr(args, "use_peft_lora", False) and getattr(args, "lora_r", 0) <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")
    sft(
        data_provider_impl,
        model_provider_impl,
        forward_step_impl,
        args,
    )