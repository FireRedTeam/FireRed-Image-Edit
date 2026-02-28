"""
训练脚本命令行参数解析。

定义 SFT 所需的所有 CLI 参数（数据、模型、优化器、分布式、checkpoint 等），
并兼容 torchrun/accelerate 的 LOCAL_RANK 环境变量。
"""
import argparse
import logging
import os
from .utils.log_utils import get_logger, log_once

logger = get_logger(__name__)


def parse_args(extra_parser=None):
    if extra_parser is not None:
        parser = argparse.ArgumentParser(description="Simple example of a training script.", parents=[extra_parser])
    else:
        parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # ===================== 分布式 / 环境 =====================
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练时的 local_rank（由 torchrun/accelerate 注入）")
    parser.add_argument("--nnodes", type=int, default=1, help="节点数量")
    parser.add_argument("--num_processes_per_node", type=int, default=1, help="每节点进程数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于复现训练")
    parser.add_argument("--use_fsdp", action="store_true", help="是否使用 FSDP")

    # ===================== 模型与路径 =====================
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="预训练模型路径或 HuggingFace 模型 id",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="若要从其他 checkpoint 加载 transformer 权重，填写其路径",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="若要从其他 checkpoint 加载 VAE 权重，填写其路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model-finetuned",
        help="模型与 checkpoint 输出目录",
    )

    # ===================== 数据路径与加权 =====================
    parser.add_argument(
        "--train_data_meta_dir",
        type=str,
        default=None,
        help="训练 meta 目录：其下一级子目录名为 task，每个 task 目录下为若干 .jsonl 标注文件",
    )
    parser.add_argument(
        "--train_data_weights",
        type=str,
        default=None,
        help="按 train_data_meta_dir 下的一级子目录（task）加权采样，格式：'dirA=1,dirB=2,dirC=0.5'",
    )
    parser.add_argument(
        "--train_src_img_num_weights",
        type=str,
        default="0=0,1=10,2=0,3=0",
        help="按源图数量加权。格式：'0=0,1=10,2=0,3=0' 表示 0 张源图权重 0，1 张权重 10，以此类推",
    )
    # ===================== 训练步数 / 轮数 / batch =====================
    parser.add_argument("--num_train_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="总训练步数；若设置则覆盖 num_train_epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="每设备的 batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数，再执行一次 backward/update",
    )

    # ===================== 学习率与调度器 =====================
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="初始学习率（warmup 之后）",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="是否按 GPU 数、梯度累积步数、batch size 缩放学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='学习率调度器：["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="学习率 warmup 步数")

    # ===================== 优化器 =====================
    parser.add_argument("--use_8bit_adam", action="store_true", help="是否使用 8-bit Adam（bitsandbytes）")
    parser.add_argument("--use_came", action="store_true", help="是否使用 CAME 优化器")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam 权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")

    # ===================== 梯度与精度 =====================
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪的最大范数")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="是否使用梯度检查点以省显存（反向传播会更慢）",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="混合精度：fp16/bf16；默认跟随 accelerate 配置",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="是否在 Ampere GPU 上允许 TF32 以加速训练",
    )

    # ===================== 可训练模块 =====================
    parser.add_argument(
        "--trainable_modules",
        nargs="+",
        help="需要参与训练、使用正常学习率的模块名列表",
    )
    parser.add_argument(
        "--trainable_modules_low_learning_rate",
        nargs="+",
        default=[],
        help="需要参与训练、使用一半学习率的模块名列表",
    )

    # ===================== 采样与 loss 加权（SD3 相关） =====================
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="采样步数（用于 loss 加权等）",
    )
    parser.add_argument(
        "--uniform_sampling",
        action="store_true",
        help="是否对采样步做均匀采样（否则按 weighting_scheme 加权）",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="loss 加权方案；'none' 表示均匀",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0, help="weighting_scheme='logit_normal' 时的均值")
    parser.add_argument("--logit_std", type=float, default=1.0, help="weighting_scheme='logit_normal' 时的标准差")
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="weighting_scheme='mode' 时的 scale",
    )

    # ===================== 数据加载与图像尺寸 =====================
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="训练时图像采样边长（正方形）",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="DataLoader 子进程数；0 表示主进程加载",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="每个 worker 预取的 sample 数",
    )
    parser.add_argument("--sync_text_encoder", action="store_true", help="是否在 dataloader 内同步预计算 text encoder 输出，暂不支持")
    parser.add_argument("--enable_inverse", action="store_true", help="是否在数据集中启用 inverse captions")

    # ===================== VAE 与流水线 =====================
    parser.add_argument("--vae_mini_batch", type=int, default=32, help="VAE 编码/解码时的 mini batch 大小")

    # ===================== 日志与实验追踪 =====================
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard 等日志目录",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='上报结果与日志的平台：tensorboard / wandb / comet_ml；"all" 表示全部',
    )
    parser.add_argument(
        "--report_model_info",
        action="store_true",
        help="是否在训练中上报模型信息（如 norm、grad）",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="rededit-finetune",
        help="传给 Accelerator.init_trackers 的 project 名称",
    )

    # ===================== Checkpoint =====================
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="每 X 步保存一次 checkpoint（仅用于 --resume_from_checkpoint 恢复）",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="最多保留的 checkpoint 数量，超出时删除最旧的",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='从该 checkpoint 恢复训练；可为 "latest" 自动选最新',
    )

    # ===================== 梯度异常与调试 =====================
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help="从该步开始对异常梯度做额外裁剪/处理",
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help="初始阶段 max_grad_norm 的倍数（线性衰减到 max_grad_norm）",
    )

    # ===================== 流式 / 其他 =====================
    parser.add_argument("--streaming", action="store_true", help="是否使用流式数据模式")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    log_once(logger, logging.DEBUG, "Parsed args: output_dir=%s, pretrained=%s", args.output_dir, getattr(args, "pretrained_model_name_or_path", None))
    return args
