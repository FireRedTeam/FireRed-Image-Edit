"""
模型构建：加载 VAE、Transformer、可选 Text Encoder 与 Scheduler，并返回供 SFT 使用的组件。
"""
import logging
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from .utils.discrete_sampler import DiscreteSampling
from .utils.log_utils import get_logger, log_once

logger = get_logger(__name__)

def create_peft_lora_model(transformer3d, args, weight_dtype=None):
    """
    使用 PEFT 库为 transformer 创建 LoRA 模型
    参考: https://huggingface.co/docs/peft/accelerate/fsdp
    参考: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh
    """
    from peft import LoraConfig, get_peft_model
    
    # 解析 target_modules
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    print(f"Creating LoRA with config:")
    print(f"  rank (r): {args.lora_r}")
    print(f"  alpha: {args.lora_alpha}")
    print(f"  dropout: {args.lora_dropout}")
    print(f"  target_modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        init_lora_weights="gaussian",
    )
    
    # 先冻结所有参数
    transformer3d.requires_grad_(False)
    
    # 用 PEFT 包装模型
    transformer3d = get_peft_model(transformer3d, lora_config)
    
    # 关键修复: 将 LoRA 参数转换为与基础模型相同的 dtype
    # FSDP 要求所有参数必须具有统一的 dtype
    if weight_dtype is not None:
        print(f"Converting LoRA parameters to {weight_dtype} for FSDP compatibility")
        for name, param in transformer3d.named_parameters():
            if param.requires_grad:  # 只转换 LoRA 参数
                param.data = param.data.to(weight_dtype)
    
    # 打印可训练参数信息
    transformer3d.print_trainable_parameters()
    
    # 验证所有参数的 dtype
    dtypes = set()
    for name, param in transformer3d.named_parameters():
        dtypes.add(param.dtype)
    print(f"Model parameter dtypes: {dtypes}")
    if len(dtypes) > 1:
        print("WARNING: Model has mixed dtypes, this may cause issues with FSDP!")
    
    # 如果有预训练的 LoRA 权重，加载它
    if args.lora_path is not None:
        print(f"Loading pretrained LoRA weights from: {args.lora_path}")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        import os
        
        if os.path.isdir(args.lora_path):
            # 如果是目录，尝试加载 adapter_model.safetensors
            lora_file = os.path.join(args.lora_path, "adapter_model.safetensors")
            if not os.path.exists(lora_file):
                lora_file = os.path.join(args.lora_path, "adapter_model.bin")
        else:
            lora_file = args.lora_path
            
        if lora_file.endswith(".safetensors"):
            state_dict = load_file(lora_file)
        else:
            state_dict = torch.load(lora_file, map_location="cpu")
        
        # 将加载的权重也转换为正确的 dtype
        if weight_dtype is not None:
            state_dict = {k: v.to(weight_dtype) for k, v in state_dict.items()}
        
        set_peft_model_state_dict(transformer3d, state_dict)
        print(f"LoRA weights loaded successfully!")
    
    return transformer3d


def model_provider_impl(
    args, 
    weight_dtype, 
    device
):
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    ).to(weight_dtype)
    vae.eval()

    transformer3d = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    # 冻结 VAE，仅训练 Transformer
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    if args.transformer_path is not None:
        log_once(logger, logging.INFO, "Loading transformer from checkpoint: %s", args.transformer_path)
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "Transformer load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    if args.vae_path is not None:
        log_once(logger, logging.INFO, "Loading VAE from checkpoint: %s", args.vae_path)
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "VAE load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    # 根据是否使用 LoRA 来设置训练模式
    if getattr(args, 'use_peft_lora', False):
        # 使用 PEFT LoRA 微调
        log_once(logger, logging.INFO, "=== Using PEFT LoRA for training ===")
        # 传入 weight_dtype 以确保 LoRA 参数与基础模型具有相同的 dtype
        transformer3d = create_peft_lora_model(transformer3d, args, weight_dtype=weight_dtype)
        transformer3d.train()
    else:
        # 原有的全量微调逻辑
        transformer3d.train()
        for name, param in transformer3d.named_parameters():
            for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break


    # 可选：同步加载 Text Encoder（用于在线编码），否则使用预提取的 embedding
    if args.condition_encoder_mode == "sync":
        # Get Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="tokenizer"
        )

        # Get processor
        processor = Qwen2VLProcessor.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="processor"
        )

        # Get text encoder
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.to(device)
        log_once(logger, logging.INFO, "Text encoder loaded (condition_encoder_mode=sync).")
    else:
        tokenizer = None
        processor = None
        text_encoder = None

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)

    # 噪声调度器与时间步采样
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    extra_modules = {
        "noise_scheduler": noise_scheduler,
        "idx_sampling": idx_sampling,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "dit_class": QwenImageTransformer2DModel,
        "tokenizer": tokenizer,
        "processor": processor,
    }
    
    return (transformer3d, text_encoder, vae, extra_modules)