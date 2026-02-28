"""
模型构建：加载 VAE、Transformer、可选 Text Encoder 与 Scheduler，并返回供 SFT 使用的组件。
"""
import logging
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from .discrete_sampler import DiscreteSampling
from .utils.log_utils import get_logger, log_once

logger = get_logger(__name__)


def model_provider_impl(
    extra_args, 
    weight_dtype, 
    device
):
    vae = AutoencoderKLQwenImage.from_pretrained(
        extra_args.pretrained_model_name_or_path,
        subfolder="vae"
    ).to(weight_dtype)
    vae.eval()

    transformer3d = QwenImageTransformer2DModel.from_pretrained(
        extra_args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    # 冻结 VAE，仅训练 Transformer
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    if extra_args.transformer_path is not None:
        log_once(logger, logging.INFO, "Loading transformer from checkpoint: %s", extra_args.transformer_path)
        if extra_args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(extra_args.transformer_path)
        else:
            state_dict = torch.load(extra_args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "Transformer load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    if extra_args.vae_path is not None:
        log_once(logger, logging.INFO, "Loading VAE from checkpoint: %s", extra_args.vae_path)
        if extra_args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(extra_args.vae_path)
        else:
            state_dict = torch.load(extra_args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "VAE load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    transformer3d.train()
    # 仅对 trainable_modules / trainable_modules_low_learning_rate 中的参数开启 requires_grad
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in extra_args.trainable_modules + extra_args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # 可选：同步加载 Text Encoder（用于在线编码），否则使用预提取的 embedding
    if extra_args.sync_text_encoder:
        # Get Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            extra_args.pretrained_model_name_or_path, subfolder="tokenizer"
        )

        # Get processor
        processor = Qwen2VLProcessor.from_pretrained(
            extra_args.pretrained_model_name_or_path,
            subfolder="processor"
        )

        # Get text encoder
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            extra_args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.to(device)
        log_once(logger, logging.INFO, "Text encoder loaded (sync_text_encoder=True).")
    else:
        tokenizer = None
        processor = None
        text_encoder = None

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)

    # 噪声调度器与时间步采样
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        extra_args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    idx_sampling = DiscreteSampling(extra_args.train_sampling_steps, uniform_sampling=extra_args.uniform_sampling)

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