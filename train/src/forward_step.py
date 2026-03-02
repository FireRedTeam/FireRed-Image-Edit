# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""
前向与损失：准备 latent、噪声与条件，调用 Transformer 预测 flow target，计算加权 MSE loss。
"""
import torch
import torch.nn.functional as F
import numpy as np
from diffusers.training_utils import (
    EMAModel,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from .utils.log_utils import get_logger
from .utils.image_utils import resize_source_images_for_condition

logger = get_logger(__name__)


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """将 latent 重排为 (B, seq, C) 的 patch 序列格式。"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


DEFAULT_SYSTEM_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

DEFAULT_SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """根据图像序列长度计算 flow 的 shift 系数（线性插值）。"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def get_embedding(source_exist, instructions, source_images, processor, text_encoder):
    def _extract_valid_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        valid_hidden_states = []
        for i in range(batch_size):
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_hidden_states.append(hidden_states[i][valid_indices])
        return valid_hidden_states

    with torch.no_grad():
        conversations = []
        if not source_exist:
            source_images = [[] for _ in range(len(instructions))]
        for instruction, source_images_sample in zip(instructions, source_images):
            system_prompt = DEFAULT_SYSTEM_PROMPT_T2I if not source_exist else DEFAULT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": []},
            ]
            for condition_image in source_images_sample:
                messages[1]["content"].append({"type": "image", "image": condition_image})
            messages[1]["content"].append({"type": "text", "text": instruction})
            conversations.append(messages)
        
        batch_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(text_encoder.device)

        encoder_hidden_states = text_encoder(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            pixel_values=batch_inputs["pixel_values"] if source_exist else None,
            image_grid_thw=batch_inputs["image_grid_thw"] if source_exist else None,
            output_hidden_states=True,
        )

        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = _extract_valid_hidden_states(hidden_states, batch_inputs["attention_mask"])
        start_idx = 64 if source_exist else 34
        split_hidden_states = [e[start_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

    return prompt_embeds, encoder_attention_mask

def prepare_data(args,
                 batch,
                 processor, text_encoder, vae, latents_mean, latents_std,
                 noise_scheduler, idx_sampling,
                 device, weight_dtype, torch_rng):
    """
    从 batch 中编码 pixel_values/source 为 latent，采样时间步，构造 noisy_latents 与 target。
    返回模型前向所需的所有张量与元信息。
    """
    pixel_values = batch["pixel_values"].to(weight_dtype)
    source_images_transposed = batch["source_images_transposed"]
    encoder_attention_mask = batch['encoder_attention_mask']
    prompt_embeds = batch['encoder_hidden_states']
    instructions = batch['text']

    # prepare target latents
    def _batch_encode_vae(pixel_values):
        with torch.no_grad():
            pixel_values = vae.encode(pixel_values.to(vae.device).to(vae.dtype).unsqueeze(2))[0]
            pixel_values = pixel_values.sample()
            return pixel_values
    
    latents = _batch_encode_vae(pixel_values)
    latents = ((latents - latents_mean) * latents_std).to(dtype=weight_dtype)
    bsz, channel, _, height, width = latents.size()
    latents = pack_latents(latents, bsz, channel, height, width)
    noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

    # prepare source latents
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    vae_image_sizes_transposed = []
    source_latents_transposed = []
    source_exist = False

    for i_step, _vae_images in enumerate(source_images_transposed):
        _source_pixel_values = []
        _vae_image_sizes = []
        for j_img, _vae_image in enumerate(_vae_images):
            _vae_image_sizes.append(_vae_image.size)

            _vae_image = torch.from_numpy(np.expand_dims(np.array(_vae_image), 0)).to(dtype=weight_dtype, device=vae.device)
            _vae_image = _vae_image.permute(0, 3, 1, 2).contiguous()
            _vae_image = _vae_image / 255.
            _vae_image = (_vae_image - 0.5) / 0.5 # [1, C, H, W]
            _source_pixel_values.append(_vae_image)

        
        if len(_source_pixel_values) > 0:
            source_exist = True
        
        _source_pixel_values = torch.cat(_source_pixel_values, dim = 0) # [B, C, H, W]
        _source_latents = _batch_encode_vae(_source_pixel_values)
        _source_latents = ((_source_latents - latents_mean) * latents_std).to(dtype=weight_dtype)
        src_bsz, src_channel, _, src_height, src_width = _source_latents.size()
        _source_latents = pack_latents(_source_latents, src_bsz, src_channel, src_height, src_width) # [B, seq, C]

        source_latents_transposed.append(_source_latents.unbind(dim=0)) # [ [latent1, latent3], [latent2, latent4], ... ]
        vae_image_sizes_transposed.append(_vae_image_sizes) # [ [ (W1, H1), (W3, H3) ], [ (W2, H2), (W4, H4) ], ... ]
    
    if source_exist:
        # [ [latent1, latent3], [latent2, latent4] ] → [ [latent1, latent2], [latent3, latent4] ]
        source_latents_tmp = list(map(list, zip(*source_latents_transposed)))
        vae_image_sizes = list(map(list, zip(*vae_image_sizes_transposed)))

        # cat source_latents to tensor [B, seq, C]
        source_latents = [torch.cat(_source_latents, dim = 0).unsqueeze(0) for _source_latents in source_latents_tmp] # [[1, seq, C], [1, seq, C], ... ]
        source_latents = torch.cat(source_latents, dim = 0) # [B, seq, C]
        logger.debug(
            "source_latents final: shape=%s (B, seq, C)",
            tuple(source_latents.shape),
        )

        source_images = list(map(list, zip(*source_images_transposed)))
        # sync 模式下送入 text_encoder 的 source 图做与 extract_vlm_embeds 一致的 Resize+CenterCrop，以 target 尺寸为参考
        if args.condition_encoder_mode == "sync":
            ref_h, ref_w = pixel_values.shape[2], pixel_values.shape[3]
            source_images = resize_source_images_for_condition(
                source_images,
                ref_height=ref_h,
                ref_width=ref_w,
                image_sample_size=args.image_sample_size,
            )
    else:
        source_latents = None
        vae_image_sizes = []
        source_images = []
        logger.debug("source_latents: None (no source images)")
    
    # prepare prompt embeds
    if args.condition_encoder_mode == "sync":
        prompt_embeds, encoder_attention_mask = get_embedding(source_exist, instructions, source_images, processor, text_encoder)

    prompt_embeds = prompt_embeds.to(dtype=latents.dtype, device=device)
    encoder_attention_mask = encoder_attention_mask.to(device=device)

    if not args.uniform_sampling:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
    else:
        indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
        indices = indices.long().cpu()

    sigmas = np.linspace(1.0, 1 / args.train_sampling_steps, args.train_sampling_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.get("base_image_seq_len", 256),
        noise_scheduler.config.get("max_image_seq_len", 4096),
        noise_scheduler.config.get("base_shift", 0.5),
        noise_scheduler.config.get("max_shift", 1.15),
    )
    noise_scheduler.set_timesteps(sigmas=sigmas, device=latents.device, mu=mu)
    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Flow matching: zt = (1 - sigma) * x + sigma * noise, target = noise - latents
    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    target = noise - latents

    if source_exist:
        img_shapes = [
            [
                (1, height // 2, width // 2),
                *[
                    (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2)
                    for (vae_width, vae_height) in vae_image_sizes[0] #TODO
                ],
            ]
        ] * latents.size(0)
    else:
        img_shapes = [[(1, height // 2, width // 2)]] * latents.size(0)
    txt_seq_lens = encoder_attention_mask.sum(dim=1).tolist() if encoder_attention_mask is not None else None


    if source_latents is not None:
        noisy_latents_and_image_latents = torch.cat([noisy_latents, source_latents], dim=1)
    else:
        noisy_latents_and_image_latents = noisy_latents     

    logger.debug("   prompt_embeds.shape: %s", prompt_embeds.shape)
    logger.debug("   noisy_latents.shape: %s", noisy_latents.shape)
    logger.debug("   noisy_latents_and_image_latents.shape: %s", noisy_latents_and_image_latents.shape)
    logger.debug("   img_shapes: %s", img_shapes)

    return latents, noisy_latents, noisy_latents_and_image_latents, \
        encoder_attention_mask, prompt_embeds, \
        timesteps, img_shapes, txt_seq_lens, target, sigmas

def forward_step_impl(
    args,
    process_index,
    transformer3d,
    vae,
    text_encoder,
    extra_modules,
    batch,
    weight_dtype,
    device,
):
    """
    单步前向：prepare_data 后调用 Transformer 预测 flow，用 compute_loss_weighting_for_sd3 加权 MSE。
    """
    noise_scheduler = extra_modules["noise_scheduler"]
    idx_sampling = extra_modules["idx_sampling"]
    latents_mean = extra_modules["latents_mean"]
    latents_std = extra_modules["latents_std"]
    processor = extra_modules["processor"]

    torch_rng = torch.Generator(device=device).manual_seed(args.seed + process_index)

    latents, noisy_latents, noisy_latents_and_image_latents, encoder_attention_mask, prompt_embeds, timesteps, img_shapes, txt_seq_lens, target, sigmas = prepare_data(
        args, 
        batch, 
        processor, text_encoder, vae, latents_mean, latents_std, 
        noise_scheduler, idx_sampling, 
        device, weight_dtype, torch_rng)

    # Predict the noise residual
    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
        noise_pred = transformer3d(
            hidden_states=noisy_latents_and_image_latents,
            timestep=timesteps / 1000,
            encoder_hidden_states_mask=encoder_attention_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : noisy_latents.size(1)]
    
    def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
        noise_pred = noise_pred.float()
        target = target.float()
        diff = noise_pred - target
        mse_loss = F.mse_loss(noise_pred, target, reduction='none')
        mask = (diff.abs() <= threshold).float()
        masked_loss = mse_loss * mask
        if weighting is not None:
            masked_loss = masked_loss * weighting
        final_loss = masked_loss.mean()
        return final_loss
    
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
    loss = loss.mean()
    return loss