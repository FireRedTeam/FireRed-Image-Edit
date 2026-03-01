import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration
from optimum.quanto import quantize, qint8, freeze
import cache_dit
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig


# ======================== Internal Optimization Helpers ========================


def _linear_forward_hook(self, x: torch.Tensor, *args, **kwargs):
    """
    Custom forward pass to support Graph Capture compatibility with LoRA and Quanto.
    Essential for torch.compile to function correctly without graph breaks.
    """
    result = self.base_layer(x, *args, **kwargs)
    if not hasattr(self, "active_adapters"):
        return result
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A: continue
        lora_A, lora_B = self.lora_A[active_adapter], self.lora_B[active_adapter]
        dropout, scaling = self.lora_dropout[active_adapter], self.scaling[active_adapter]
        x_input = x.to(lora_A.weight.dtype)
        output = lora_B(lora_A(dropout(x_input))) * scaling
        result = result + output.to(result.dtype)
    return result


def _apply_compile(pipeline):
    """
    Activates static compilation for Transformer and VAE components.
    Includes Monkey-patching Linear layers for PEFT compatibility.
    """
    from peft.tuners.lora.layer import Linear
    
    # Apply custom forward patch to all LoRA layers
    for module in pipeline.transformer.modules():
        if isinstance(module, Linear):
            module.forward = _linear_forward_hook.__get__(module, Linear)
            
    torch._dynamo.config.recompile_limit = 1024
    
    # Static compilation for repetitive Transformer blocks
    pipeline.transformer.compile_repeated_blocks(mode="default", dynamic=True)
    
    # Compile VAE with reduced overhead mode for memory efficiency
    pipeline.vae = torch.compile(pipeline.vae, mode="reduce-overhead")


def _apply_cache(pipeline):
    """
    Enables Deep-Cache-DiT (DBCache) to skip redundant noise prediction steps.
    """
    cache_dit.enable_cache(
        pipeline,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=8, 
            Bn_compute_blocks=0,
            residual_diff_threshold=0.15, 
            max_warmup_steps=3
        ),
        calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1)
    )


# ======================== Public API ========================


def load_fast_pipeline(model_path: str, device: str = "cuda:0"):
    """
    Initializes an accelerated FireRed-Image-Edit pipeline.
    
    This loader applies:
    1. Int8 Quantization (Text Encoder & Transformer)
    2. DBCache (DiT acceleration)
    3. Static Compilation (VAE & Transformer)
    4. Hardware Warmup
    """
    weight_dtype = torch.bfloat16
    print(f"🚀 Initializing Fast Pipeline from: {model_path}")


    # 1. Component Loading & Quantization
    print("[1/4] Quantizing Text Encoder & Transformer...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)


    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    # Exclude output projection to maintain generation fidelity
    quantize(transformer, weights=qint8, exclude=["proj_out"])
    freeze(transformer)


    # 2. Pipeline Assembly
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path, 
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=weight_dtype
    )


    # 3. Apply Speed Optimizations
    print("[2/4] Enabling DiT Cache & Static Compilation...")
    _apply_cache(pipeline)
    _apply_compile(pipeline)
    
    # Memory Management Strategy
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    pipeline.to(device)


    # 4. Engine Warmup
    # First execution triggers kernel compilation, usually takes ~2-5 mins
    print("[3/4] Warming up... usually takes ~2-5 mins")
    fake_pil = Image.new('RGB', (896, 896), (128, 128, 128))
    with torch.no_grad():
        pipeline(
            image=[fake_pil], 
            prompt="warmup session", 
            num_inference_steps=4,
            negative_prompt=" "
        )


    print("✅ Optimization ready.")
    return pipeline