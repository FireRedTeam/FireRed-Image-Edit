## Quick Start

The workflow has two steps: **Step 1** extract VLM embeddings offline, **Step 2** train using the extracted embeddings. Multi-dataset training with configurable weights is supported.

---

### Step 1: Offline VLM Embedding Extraction

Use `src/extract_vlm_embeds.py` to read image–text pairs from JSONL, extract multimodal embeddings via Qwen2-VL, and save them to disk for later training. This reduces GPU memory usage and improves utilization.

**Data format**: Input is JSONL, one sample per line. Required fields (names can be overridden via arguments; below are the defaults):

- `source_image`: Path or URL of the source image(s) (can be a list for multiple source images)
- `target_image`: Path or URL of the target image
- **Text / instruction fields** (used to build VLM dialogue and embeddings; treated as empty string if missing):
  - `instruction`: Main instruction (English), describing how to go from source to target image
  - `instruction_cn`: Main instruction (Chinese)
  - `inverse_instruction`: (Optional) Inverse instruction (English), text description inferred from the target image; only needed when not using `--disable_inverse`
  - `inverse_instruction_cn`: (Optional) Inverse instruction (Chinese)

  **Note**: With `--disable_inverse`, only `instruction` / `instruction_cn` are required. With `--t2i_mode` (text-to-image), only the main instruction is needed, and inverse is implicitly disabled.

**JSONL example**:

```jsonl
{"source_image": "/data/img/001.png", "target_image": "/data/img/001_edit.png", "instruction": "Change the sky to sunset.", "instruction_cn": "把天空改成日落。", "inverse_instruction": "A photo of a landscape with a blue sky.", "inverse_instruction_cn": "一张蓝天下的风景照。"}
{"source_image": ["/data/ref1.png", "/data/ref2.png"], "target_image": "/data/out.png", "instruction": "Merge the two characters into one scene.", "instruction_cn": "把两个角色合成到一个场景里。"}
{"source_image": null, "target_image": "/data/generated.png", "instruction": "A cat sitting on a windowsill.", "instruction_cn": "一只猫坐在窗台上。"}
```

In the above: the first line is standard image editing (with inverse instruction); the second uses multiple source images; the third is T2I (no source image; use with `--t2i_mode --disable_inverse`).

**Single-node multi-GPU example** (run from project root):

```bash
# Single node, 8 GPUs
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=6003

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK \
  --master_port $MASTER_PORT --master_addr $MASTER_ADDR \
  src/extract_vlm_embeds.py \
  /path/to/your/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/Qwen-Image-Edit-2509 \
  --batch_size 4
```

**Common arguments**:

| Argument | Description | Default |
|----------|-------------|---------|
| `jsonl_path` | Input JSONL path | Required |
| `--output_jsonl_dir` | Output JSONL directory (one file per rank) | Required |
| `--embeddings_save_dir` | Directory to save embeddings | Required |
| `--model_path` | Qwen2-VL / Qwen-Image model path | `/dev/shm/Qwen-Image` |
| `--batch_size` | Batch size | 4 |
| `--image_sample_size` | Image sampling size | 512 |
| `--disable_inverse` | Disable inverse prompt | - |
| `--t2i_mode` | T2I mode (text-to-image) | - |

For multi-node, set `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` and launch with `torchrun` using `--nnodes`, `--node_rank`, etc. More examples in `examples/extract_vlm_embeds.sh`.

---

### Step 2: Training

The training script reads the directory layout produced in Step 1: each **subdirectory** is treated as one task (e.g. one dataset), containing its JSONL and embedding files. Use `train_data_weights` to set sampling weights per task for mixed training.

**Directory convention**:

- Under `train_data_meta_dir`, each top-level subdirectory = one task (e.g. `dataset_a`, `dataset_b`).
- Each task directory contains that task’s JSONL and embeddings (i.e. Step 1’s `--output_jsonl_dir` and `--embeddings_save_dir` organized per task under this directory).

**Single-node multi-GPU training example**:

```bash
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=6003

accelerate launch --mixed_precision="bf16" --use_fsdp \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=QwenImageTransformerBlock \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --num_processes $((GPUS_PER_NODE * NNODES)) \
  --num_machines $NNODES \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  -m src.sft \
  --pretrained_model_name_or_path="/path/to/Qwen-Image-Edit-2509" \
  --train_data_meta_dir="/path/to/your_meta_dir" \
  --train_data_weights="dataset_a=0.5,dataset_b=1.2,dataset_c=1.0" \
  --train_src_img_num_weights="0=1,1=1,2=1,3=0" \
  --train_batch_size=1 \
  --image_sample_size=512 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=1 \
  --max_train_steps=512 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=100 \
  --output_dir="/path/to/ckpts" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --trainable_modules "." \
  --vae_mini_batch=1
```

**Mixed-training related arguments**:

| Argument | Description |
|----------|-------------|
| `--train_data_meta_dir` | Root directory for training meta; its top-level subdirs are tasks (datasets) |
| `--train_data_weights` | Sampling weight per task, format: `task1=w1,task2=w2`; tasks not listed are excluded |
| `--train_src_img_num_weights` | Weight by number of source images, format: `0=w0,1=w1,2=w2,3=w3` (0/1/2/3 source images) |

Adjust dataset names and weights in `train_data_weights` as needed for your multi-dataset mix. More examples in `examples/train.sh`.

---

## Dependencies and Environment

- Python 3.12
- PyTorch, Transformers, Accelerate, Diffusers, etc. (see project `requirements` or your environment setup)

---

## Directory Structure Example

```
post-trainers/
├── README.md
├── src/
│   ├── extract_vlm_embeds.py   # Step 1: offline VLM embedding extraction
│   ├── sft.py                  # Step 2: SFT training entry
│   ├── arguments.py
│   ├── data_provider.py
│   └── ...
└── examples/
    ├── extract_vlm_embeds.sh   # Extraction example
    └── train.sh                # Training example
```

Step 1 produces one task directory per JSONL (containing jsonl + embeddings). Place these task directories under a single `train_data_meta_dir`, then set weights in Step 2 via `train_data_weights` for mixed training.
