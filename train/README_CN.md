## 快速开始

整体流程分为两步：**第一步** 离线抽取 VLM embeddings，**第二步** 使用抽取好的 embeddings 进行训练。支持对多个数据集按权重混合训练。

---

### 第一步：离线抽取 VLM Embeddings

使用 `src/extract_vlm_embeds.py` 从 JSONL 中读取图文对，通过 Qwen2-VL 抽取多模态 embedding 并落盘，供后续训练直接加载，降低显存占用，提高GPU利用率。

**数据格式**：输入为 JSONL，每行一条样本，需包含字段（字段名可通过参数指定，以下为默认）：

- `source_image`：源图路径或 URL（可为列表，多条源图）
- `target_image`：目标图路径或 URL
- **文本/指令字段**（用于构造 VLM 的对话与 embedding，缺失时按空字符串处理）：
  - `instruction`：主指令（英文），描述希望如何从源图得到目标图
  - `instruction_cn`：主指令（中文）
  - `inverse_instruction`：（可选）逆向指令（英文），从目标图反推的文本描述；仅在不使用 `--disable_inverse` 时需要
  - `inverse_instruction_cn`：（可选）逆向指令（中文）

  **说明**：启用 `--disable_inverse` 时只需提供 `instruction` / `instruction_cn`；使用 `--t2i_mode`（文生图）时同样只需主指令，且会强制视为禁用逆向。

**JSONL 示例**：

```jsonl
{"source_image": "/data/img/001.png", "target_image": "/data/img/001_edit.png", "instruction": "Change the sky to sunset.", "instruction_cn": "把天空改成日落。", "inverse_instruction": "A photo of a landscape with a blue sky.", "inverse_instruction_cn": "一张蓝天下的风景照。"}
{"source_image": ["/data/ref1.png", "/data/ref2.png"], "target_image": "/data/out.png", "instruction": "Merge the two characters into one scene.", "instruction_cn": "把两个角色合成到一个场景里。"}
{"source_image": null, "target_image": "/data/generated.png", "instruction": "A cat sitting on a windowsill.", "instruction_cn": "一只猫坐在窗台上。"}
```

上例中：第一条为常规图编辑（含逆向指令）；第二条为多源图；第三条为 T2I（无源图，需配合 `--t2i_mode --disable_inverse`）。

**单机多卡示例**（在项目根目录下执行）：

```bash
# 单机 8 卡
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
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
```

**常用参数**：

| 参数 | 说明 | 默认 |
|------|------|------|
| `jsonl_path` | 输入 JSONL 路径 | 必填 |
| `--output_jsonl_dir` | 输出 JSONL 目录（每 rank 一个文件） | 必填 |
| `--embeddings_save_dir` | embedding 保存目录 | 必填 |
| `--model_path` | FireRed-Image-Edit-1.0 模型路径 | `/dev/shm/FireRed-Image-Edit-1.0` |
| `--batch_size` | 批次大小 | 4 |
| `--image_sample_size` | 图像采样尺寸 | 512 |
| `--disable_inverse` | 禁用逆向提示词 | - |
| `--t2i_mode` | T2I 模式（文生图） | - |

多机时设置环境变量 `WORLD_SIZE`、`RANK`、`MASTER_ADDR`、`MASTER_PORT` 后，用 `torchrun` 的 `--nnodes`、`--node_rank` 等启动即可。更多示例见 `examples/extract_vlm_embeds.sh`。

---

### 第二步：训练

训练脚本读取「第一步」产出的目录结构：每个**子目录**视为一个 task（例如一个数据集），其下为对应的 JSONL 与 embedding 文件。通过 `train_data_weights` 指定各 task 的采样权重，实现混合训练。

**目录约定**：

- `train_data_meta_dir` 下每个一级子目录 = 一个 task（如 `dataset_a`、`dataset_b`）。
- 每个 task 目录内包含该 task 的 JSONL 与对应的 embeddings（即第一步的 `--output_jsonl_dir` 与 `--embeddings_save_dir` 按 task 组织到该目录下）。

**单机多卡训练示例**：

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
  --pretrained_model_name_or_path="/path/to/FireRed-Image-Edit-1.0" \
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

**混合训练相关参数**：

| 参数 | 说明 |
|------|------|
| `--train_data_meta_dir` | 训练 meta 根目录，其下一级子目录为各 task（数据集） |
| `--train_data_weights` | 各 task 采样权重，格式：`task1=w1,task2=w2`，未列出的 task 不参与训练 |
| `--train_src_img_num_weights` | 按「源图数量」加权，格式：`0=w0,1=w1,2=w2,3=w3`（0/1/2/3 张源图） |

在 `train_data_weights` 中填入你的任务名（与 `train_data_meta_dir` 下子目录名一致）及所需采样权重即可；仅在此列出的任务会参与训练。更多可运行脚本见 `examples/`（如 `examples/train.sh`），完整训练参数见 `src/arguments.py`。多机训练时需配置环境变量 `WORLD_SIZE`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`。

---

## 依赖与环境

- Python 3.12
- PyTorch、Transformers、Accelerate、Diffusers 等（见项目 `requirements` 或环境配置）

