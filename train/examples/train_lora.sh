export REDEDIT_LOG_LEVEL="INFO"

NCCL_DEBUG=INFO
if [ $WORLD_SIZE -gt 1 ];
then
   GPUS_PER_NODE=8
   NNODES=$WORLD_SIZE
   NODE_RANK=$RANK
   MASTER_ADDR=$MASTER_ADDR
   MASTER_PORT=$MASTER_PORT
 else
   GPUS_PER_NODE=8
   NNODES=1
   NODE_RANK=0
   MASTER_ADDR=localhost
   MASTER_PORT=6003
fi

DISTRIBUTED_ARGS="
    --num_processes $((GPUS_PER_NODE * NNODES)) \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT
"

accelerate launch --mixed_precision="bf16" \
  --num_processes $((GPUS_PER_NODE * NNODES)) \
  --num_machines $NNODES \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  -m src.lora \
  --pretrained_model_name_or_path="/path/to/FireRed-Image-Edit-1.0" \
  --train_data_meta_dir="/path/to/your_meta_dir" \
  --train_data_weights="dataset_a=0.5,dataset_b=1.2,dataset_c=1.0" \
  --train_src_img_num_weights="0=1,1=1,2=1,3=0" \
  --train_batch_size=2 \
  --image_sample_size=512 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=1 \
  --max_train_steps=512 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=100 \
  --output_dir="/path/to/lora_ckpts" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --lora_r=8 \
  --lora_alpha=8
