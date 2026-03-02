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

accelerate launch --mixed_precision="bf16" $DISTRIBUTED_ARGS -m src.sft \
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
  --output_dir="/path/to/ckpts_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --condition_encoder_mode="sync" \
  --use_peft_lora \
  --lora_r 128 \
  --lora_alpha 128 \
  --lora_dropout 0.0 \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1
