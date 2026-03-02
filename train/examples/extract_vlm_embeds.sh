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

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK \
  --master_port $MASTER_PORT --master_addr $MASTER_ADDR \
  -m src.extract_vlm_embeds \
  /path/to/your/data.jsonl \
  --output_jsonl_dir /path/to/output_jsonl \
  --embeddings_save_dir /path/to/embeddings \
  --model_path /path/to/FireRed-Image-Edit-1.0 \
  --batch_size 4
