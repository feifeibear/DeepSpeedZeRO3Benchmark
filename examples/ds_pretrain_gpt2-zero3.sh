#! /bin/bash

# Change for multinode config
MP_SIZE=${MP_SIZE:-1}
NUM_WORKERS=${NUM_GPU:-1}
NUM_GPUS_PER_WORKER=${NUM_GPUS_PER_WORKER:-1} #${DLTS_NUM_GPU_PER_WORKER}

DEBUG=${DEBUG:0}
MODEL_SIZE=${MODEL_SIZE:-4}
BATCHSIZE=${BS:-8}
export PA_CPU=${PA_CPU:-false}


if [[ ${DEBUG} == 1 ]];  then
       echo "debug model"
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=1
       # HIDDEN_SIZE=1024
       HIDDEN_SIZE=32
       # NUM_LAYERS=5
       NUM_LAYERS=1
       BATCHSIZE=1
else
       echo "MODEL SIZE ${MODEL_SIZE} MP_SIZE ${MP_SIZE} NUM_GPUS_PER_WORKER ${NUM_GPUS_PER_WORKER}"
       if [[ ${MODEL_SIZE} == 1 ]]; then
            HIDDEN_SIZE=2048
            NUM_LAYERS=20
       elif [[ ${MODEL_SIZE} == 2 ]]; then
            HIDDEN_SIZE=2048
            NUM_LAYERS=40
       elif [[ ${MODEL_SIZE} == 4 ]]; then
            HIDDEN_SIZE=2304
            NUM_LAYERS=64
       elif [[ ${MODEL_SIZE} == 6 ]]; then
            HIDDEN_SIZE=3072
            NUM_LAYERS=53
       elif [[ ${MODEL_SIZE} == 8 ]]; then
            HIDDEN_SIZE=3072
            NUM_LAYERS=72
       elif [[ ${MODEL_SIZE} == 10 ]]; then
            HIDDEN_SIZE=4096
            NUM_LAYERS=50
       elif [[ ${MODEL_SIZE} == 12 ]]; then
            HIDDEN_SIZE=4096
            NUM_LAYERS=60
       elif [[ ${MODEL_SIZE} == 13 ]]; then
            HIDDEN_SIZE=4096
            NUM_LAYERS=65
       elif [[ ${MODEL_SIZE} == 15 ]]; then
            HIDDEN_SIZE=4096
            NUM_LAYERS=78
       elif [[ ${MODEL_SIZE} == 20 ]]; then
            HIDDEN_SIZE=8192
            NUM_LAYERS=25
       elif [[ ${MODEL_SIZE} == 30 ]]; then
            HIDDEN_SIZE=8192
            NUM_LAYERS=37
       elif [[ ${MODEL_SIZE} == 40 ]]; then
            HIDDEN_SIZE=8192
            NUM_LAYERS=50
       else
           echo "Model size ${MODEL_SIZE} is not supported"
       fi
       #HIDDEN_SIZE=4096
       #NUM_LAYERS=24 # 50
fi


# DATA_PATH=/data/megatron-data/indexed/my-gpt2_text_document
# DATA_PATH=/apdcephfs/share_47076/jiaruifang/gpt2_webtext_data/gpt2_bin_data/my-gpt2_text_document
# VOCAB_PATH=/apdcephfs/share_47076/jiaruifang/gpt2_webtext_data/gpt2-vocab.json
# MERGE_PATH=/apdcephfs/share_47076/jiaruifang/gpt2_webtext_data/gpt2-merges.txt
# CHECKPOINT_PATH=checkpoints/gpt2_${MODEL_SIZE}B_ds #checkpoints/gpt2_345m_ds

BASE_DATA_PATH=/workspace/GPT_data
DATA_PATH=${BASE_DATA_PATH}/gpt2_bin_data/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/bpe/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/bpe/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

CHECKPOINT_PATH=checkpoints/gpt2_${MODEL_SIZE}B_ds #checkpoints/gpt2_345m_ds


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="$script_dir/ds_zero_stage_3_config.json"
else
       config_json=$script_dir/`basename $1`
       echo "config json ${config_json}"
fi

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1

PA=true
CC=true
SYNCHRONIZE=true
PROFILE=false


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --train-iters 32 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 2000 \
        --eval-iters 10 \
        --fp16 \
        --cpu-optimizer
"
        # --cpu_torch_adam \
        #--tensorboard-dir ${LOGDIR}
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
echo "PA " ${PA}
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi


full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
