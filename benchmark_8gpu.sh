cpu_size=1024
DIRNAME=colossal_ai_compare_${cpu_size}
mkdir -p ${DIRNAME} 
export MP_SIZE=${MP_SIZE:-1}
export GPU_NUM=${GPU_NUM:-1}
export PA_CPU=false

export GPU_NUM=8
for MP_SIZE in 1 
do
for MODEL_SIZE in 4 
do
for BS in 1 2 
do
echo "runing ${MODEL_SIZE} ${GPU_NUM} ${BS}"
env PA_CPU=${PA_CPU} NUM_GPUS_PER_WORKER=${GPU_NUM} MP_SIZE=${MP_SIZE} MODEL_SIZE=${MODEL_SIZE} BS=${BS} bash examples/ds_pretrain_gpt2-zero3.sh 2>&1 | tee ${DIRNAME}/log.model_${MODEL_SIZE}B_bs_${BS}_gpu_${GPU_NUM}_mp_${MP_SIZE}_PACPU_${PA_CPU}
done
done
done
