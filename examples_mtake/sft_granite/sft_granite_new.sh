#!/usr/bin/env bash

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"

cleanup() {
    echo "XXX SIGNAL_RECEIVED" | tee -a ${LOGFILE}
    END_TIME="$(${DATE_CMD} +%s)"
    END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
    echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
    echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
    exit 1
}
trap cleanup INT TERM HUP

echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

VENV=../../.venv
if [[ -d "${VENV}" ]]; then
    source "${VENV}/bin/activate"
fi

ENV=""
ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"  # deprecated
ENV="PYTORCH_ALLOC_CONF=expandable_segments:True ${ENV}"
ENV="NCCL_DEBUG=INFO ${ENV}"

if false; then
ENV="TORCH_CPP_LOG_LEVEL=INFO ${ENV}"
ENV="TORCH_DISTRIBUTED_DEBUG=DETAIL ${ENV}"

ENV="NCCL_DEBUG_SUBSYS=ALL ${ENV}"
#ENV="CUDA_LAUNCH_BLOCKING=1 ${ENV}"
#ENV="TORCH_USE_CUDA_DSA=1 ${ENV}"

ENV="NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"  # deprecated
ENV="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"

#ENV="NCCL_P2P_DISABLE=1 ${ENV}"
#ENV="NCCL_SHM_DISABLE=1 ${ENV}"
#ENV="NCCL_IB_DISABLE=1 ${ENV}"
fi

# @@@ahoaho XXX ???
# # @@@ahoaho XXX
# DATASET=trl-lib/DeepMath-103K

# DATASET_S="${DATASET##*/}"
# if [[ "${DATASET_S}" == *.yaml ]]; then
#     DATASET_S="${DATASET_S%.yaml}"
#     USE_CONFIG=1
# elif [[ "${DATASET_S}" == *.jsonl ]]; then
#     DATASET_S="${DATASET_S%.jsonl}"
# elif [[ "${DATASET_S}" == *.json ]]; then
#     DATASET_S="${DATASET_S%.json}"
# fi

if [[ -n "${DATASET_S}" ]]; then
    _DATASET_S="-${DATASET_S}"
else
    _DATASET_S=""
fi

#MODEL=ibm-granite/granite-3.3-8b-instruct  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
#MODEL=ibm-granite/granite-4.0-micro  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
#MODEL=ibm-granite/granite-4.0-h-micro  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
#MODEL=ibm-granite/granite-4.0-h-tiny  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
#MODEL=ibm-granite/granite-4.0-h-small  # OK with per_device_train_batch_size=16, max_length=20000, fsdp2_1node_8proc.yaml  # CUDA OOM with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_8proc.yaml
#MODEL=ibm-granite/granite-4.1-8b  # OK with per_device_train_batch_size=32, max_length=20000, deepspeed_zero3_1node_4proc.yaml
MODEL=ibm-granite/granite-4.2-8b  # OK with per_device_train_batch_size=32, max_length=20000, deepspeed_zero3_1node_4proc.yaml
#MODEL=models/granite-4.1-30b  # OK with per_device_train_batch_size=32, max_length=20000, deepspeed_zero3_1node_8proc_offload.yaml

MODEL_S="${MODEL##*/}"
MODEL_S="${MODEL_S//./_}"

#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_2proc.yaml  # CUDA OOM for g338b
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_4proc.yaml  # CUDA OOM for g338b
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_2proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_1proc.yaml
####ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_2proc.yaml  # OK for g338b, g4m, g4hm, g4ht, ERR for g418b
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_8proc.yaml  # OK for g4hs
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_2proc.yaml  # CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_2proc.yaml  # CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_2proc.yaml  # CUDA OOM for g338b
ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_4proc.yaml  # OK for g338b, g418b, g428b
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_8proc.yaml
# ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_8proc_offload.yaml  # OK g4130b

ACCELERATE_OPT=""
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 2"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 4"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 8"
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_optimizer_device cpu"  # for zero_stage>=2  # DPO OK for g4hs
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_param_device cpu"  # for zero_stage>=3  # DPO OK for g4hs

#OUTPUT_DIR="trainer_output/${MODEL##*/}${_DATASET_S}-sft_granite_new-${START_TIME_STR}-${HOSTNAME_S}"  # default: trainer_output, # NOTE neither timestamp nor hostname works with preemptable queue
OUTPUT_DIR="trainer_output/${MODEL##*/}${_DATASET_S}-sft_granite_new"  # default: trainer_output, # NOTE neither timestamp nor hostname works with preemptable queue

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}

cmd="${ENV}accelerate launch --config_file ${ACCELERATE_CONFIG}${ACCELERATE_OPT} ${BASENAME}.py --model_name_or_path ${MODEL}"
cmd="$cmd --output_dir ${OUTPUT_DIR}"
cmd="$cmd --bf16 True"  # default: None
# cmd="$cmd --use_liger_kernel True"
cmd="$cmd --max_length 20000"  # default: 1024
cmd="$cmd --per_device_train_batch_size 32"  # default: 8, # 32 OK for g338b, g4m, g4hm, g4ht, g418b, g4130b, # 16 OK for g4hs
# @@@ahoaho XXX NOT TESTED
# cmd="$cmd --resume_from_checkpoint True"  # default: None  # GRPO OK for g428bpre(preemptable) save_strategy=steps save_steps=50 resume_from_checkpoint=True, GRPO OK? for g413b(preemptable) save_strategy=steps save_steps=50 resume_from_checkpoint=True use_vllm=True, GRPO OK? for g428b(preemptable) use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5 save_strategy=steps save_steps=50 resume_from_checkpoint=True
# cmd="$cmd --gradient_accumulation_steps 8"  # default: 1
cmd="$cmd --dataset_num_proc 8"  # default: None
# cmd="$cmd --num_train_epochs 1"  # default: 3
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
