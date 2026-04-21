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

if true; then
ENV="CUDA_LAUNCH_BLOCKING=1 ${ENV}"
ENV="TORCH_USE_CUDA_DSA=1 ${ENV}"
fi

if false; then
ENV="TORCH_CPP_LOG_LEVEL=INFO ${ENV}"
ENV="TORCH_DISTRIBUTED_DEBUG=DETAIL ${ENV}"

ENV="NCCL_DEBUG_SUBSYS=ALL ${ENV}"

ENV="NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"  # deprecated

ENV="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"

#ENV="NCCL_P2P_DISABLE=1 ${ENV}"
#ENV="NCCL_SHM_DISABLE=1 ${ENV}"
#ENV="NCCL_IB_DISABLE=1 ${ENV}"
fi

# @@@ahoaho XXX
#DATASET=datasets/messages_data__jfe-technical-report_r5.jsonl
#DATASET=datasets/retriever_call_train_data.granite4_8b.v2.0406-SFT.jsonl
#DATASET=jfe.yaml
DATASET=zragrtrvr.v2-SFT.yaml

DATASET_S="${DATASET##*/}"
if [[ "${DATASET_S}" == *.yaml ]]; then
    DATASET_S="${DATASET_S%.yaml}"
    USE_CONFIG=1
elif [[ "${DATASET_S}" == *.jsonl ]]; then
    DATASET_S="${DATASET_S%.jsonl}"
elif [[ "${DATASET_S}" == *.json ]]; then
    DATASET_S="${DATASET_S%.json}"
fi

# @@@ahoaho XXX
#MODEL=Qwen/Qwen2-0.5B-Instruct
#MODEL=Qwen/Qwen3-0.6B
#MODEL=ibm-granite/granite-3.3-8b-instruct
#MODEL=ibm-granite/granite-4.0-micro
#MODEL=ibm-granite/granite-4.0-h-micro
#MODEL=ibm-granite/granite-4.0-h-tiny
####MODEL=ibm-granite/granite-4.0-h-small  # may cause OSError due to slow system system
####MODEL=models/granite-4.0-h-small
MODEL=models/granite-4.1-8b
#MODEL=models/granite-4.1-30b

#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_2proc.yaml  # SFT(jfe) CUDA OOM for g338b, DPO(rtrvr) OK for q205b, DPO(rtrvr) CUDA OOM for g338b, g4m, DPO(rtrvr) CUDA OOM for g338b dtype=bfloat16, DPO(rtrvr) OK for g4m, g4hm dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_4proc.yaml  # SFT(jfe) CUDA OOM for g338b, DPO(rtrvr) CUDA OOM for g338b, g4m, DPO(rtrvr) CUDA OOM for g338b, g4ht dtype=bfloat16, DPO(rtrvr) OK for g4m, g4hm dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_8proc.yaml  # DPO(rtrvr) CUDA OOM for g338b, g4m, DPO(rtrvr) CUDA OOM for g338b, g4ht dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_2proc.yaml  # DPO(rtrvr) CUDA BUSY for q205b
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_2proc.yaml  # SFT(jfe) OK for g338b, g4m, g4hm, g4ht, DPO(rtrvr) CUDA BUSY for q205b, DPO(rtrvr) CUDA BUSY for q205b, g4m dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_4proc.yaml  # DPO(rtrvr) CUDA BUSY for q205b dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_8proc.yaml  # SFT(jfe) OK for g4hs
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_2proc.yaml  # SFT(jfe) CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_2proc.yaml  # SFT(jfe) CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero3_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero3_1node_2proc.yaml  # SFT(jfe) CUDA OOM for g338b, DPO(rtrvr) OK for g4m, g4hm, g4ht dtype=bfloat16, DPO(rtrvr) CUDA OOM for g418b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1
ACCELERATE_CONFIG=accelerate_configs/zero3_1node_4proc.yaml  # SFT(jfe) OK for g338b, DPO(rtrvr) OK for q205b, g338b, g4m, g4hm, g4ht dtype=bfloat16, DPO(rtrvr) CUDA OOM for g4hs dtype=bfloat16, DPO(rtrvr) OK for g418b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, SFT(rtrvr.v2) CUDA OOM for g418b dtype=bfloat16, per_device_train_batch_size=16, max_length=8192, SFT(rtrvr.v2) OK for g418b dtype=bfloat16, per_device_train_batch_size=8, max_length=8192
#ACCELERATE_CONFIG=accelerate_configs/zero3_1node_8proc.yaml  # DPO(rtrvr) OK for g338b dtype=bfloat16, DPO(rtrvr) CUDA OOM for g4hs dtype=bfloat16, DPO(rtrvr) CUDA OOM for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, DPO(rtrvr) OK for g4hs offload_optimizer_device=cpu offload_param_device=cpu dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, SFT(rtrvr.v2) OK for g4130b dtype=bfloat16, per_device_train_batch_size=4, max_length=8192
####ACCELERATE_CONFIG=accelerate_configs/zero3_1node_8proc_offload.yaml  # DPO(rtrvr) OK for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1

ACCELERATE_OPT=""
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 2"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 4"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 8"
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_optimizer_device cpu"  # for zero_stage>=2  # DPO(rtrvr) OK for g4hs
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_param_device cpu"  # for zero_stage>=3  # DPO(rtrvr) OK for g4hs

#OUTPUT_DIR="trainer_output/${MODEL##*/}-${DATASET_S}-sft-${START_TIME_STR}-${HOSTNAME_S}"  # NOTE neither timestamp nor hostname works with preemptable queue
OUTPUT_DIR="trainer_output/${MODEL##*/}-${DATASET_S}-sft"  # NOTE neither timestamp nor hostname works with preemptable queue

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}


# See https://github.com/mtake/trl/blob/main/trl/scripts/sft.py
cmd="${ENV}accelerate launch --config_file ${ACCELERATE_CONFIG}${ACCELERATE_OPT} ${BASENAME}.py --model_name_or_path ${MODEL}"
if [[ -n "${USE_CONFIG}" ]]; then
    cmd="$cmd --config ${DATASET}"
else
    cmd="$cmd --dataset_name ${DATASET}"
fi
cmd="$cmd --output_dir ${OUTPUT_DIR}"  # default: trainer_output
####cmd="$cmd --per_device_train_batch_size 32"  # default: 8  # SFT(jfe) OK for g338b, g4m, g4hm, g4ht, g418b, g4130b
#cmd="$cmd --per_device_train_batch_size 16"  # default: 8  # SFT(jfe) OK for g4hs, SFT(rtrve.v2) CUDA OOM for g418b per_device_train_batch_size=16, max_length=8192
cmd="$cmd --per_device_train_batch_size 8"  # default: 8  # SFT(rtrve.v2) OK for g418b per_device_train_batch_size=8, max_length=8192, SFT(rtrve.v2) CUDA OOM for g4130b per_device_train_batch_size=8, max_length=8192
#cmd="$cmd --per_device_train_batch_size 4"  # default: 8  # SFT(rtrve.v2) OK for g4130b per_device_train_batch_size=4, max_length=8192
#cmd="$cmd --num_train_epochs 1"  # default: 3
cmd="$cmd --save_strategy epoch"  # default: steps
####cmd="$cmd --max_steps 10"  # default: -1 (len(train split) * num_train_epochs)
#cmd="$cmd --save_steps 100"  # default: 500. an integer as steps or a float in range `[0,1)` as ratio of total training steps.
#cmd="$cmd --gradient_accumulation_steps 8"  # default: 1
#cmd="$cmd --learning_rate 5.0e-7"  # default: 2e-05
#cmd="$cmd --use_liger_kernel True"  # default: False
cmd="$cmd --dtype bfloat16"
cmd="$cmd --bf16 True"  # default: None
cmd="$cmd --dataset_num_proc 8"  # default: None
#cmd="$cmd --no_remove_unused_columns"  # default: False
####cmd="$cmd --max_length 20000"  # default: 1024  # SFT(jfe) OK
cmd="$cmd --max_length 8192"  # default: 1024  # SFT(rtrve.v2) OK for g418b per_device_train_batch_size=8, max_length=8192, SFT(rtrve.v2) OK for g4130b per_device_train_batch_size=4, max_length=8192
#cmd="$cmd --eval_strategy steps"  # default: no  # requires test split
#cmd="$cmd --eval_steps 50"
#cmd="$cmd --per_device_eval_batch_size 1"  # default: 8
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
