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
#DATASET=trl-lib/ultrafeedback_binarized
DATASET=datasets/retriever_call_train_data.granite4_8b.jsonl

# @@@ahoaho XXX
#MODEL=Qwen/Qwen2-0.5B-Instruct
#MODEL=ibm-granite/granite-3.3-8b-instruct
#MODEL=ibm-granite/granite-4.0-micro
#MODEL=ibm-granite/granite-4.0-h-micro
#MODEL=ibm-granite/granite-4.0-h-tiny
#MODEL=ibm-granite/granite-4.0-h-small
MODEL=models/granite-4.0-8b-preview-r260310a

#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_2proc.yaml  # SFT CUDA OOM for g338b, DPO OK for q205b, DPO CUDA OOM for g338b, g4m, DPO CUDA OOM for g338b dtype=bfloat16, DPO OK for g4m, g4hm dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_4proc.yaml  # SFT CUDA OOM for g338b, DPO CUDA OOM for g338b, g4m, DPO CUDA OOM for g338b, g4ht dtype=bfloat16, DPO OK for g4m, g4hm dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_8proc.yaml  # DPO CUDA OOM for g338b, g4m, DPO CUDA OOM for g338b, g4ht dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_2proc.yaml  # DPO CUDA BUSY for q205b
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_2proc.yaml  # SFT OK for g338b, g4m, g4hm, g4ht, DPO CUDA BUSY for q205b, DPO CUDA BUSY for q205b, g4m dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_4proc.yaml  # DPO CUDA BUSY for q205b dtype=bfloat16
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_8proc.yaml  # SFT OK for g4hs
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_2proc.yaml  # SFT CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_2proc.yaml  # SFT CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero2_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_2proc.yaml  # SFT CUDA OOM for g338b, DPO OK for g4m, g4hm, g4ht dtype=bfloat16, DPO CUDA OOM for g48b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1
ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_4proc.yaml  # SFT OK for g338b, DPO OK for q205b, g338b, g4m, g4hm, g4ht dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16, DPO OK for g48b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_8proc.yaml  # DPO OK for g338b dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, DPO OK for g4hs offload_optimizer_device=cpu offload_param_device=cpu dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_8proc_offload.yaml  # DPO OK for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1

ACCELERATE_OPT=""
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 2"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 4"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 8"
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_optimizer_device cpu"  # for zero_stage>=2  # DPO OK for g4hs
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_param_device cpu"  # for zero_stage>=3  # DPO OK for g4hs

OUTPUT_DIR="trainer_output/${MODEL##*/}-${DATASET##*/}-dpo-${START_TIME_STR}-${HOSTNAME_S}"

# @@@ahoaho XXX
echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}


# See https://github.com/mtake/trl/blob/main/trl/scripts/dpo.py
cmd="${ENV}accelerate launch --config_file ${ACCELERATE_CONFIG}${ACCELERATE_OPT} ${BASENAME}.py --dataset_name ${DATASET} --model_name_or_path ${MODEL}"
# @@@ahoaho XXX
cmd="$cmd --dataset_num_proc 8"
cmd="$cmd --dtype bfloat16"
cmd="$cmd --bf16 True"
#cmd="$cmd --learning_rate 5.0e-7"
#cmd="$cmd --num_train_epochs 1"  # default is 3
####cmd="$cmd --per_device_train_batch_size 2"  # default is 8
cmd="$cmd --per_device_train_batch_size 1"  # default is 8  # DPO OK for g4hs
####cmd="$cmd --max_steps 10"  # default is -1 (len(train split) * num_train_epochs)
####cmd="$cmd --gradient_accumulation_steps 8"
cmd="$cmd --gradient_accumulation_steps 1"  # default is 1  # DPO OK for g4hs
# @@@ahoaho XXX
#cmd="$cmd --eval_strategy no"
cmd="$cmd --eval_strategy steps"
cmd="$cmd --eval_steps 50"
cmd="$cmd --output_dir ${OUTPUT_DIR}"
# @@@ahoaho XXX
#cmd="$cmd --no_remove_unused_columns"
# @@@ahoaho XXX
####cmd="$cmd --max_length 512"  # default is 1024
####cmd="$cmd --max_length 256"  # default is 1024
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
