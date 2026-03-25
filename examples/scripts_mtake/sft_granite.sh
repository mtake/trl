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

#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_2proc.yaml  # CUDA OOM for g338b
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_4proc.yaml  # CUDA OOM for g338b
#ACCELERATE_CONFIG=accelerate_configs/multi_gpu_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_2proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_1proc.yaml
ACCELERATE_CONFIG=accelerate_configs/fsdp2_1node_2proc.yaml  # OK for g338b, g4m, g4hm, g4ht
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
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_4proc.yaml  # OK for g338b
#ACCELERATE_CONFIG=accelerate_configs/deepspeed_zero3_1node_8proc.yaml

cmd="${ENV}accelerate launch --config_file ${ACCELERATE_CONFIG} ${BASENAME}.py"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
