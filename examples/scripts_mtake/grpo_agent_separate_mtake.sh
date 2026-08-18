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

# @@@ahoaho XXX ???
# TOOLS="query_biogrid"
TOOLS="grpo_agent_separate_lib.query_biogrid"

# @@@ahoaho XXX ???
# REWARD_FUNCS="correctness_reward structure_reward query_reward"
REWARD_FUNCS="grpo_agent_separate_lib.correctness_reward grpo_agent_separate_lib.structure_reward grpo_agent_separate_lib.query_reward"

# @@@ahoaho XXX
#MODEL=Qwen/Qwen2-0.5B-Instruct
####MODEL=Qwen/Qwen3-0.6B  # GRPO AGENT OK
#MODEL=ibm-granite/granite-3.3-8b-instruct
#MODEL=ibm-granite/granite-4.0-micro
#MODEL=ibm-granite/granite-4.0-h-micro
#MODEL=ibm-granite/granite-4.0-h-tiny
####MODEL=ibm-granite/granite-4.0-h-small  # may cause OSError due to slow system system
####MODEL=models/granite-4.0-h-small
#MODEL=ibm-granite/granite-4.1-3b  # GRPO AGENT OK for g413b use_vllm=True (see trl/chat_template_utils.py)
#MODEL=ibm-granite/granite-4.1-8b  # GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
####MODEL=trainer_output/granite-4.1-8b-retriever_call_train_data.granite4_8b.v2.0406-SFT-sft-20260409-150354-p4-r26-n3-g418b-3epochs-8192length-rtrvr.v2
####MODEL=trainer_output/granite-4.1-8b-retriever_call_train_data.granite4_8b.v2.0406-SFT-sft-20260416-100514-p1-r23-n3-g418b-3epochs-8192length-rtrvr.v2-transformers4576
#MODEL=models/granite-4.1-30b
#MODEL=trainer_output/granite-4.1-30b-zragrtrvr.v2-SFT-sft-20260421-134025-p6-r14-n1-g4130b-3epochs-8192length-rtrvr.v2
MODEL=ibm-research/granite-4.2-3b  # GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
#MODEL=ibm-research/granite-4.2-8b  # GRPO OK? for g428b(preemptable) use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5 save_strategy=steps save_steps=50 resume_from_checkpoint=True, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5

# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/examples/scripts_mtake/grpo_agent_mtake.py", line 340, in <module>
# [rank2]:     trainer.train()
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/transformers/trainer.py", line 1437, in train
# [rank2]:     return inner_training_loop(
# [rank2]:            ^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/transformers/trainer.py", line 1519, in _inner_training_loop
# [rank2]:     self._run_epoch(
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/transformers/trainer.py", line 1747, in _run_epoch
# [rank2]:     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
# [rank2]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/trainer/grpo_trainer.py", line 1549, in training_step
# [rank2]:     output = super().training_step(model, inputs, num_items_in_batch)
# [rank2]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/transformers/trainer.py", line 1913, in training_step
# [rank2]:     inputs = self._prepare_inputs(inputs)
# [rank2]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/extras/profiling.py", line 211, in wrapper
# [rank2]:     return func(self, *args, **kwargs)
# [rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/trainer/grpo_trainer.py", line 1578, in _prepare_inputs
# [rank2]:     generation_batch = self._generate_and_score_completions(generation_batch)
# [rank2]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/trainer/grpo_trainer.py", line 2689, in _generate_and_score_completions
# [rank2]:     rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
# [rank2]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/extras/profiling.py", line 211, in wrapper
# [rank2]:     return func(self, *args, **kwargs)
# [rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/trl/trainer/grpo_trainer.py", line 1663, in _calculate_rewards
# [rank2]:     output_reward_func = reward_func(
# [rank2]:                          ^^^^^^^^^^^^
# [rank2]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/examples/scripts_mtake/grpo_agent_mtake.py", line 128, in correctness_reward
# [rank2]:     raw = completion[-1]["content"].lower()
# [rank2]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank2]: AttributeError: 'list' object has no attribute 'lower'

MODEL_S="${MODEL##*/}"
MODEL_S="${MODEL_S//./_}"

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
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_2proc.yaml  # SFT CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero1_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_2proc.yaml  # SFT CUDA OOM
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_4proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero2_1node_8proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero3_1node_1proc.yaml
#ACCELERATE_CONFIG=accelerate_configs/zero3_1node_2proc.yaml  # SFT CUDA OOM for g338b, DPO OK for g4m, g4hm, g4ht dtype=bfloat16, DPO CUDA OOM for g418b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1
ACCELERATE_CONFIG=accelerate_configs/zero3_1node_4proc.yaml  # SFT OK for g338b, DPO OK for q205b, g338b, g4m, g4hm, g4ht dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16, DPO OK for g418b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, DPO CUDA OOM for g4130b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, GRPO AGENT OK for q306b, GRPO AGENT OK for g413b use_vllm=True, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
####ACCELERATE_CONFIG=accelerate_configs/zero3_1node_8proc.yaml  # DPO OK for g338b dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16, DPO CUDA OOM for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, DPO OK for g4hs offload_optimizer_device=cpu offload_param_device=cpu dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, DPO OK for g4130b dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1, GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
####ACCELERATE_CONFIG=accelerate_configs/zero3_1node_8proc_offload.yaml  # DPO OK for g4hs dtype=bfloat16 per_device_train_batch_size=1 gradient_accumulation_steps=1

ACCELERATE_OPT=""
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 2"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 4"
#ACCELERATE_OPT="${ACCELERATE_OPT} --num_processes 8"
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_optimizer_device cpu"  # for zero_stage>=2  # DPO OK for g4hs
#ACCELERATE_OPT="${ACCELERATE_OPT} --offload_param_device cpu"  # for zero_stage>=3  # DPO OK for g4hs

#OUTPUT_DIR="trainer_output/${MODEL##*/}${_DATASET_S}-grpo_agent_separate-${START_TIME_STR}-${HOSTNAME_S}"  # NOTE neither timestamp nor hostname works with preemptable queue
OUTPUT_DIR="trainer_output/${MODEL##*/}${_DATASET_S}-grpo_agent_separate"  # NOTE neither timestamp nor hostname works with preemptable queue

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}


# See https://github.com/mtake/trl/blob/main/examples/scripts/grpo_agent.py
cmd="${ENV}accelerate launch --config_file ${ACCELERATE_CONFIG}${ACCELERATE_OPT} ${BASENAME}.py --model_name_or_path ${MODEL}"

# @@@ahoaho XXX ???
# if [[ -n "${USE_CONFIG}" ]]; then
#     cmd="$cmd --config ${DATASET}"
# else
#     cmd="$cmd --dataset_name ${DATASET}"
# fi

cmd="$cmd --output_dir ${OUTPUT_DIR}"

# @@@ahoaho XXX ???
if [[ -n "${TOOLS}" ]]; then
    cmd="$cmd --tools ${TOOLS}"
fi

# @@@ahoaho XXX ???
if [[ -n "${REWARD_FUNCS}" ]]; then
    cmd="$cmd --reward_funcs ${REWARD_FUNCS}"
fi

# @@@ahoaho XXX
#cmd="$cmd --num_generations 4"  # default: 8. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.
#cmd="$cmd --num_generations 8"  # default: 8. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.  # GRPO OK? for g418b num_processes=4 per_device_train_batch_size=2 num_generations=8, GRPO OK? for q306b num_processes=4 per_device_train_batch_size=4 num_generations=8, GRPO OK? for q306b num_processes=4 per_device_train_batch_size=8 num_generations=8, GRPO AGENT OK for q306b num_processes=4 per_device_train_batch_size=8 num_generations=8, GRPO AGENT OK for g413b use_vllm=True, GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
# @@@ahoaho XXX
#cmd="$cmd --per_device_train_batch_size 1"  # default: 8  # DPO OK for g4hs
#cmd="$cmd --per_device_train_batch_size 2"  # default: 8  # GRPO OK? for g418b num_processes=4 per_device_train_batch_size=2 num_generations=8
#cmd="$cmd --per_device_train_batch_size 4"  # default: 8  # GRPO OK? for q306b num_processes=4 per_device_train_batch_size=4 num_generations=8
#cmd="$cmd --per_device_train_batch_size 8"  # default: 8  # GRPO OK? for q306b num_processes=4 per_device_train_batch_size=8 num_generations=8, GRPO AGENT OK for g413b use_vllm=True, GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
#cmd="$cmd --num_train_epochs 1"  # default: 3
# @@@ahoaho XXX WIP for functional test
####cmd="$cmd --save_strategy epoch"  # default: steps
cmd="$cmd --save_strategy steps"  # default: steps
####cmd="$cmd --max_steps 30"  # default: -1 (len(train split) * num_train_epochs)
# @@@ahoaho XXX WIP for functional test
cmd="$cmd --max_steps 30"  # default: -1 (len(train split) * num_train_epochs)
#cmd="$cmd --save_steps 100"  # default: 500. an integer as steps or a float in range `[0,1)` as ratio of total training steps.
# @@@ahoaho XXX WIP for functional test
cmd="$cmd --save_steps 10"  # default: 500. an integer as steps or a float in range `[0,1)` as ratio of total training steps.
# @@@ahoaho XXX NOT TESTED
# cmd="$cmd --resume_from_checkpoint True"  # default: None  # GRPO OK for g428bpre(preemptable) save_strategy=steps save_steps=50 resume_from_checkpoint=True, GRPO OK? for g413b(preemptable) save_strategy=steps save_steps=50 resume_from_checkpoint=True use_vllm=True, GRPO OK? for g428b(preemptable) use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5 save_strategy=steps save_steps=50 resume_from_checkpoint=True
#cmd="$cmd --gradient_accumulation_steps 8"  # default: 1  # DPO OK for g4hs
# @@@ahoaho XXX
#cmd="$cmd --logging_strategy epoch"  # default: steps, choices: [no, steps, epoch]
#cmd="$cmd --logging_steps 10"  # default: 10. an integer as steps or a float in range `[0,1)` as ` as ratio of total training steps.
cmd="$cmd --log_completions True"  # default: False
cmd="$cmd --num_completions_to_print 10"  # default: None (means all)
cmd="$cmd --report_to trackio"  # default: none, choices: [none, all, trackio, wandb]
# * Trackio project initialized: grpo_agent_mtake
# * Trackio metrics logged to: /u/mtake/.cache/huggingface/trackio
# * View dashboard by running in your terminal: trackio show --project "grpo_agent_mtake"
# * or by running in Python: trackio.show(project="grpo_agent_mtake")
# * NVIDIA GPU detected, enabling automatic GPU metrics logging
# * psutil detected, enabling automatic CPU/system metrics logging
# * Trackio directory /u/mtake/.cache/huggingface/trackio appears to be on a network filesystem: logging via append-only JSONL fragments instead of direct SQLite writes. Set TRACKIO_STORAGE_MODE=sqlite to override.
# * Created new run: brave-forest-1
cmd="$cmd --project ${BASENAME}"  # default: huggingface. The name of the project to use for logging. Currently, only used by Trackio.
cmd="$cmd --run_name ${MODEL_S}"  # default: None. A descriptor for the run. Typically used for trackio, wandb, etc.
#cmd="$cmd --learning_rate 5.0e-7"  # default: 1e-06
#cmd="$cmd --use_liger_kernel True"  # default: False
cmd="$cmd --dtype bfloat16"
cmd="$cmd --bf16 True"
#cmd="$cmd --no_remove_unused_columns"  # default: False
#cmd="$cmd --eval_strategy steps"  # default: no  # requires test split
#cmd="$cmd --eval_steps 50"
#cmd="$cmd --per_device_eval_batch_size 1"  # default: 8
cmd="$cmd --use_vllm True"  # default: False (see. trl/trainer/grpo_config.py)  # GRPO OK for q306b num_processes=4 per_device_train_batch_size=8 num_generations=8 `bsub -gpu num=4:mode=shared:j_exclusive=yes:gmodel=XXX ...` (NOTE: `bsub -gpu num=4:mode=exclusive_process:gmodel=XXX ...` causes CUDA warning: CUDA-capable device(s) is/are busy or unavailable (function destroyEvent)), GRPO OK? for g413b(preemptable) save_strategy=steps save_steps=50 resume_from_checkpoint=True use_vllm=True, GRPO AGENT OK for g413b use_vllm=True, GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
cmd="$cmd --vllm_gpu_memory_utilization 0.5"  # default: 0.3  # GRPO AGENT OK for g418b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g428b use_vllm=True num_processes=8 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5, GRPO AGENT OK for g423b use_vllm=True num_processes=4 per_device_train_batch_size=8 num_generations=8 vllm_gpu_memory_utilization=0.5
#cmd="$cmd --vllm_max_model_length 65536"  # default: model's max seq len (see "max_position_embeddings" value in config.json)
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
