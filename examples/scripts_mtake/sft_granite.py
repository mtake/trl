# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "Pillow",
#     "trackio",
#     "kernels",
# ]
# ///

#
# NOTE: this code is based on examples/scripts/sft_gemma3.py
#

"""
Train Granite on the JFE Technical Report dataset.

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts_mtake/sft_granite.py
"""

from datasets import load_dataset
from datetime import datetime
import os

from trl import SFTConfig, SFTTrainer


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _timestamp = f"__{timestamp}"

    data_name = "jfe-technical-report_r5"
    _data_name = f"__{data_name}" if data_name is not None and len(data_name) > 0 else ""

    # Load dataset
    train_dataset = load_dataset("json", data_files=f"datasets/messages_data{_data_name}.jsonl", split="train")

    # Load model
    # model_id = "ibm-granite/granite-3.3-8b-instruct"  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
    # model_id = "ibm-granite/granite-4.0-micro"  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
    # model_id = "ibm-granite/granite-4.0-h-micro"  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
    # model_id = "ibm-granite/granite-4.0-h-tiny"  # OK with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_2proc.yaml
    # model_id = "ibm-granite/granite-4.0-h-small"  # OK with per_device_train_batch_size=16, max_length=20000, fsdp2_1node_8proc.yaml  # CUDA OOM with per_device_train_batch_size=32, max_length=20000, fsdp2_1node_8proc.yaml
    # model_id = "models/granite-4.1-8b"  # OK with per_device_train_batch_size=32, max_length=20000, deepspeed_zero3_1node_4proc.yaml
    model_id = "models/granite-4.1-30b"  # OK with per_device_train_batch_size=32, max_length=20000, deepspeed_zero3_1node_8proc_offload.yaml

    model_id_short = model_id[model_id.rfind("/")+1:]

    output_prefix = "trainer_output"
    os.makedirs(output_prefix, exist_ok=True)
    # output_dir = f"{output_prefix}/{model_id_short}{_data_name}{_timestamp}"  # NOTE neither timestamp nor hostname works with preemptable queue
    output_dir = f"{output_prefix}/{model_id_short}{_data_name}"  # NOTE neither timestamp nor hostname works with preemptable queue

    # Train model
    training_args = SFTConfig(
        output_dir=output_dir,  # default: trainer_output
        bf16=True,  # default: None
        # use_liger_kernel=True,
        max_length=20000,  # default: 1024
        per_device_train_batch_size=32,  # default: 8  # OK for g338b, g4m, g4hm, g4ht, g418b, g4130b
        # per_device_train_batch_size=16,  # default: 8  # OK for g4hs
        # gradient_accumulation_steps=8,  # default: 1
        dataset_num_proc=8,  # default: None
        # num_train_epochs=1,  # default: 3
    )

    trainer = SFTTrainer(
        args=training_args,
        model=model_id,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
