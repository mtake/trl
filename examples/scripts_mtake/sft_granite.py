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

"""
Train Granite on a dataset.

accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml examples/scripts_mtake/sft_granite.py
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
    train_dataset = load_dataset("json", data_files=f"messages_data{_data_name}.jsonl", split="train")

    # Load model
    # model_id = "ibm-granite/granite-3.3-8b-instruct"
    # model_id = "ibm-granite/granite-4.0-micro"
    # model_id = "ibm-granite/granite-4.0-h-micro"
    model_id = "ibm-granite/granite-4.0-h-tiny"

    model_id_short = model_id[model_id.rfind("/")+1:]

    output_prefix = "trainer_output"
    os.makedirs(output_prefix, exist_ok=True)
    output_dir = f"{output_prefix}/{model_id_short}{_data_name}{_timestamp}"

    # Train model
    training_args = SFTConfig(
        output_dir=output_dir,  # default: trainer_output
        per_device_train_batch_size=32,  # default: 8 OK with max_length=20000 fsdp2_1node_2proc.yaml
        # num_train_epochs=1,  # default: 3
        # gradient_accumulation_steps=8,  # default: 1
        bf16=True,  # default: None
        # use_liger_kernel=True,
        dataset_num_proc=8,  # default: None
        max_length=20000,  # default: 1024
    )

    trainer = SFTTrainer(
        model=model_id,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
