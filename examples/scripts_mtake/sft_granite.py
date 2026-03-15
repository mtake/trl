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

accelerate launch --config_file examples/accelerate_configs/single_gpu.yaml examples/scripts_mtake/sft_granite.py
"""

from datasets import load_dataset
import torch

from trl import SFTConfig, SFTTrainer


def main():
    data_name = "jfe-technical-report_r5"

    # Load dataset
    train_dataset = load_dataset("json", data_files=f"messages_data_{data_name}.jsonl", split="train")

    # Load model
    model_id = "ibm-granite/granite-3.3-8b-instruct"
    # model_id = "ibm-granite/granite-4.0-micro"

    model_id_short = model_id[model_id.rfind("/")+1:]

    # Train model
    training_args = SFTConfig(
        output_dir=f"{model_id_short}__{data_name}",
        # per_device_train_batch_size=1,  # default: 8
        # @@@ahoaho XXX
        per_device_train_batch_size=1,  # default: 8
        # num_train_epochs=1,  # default: 3
        # @@@ahoaho XXX
        num_train_epochs=1,  # default: 3
        # gradient_accumulation_steps=8,  # default: 1
        bf16=True,  # default: None
        # use_liger_kernel=True,
        # @@@ahoaho XXX
        model_init_kwargs={"dtype": torch.bfloat16},
        dataset_num_proc=32,  # default: None
        # @@@ahoaho XXX
        #max_length=8192,
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model_id,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
