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
import os
import sys

from trl import SFTConfig, SFTTrainer, ModelConfig, TrlParser

def main():
    parser = TrlParser((SFTConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()

    data_name = "jfe-technical-report_r5"
    _data_name = f"__{data_name}" if data_name is not None and len(data_name) > 0 else ""

    # Load dataset
    train_dataset = load_dataset("json", data_files=f"datasets/messages_data{_data_name}.jsonl", split="train")

    # Train model
    trainer = SFTTrainer(
        args=training_args,
        model=model_args.model_name_or_path,
        train_dataset=train_dataset,
    )
    # @@@ahoaho XXX
    # trainer.train()
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() in ["true", "yes", "1"]:
        resume_from_checkpoint = True
    if resume_from_checkpoint is True:
        sys.path.insert(0, os.path.dirname(__file__))
        from utils_mtake import get_last_checkpoint_safe
        resume_from_checkpoint = get_last_checkpoint_safe(training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    main()
