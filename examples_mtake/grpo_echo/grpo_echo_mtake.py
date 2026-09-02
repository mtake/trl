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
#     "openenv-echo-env @ git+https://huggingface.co/spaces/qgallouedec/echo_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Echo environment. The environment echoes back the message
sent to it and rewards longer completions.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/qgallouedec/echo_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/echo_env
uv pip install -e .
```

Usage:

```sh
python examples/grpo_echo/grpo_echo.py
python examples/grpo_echo/grpo_echo.py --model Qwen/Qwen2.5-0.5B-Instruct --env-host https://qgallouedec-echo-env.hf.space
```
"""

import argparse
# @@@ahoaho XXX
import os
import sys

# @@@ahoaho XXX
from dataclasses import dataclass, field
from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

# @@@ahoaho XXX
# from trl import GRPOConfig, GRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser


# @@@ahoaho XXX
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run GRPO training with Echo environment.")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="Qwen/Qwen3-0.6B",
#         help="Model to use for training.",
#     )
#     parser.add_argument(
#         "--env-host",
#         type=str,
#         default="https://qgallouedec-echo-env.hf.space",
#         help="URL for the Echo environment HF Space.",
#     )
#     return parser.parse_args()


@dataclass
class GRPOEchoScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        env_host (`str`, *optional*):
            URL for the Echo environment HF Space. Default is "https://qgallouedec-echo-env.hf.space".
        # tools (`list[str]`, *optional*):
        #     Available tools. Supported values are:
        #         - `"query_biogrid"`
        #         - any dotted import path " (e.g., `'my_lib.tools.custom_tool'`).
        # reward_model_name_or_path (`str`, *optional*):
        #     Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
        #     directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        # reward_funcs (`list[str]`, *optional*):
        #     Reward functions to use. Supported values are:
        #         - `"correctness_reward"`
        #         - `"structure_reward"`
        #         - `"query_reward"`
        #         - any dotted import path " (e.g., `'my_lib.rewards.custom_reward'`).
    """

    env_host: str | None = field(
        default="https://qgallouedec-echo-env.hf.space",
        metadata={
            "help": "URL for the Echo environment HF Space."
        },
    )
    # tools: list[str] | None = field(
    #     default=None,
    #     metadata={
    #         "help": "Available tools. Supported values are: `query_biogrid`, or "
    #         "any dotted import path (e.g., `'my_lib.tools.custom_tool'`)."
    #     },
    # )
    # reward_model_name_or_path: str | None = field(
    #     default=None,
    #     metadata={
    #         "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
    #         "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
    #     },
    # )
    # reward_funcs: list[str] | None = field(
    #     default=None,
    #     metadata={
    #         "help": "Reward functions to use. Supported values are: `correctness_reward`, `structure_reward`, `query_reward`, or "
    #         "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
    #     },
    # )


def reward_func(environments, **kwargs):
    return [env.reward for env in environments]


def main():
    # @@@ahoaho XXX
    # args = parse_args()
    parser = TrlParser((GRPOEchoScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    training_args.chat_template_kwargs = {"enable_thinking": False}

    # @@@ahoaho XXX
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"XXX script_args: {script_args} XXX")
        print(f"XXX training_args: {training_args} XXX")
        print(f"XXX model_args: {model_args} XXX")

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "Try to echo 'Hello World!' in the environment."}],
                [{"role": "user", "content": "Make the environment echo 'Goodbye World!'"}],
                [{"role": "user", "content": "Can you ask the environment to echo 'TRL is great!'?"}],
                [{"role": "user", "content": "What happens if you ask the environment to echo 'I love RLHF!'?"}],
                [{"role": "user", "content": "Try to make the environment echo 'OpenEnv is awesome!'"}],
            ],
        }
    )

    class EchoToolEnv:
        def __init__(self):
            self.env = EchoEnv(base_url=args.env_host)
            self.reward = 0.0

        def reset(self, **kwargs) -> None | str:
            self.reward = 0.0
            return None

        def echo(self, message: str) -> str:
            """
            Echo the message back from the environment.

            Args:
                message: The message to echo

            Returns:
                The echoed message.
            """
            observation = self.env.step(EchoAction(message=message))
            self.reward = observation.observation.reward
            return observation.observation.echoed_message

    # @@@ahoaho XXX
    # trainer = GRPOTrainer(
    #     model=args.model,
    #     train_dataset=dataset,
    #     reward_funcs=reward_func,
    #     args=GRPOConfig(
    #         chat_template_kwargs={"enable_thinking": False},
    #         log_completions=True,
    #         logging_steps=2,
    #         num_completions_to_print=1,
    #     ),
    #     environment_factory=EchoToolEnv,
    # )
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=training_args,
        # args=GRPOConfig(
        #     chat_template_kwargs={"enable_thinking": False},
        #     log_completions=True,
        #     logging_steps=2,
        #     num_completions_to_print=1,
        # ),
        environment_factory=EchoToolEnv,
    )
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
