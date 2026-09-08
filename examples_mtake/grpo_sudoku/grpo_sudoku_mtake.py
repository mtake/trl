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
#     "trl[vllm,peft]",
#     "trackio",
#     "kernels",
#     "openenv-textarena @ git+https://huggingface.co/spaces/openenv/sudoku",
# ]
# ///

"""
GRPO training for Sudoku with TextArena environment.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/sudoku
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv/envs/textarena_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/grpo_sudoku/grpo_sudoku.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/grpo_sudoku/grpo_sudoku.py --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local + Colocated vLLM (1 GPU required)

# Start the environment only if using --env-mode docker-local
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-sudoku:latest
```

```sh
python examples/grpo_sudoku/grpo_sudoku.py --env-mode docker-local --vllm-mode colocate
```

# Full example with all flags:
```sh
python examples/grpo_sudoku/grpo_sudoku.py \
    --vllm-mode colocate \
    --env-mode space \
    --env-host https://openenv-sudoku.hf.space \
    --num-generations 8 \
    --per-device-batch-size 1 \
    --max-turns 100 \
    --gradient-accumulation-steps 8 \
    --difficulty easy \
    --dataset-size 100
```
"""

from __future__ import annotations

# ruff: noqa: T201
# @@@ahoaho XXX
# import argparse
# @@@ahoaho XXX
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
# @@@ahoaho XXX
from typing import Literal

# @@@ahoaho XXX
from dataclasses import dataclass, field
from datasets import Dataset
from openenv.core.containers.runtime import LocalDockerProvider

# @@@ahoaho XXX
# from trl import GRPOConfig, GRPOTrainer, RichProgressCallback
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, RichProgressCallback


# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from textarena_env import TextArenaAction, TextArenaEnv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


# @@@ahoaho XXX
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="GRPO training for Sudoku")

#     # Model
#     parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")  # @@@ahoaho XXX model_args.model_name_or_path TODO

#     # @@@ahoaho XXX sudoku specific
#     # Environment
#     parser.add_argument("--env-host", type=str, default="https://openenv-sudoku.hf.space")
#     parser.add_argument("--env-port", type=int, default=8001)
#     parser.add_argument("--env-mode", choices=["docker-local", "docker-image", "space"], default="space")
#     parser.add_argument("--env-image", type=str, default="textarena-env:latest")

#     # @@@ahoaho XXX sudoku specific
#     # Prompts
#     parser.add_argument("--system-prompt-path", default="sudoku_prompt.txt")
#     parser.add_argument("--dataset-prompt", default="Play Sudoku like an expert.")
#     parser.add_argument("--dataset-size", type=int, default=1000)

#     # @@@ahoaho XXX sudoku specific
#     # Game settings
#     parser.add_argument("--max-turns", type=int, default=100)
#     parser.add_argument(
#         "--difficulty",
#         type=str,
#         choices=["easy", "medium", "hard"],
#         default="easy",
#         help="Training difficulty: easy=guaranteed+options, medium=only options, hard=no hints",
#     )
#     parser.add_argument(
#         "--api-delay", type=float, default=0.0, help="Delay in seconds between API calls to avoid rate limiting"
#     )

#     # Sampling
#     parser.add_argument("--temperature", type=float, default=0.8)  # @@@ahoaho XXX training_args.temperature
#     parser.add_argument("--top-k", type=int, default=10)  # @@@ahoaho XXX training_args.top_k
#     parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter")  # @@@ahoaho XXX training_args.top_p

#     # Training
#     parser.add_argument("--learning-rate", type=float, default=5e-6)  # @@@ahoaho XXX training_args.learning_rate
#     parser.add_argument("--weight-decay", type=float, default=0.0)  # @@@ahoaho XXX training_args.weight_decay
#     parser.add_argument("--gradient-accumulation-steps", type=int, default=64)  # @@@ahoaho XXX training_args.gradient_accumulation_steps
#     parser.add_argument("--warmup-steps", type=int, default=20)  # @@@ahoaho XXX training_args.warmup_steps
#     parser.add_argument("--per-device-batch-size", type=int, default=1)  # @@@ahoaho XXX training_args.per_device_train_batch_size TODO
#     parser.add_argument("--num-generations", type=int, default=8)  # @@@ahoaho XXX training_args.num_generations
#     parser.add_argument("--num-epochs", type=int, default=1)  # @@@ahoaho XXX training_args.num_train_epochs TODO
#     parser.add_argument("--max-completion-length", type=int, default=16384)  # @@@ahoaho XXX training_args.max_completion_length
#     parser.add_argument("--resume-from-checkpoint", type=bool, default=False)  # @@@ahoaho XXX training_args.resume_from_checkpoint

#     # Checkpoints
#     parser.add_argument("--save-interval", type=int, default=10)  # @@@ahoaho XXX training_args.save_steps TODO
#     parser.add_argument("--save-total-limit", type=int, default=None)  # @@@ahoaho XXX training_args.save_total_limit
#     parser.add_argument("--output-dir", default=None)  # @@@ahoaho XXX training_args.output_dir

#     # Logging
#     parser.add_argument("--run-name", default=None)  # @@@ahoaho XXX training_args.run_name
#     parser.add_argument("--project", default=None)  # @@@ahoaho XXX training_args.project
#     parser.add_argument("--trackio-space-id", default="Sudoku-GRPO")  # @@@ahoaho XXX training_args.trackio_space_id
#     parser.add_argument("--logging-steps", type=int, default=1)  # @@@ahoaho XXX training_args.logging_steps
#     parser.add_argument(
#         "--gradient-checkpointing",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Enable gradient checkpointing to save memory",
#     )  # @@@ahoaho XXX training_args.gradient_checkpointing

#     # LoRA / PEFT
#     parser.add_argument(
#         "--use-lora", action="store_true", default=False, help="Use LoRA for memory-efficient training"
#     )  # @@@ XXX model_args.use_peft TODO
#     parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")  # @@@ XXX model_args.lora_r
#     parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")  # @@@ XXX model_args.lora_alpha

#     # vLLM
#     # @@@ahoaho XXX TODO training_args.use_vllm = True
#     parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")  # @@@ahoaho XXX training_args.vllm_mode
#     parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000")  # @@@ahoaho XXX training_args.vllm_server_base_url TODO
#     parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.15)  # @@@ahoaho XXX training_args.vllm_gpu_memory_utilization

#     return parser.parse_args()


# @@@ahoaho XXX
@dataclass
class GRPOSudokuScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        env_host (`str`, *optional*):
            URL for the Sudoku environment HF Space. Default is "https://openenv-sudoku.hf.space".
        env_port (`int`, *optional*):
            Port for the Sudoku environment. Default is 8001.
        env_mode (`str`, *optional*):
            Mode for the Sudoku environment. Default is "space".
        env_image (`str`, *optional*):
            Image for the Sudoku environment. Default is "textarena-env:latest".
        system_prompt_path (`str`, *optional*):
            Path to the system prompt file. Default is "sudoku_prompt.txt".
        dataset_prompt (`str`, *optional*):
            User prompt. Default is "Play Sudoku like an expert.".
        dataset_size (`int`, *optional*):
            Dataset size. Default is 1000.
        max_turns (`int`, *optional*):
            Max turns to play game. Default is 100.
        difficulty (`str`, *optional*):
            Training difficulty: easy=guaranteed+options, medium=only options, hard=no hints. Default is "easy".
        api_delay (`float`, *optional*):
            Delay in seconds between API calls to avoid rate limiting. Default is 0.0.
        # @@@ahoaho XXX
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

    # Environment
    env_host: str | None = field(
        default="https://openenv-sudoku.hf.space",
        metadata={
            "help": "URL for the Sudoku environment HF Space."
        },
    )
    env_port: int | None = field(
        default=8001,
        metadata={
            "help": "Port for the Sudoku environment."
        },
    )
    env_mode: Literal["docker-local", "docker-image", "space"] | None = field(
        default="space",
        metadata={
            "help": "Mode for the Sudoku environment."
        },
    )
    env_image: str | None = field(
        default="textarena-env:latest",
        metadata={
            "help": "Image for the Sudoku environment."
        },
    )
    # Prompts
    system_prompt_path: str | None = field(
        default="sudoku_prompt.txt",
        metadata={
            "help": "Path to the system prompt file."
        },
    )
    dataset_prompt: str | None = field(
        default="Play Sudoku like an expert.",
        metadata={
            "help": "User prompt."
        },
    )
    dataset_size: int | None = field(
        default=1000,
        metadata={
            "help": "Dataset size."
        },
    )
    # Game settings
    max_turns: int | None = field(
        default=100,
        metadata={
            "help": "Max turns to play game."
        },
    )
    difficulty: Literal["easy", "medium", "hard"] | None = field(
        default="easy",
        metadata={
            "help": "Training difficulty: easy=guaranteed+options, medium=only options, hard=no hints."
        },
    )
    api_delay: float | None = field(
        default=0.0,
        metadata={
            "help": "Delay in seconds between API calls to avoid rate limiting."
        },
    )
    # @@@ahoaho XXX
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



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def resolve_system_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        prompt_path = Path(__file__).parent / path
    return prompt_path.read_text()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def is_valid_board_state(board_str: str) -> bool:
    """Check if the string contains an actual Sudoku board."""
    return "R1" in board_str and "R9" in board_str and "|" in board_str


def parse_board(board_str: str) -> list[list[int]]:
    """Parse board string into 9x9 grid (0 = empty)."""
    grid = [[0] * 9 for _ in range(9)]
    if not is_valid_board_state(board_str):
        return grid

    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1]) - 1  # 0-indexed
            cell_part = line_stripped[2:]
            col = 0
            for char in cell_part:
                if char == ".":
                    grid[row][col] = 0
                    col += 1
                elif char.isdigit():
                    grid[row][col] = int(char)
                    col += 1
    return grid


def count_filled_cells(board_str: str) -> int:
    """Count the number of filled cells in the board."""
    if not is_valid_board_state(board_str):
        return 0
    grid = parse_board(board_str)
    return sum(1 for row in grid for cell in row if cell != 0)


def get_valid_numbers(grid: list[list[int]], row: int, col: int) -> set[int]:
    """Get valid numbers for a cell based on Sudoku rules."""
    if grid[row][col] != 0:
        return set()

    used = set()

    # Check row
    for c in range(9):
        if grid[row][c] != 0:
            used.add(grid[row][c])

    # Check column
    for r in range(9):
        if grid[r][col] != 0:
            used.add(grid[r][col])

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] != 0:
                used.add(grid[r][c])

    return set(range(1, 10)) - used


def extract_empty_cells_with_candidates(
    board_str: str, sort_by_difficulty: bool = True
) -> list[tuple[int, int, set[int]]]:
    """Extract empty cells with their valid candidate numbers.

    Args:
        sort_by_difficulty: If True, sort by number of candidates (easiest first).
                           If False, keep natural order (top-left to bottom-right).
    """
    grid = parse_board(board_str)
    cells_with_candidates = []

    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = get_valid_numbers(grid, row, col)
                cells_with_candidates.append((row + 1, col + 1, candidates))  # 1-indexed

    if sort_by_difficulty:
        # Sort by number of candidates (easiest first = naked singles)
        cells_with_candidates.sort(key=lambda x: len(x[2]))

    return cells_with_candidates


def extract_empty_cells(board_str: str) -> list[tuple[int, int]]:
    """Extract list of empty cells (row, col) from board string."""
    empty_cells = []
    if not is_valid_board_state(board_str):
        return empty_cells

    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1])
            cell_part = line_stripped[2:]
            col = 0
            for char in cell_part:
                if char == ".":
                    col += 1
                    empty_cells.append((row, col))
                elif char.isdigit():
                    col += 1
    return empty_cells


def extract_board_only(text: str) -> str:
    """Extract just the Sudoku grid from a message."""
    if not text:
        return ""

    lines = text.split("\n")
    board_lines = []
    in_board = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("C1") or (
            stripped and stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit()
        ):
            in_board = True
        if in_board and (stripped.startswith("-") or stripped.startswith("R") or stripped.startswith("C1")):
            board_lines.append(line)
        elif (
            in_board
            and stripped
            and not stripped.startswith("-")
            and not (stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit())
        ):
            break

    return "\n".join(board_lines) if board_lines else ""


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def reward_empty_cell(environments, **kwargs) -> list[float]:
    """Reward for targeting empty cells (learn to pick valid positions first)."""
    return [env.empty_cell_reward for env in environments]


def reward_valid_moves(environments, **kwargs) -> list[float]:
    """Reward for making valid moves."""
    return [env.valid_move_reward for env in environments]


def reward_correct(environments, **kwargs) -> list[float]:
    """Reward for solving the puzzle."""
    return [env.correct_reward for env in environments]


def reward_repetition(environments, **kwargs) -> list[float]:
    """Penalty for repeating moves."""
    return [env.repetition_reward for env in environments]


def reward_progress(environments, **kwargs) -> list[float]:
    """Reward for filling more cells in the board."""
    return [env.progress_reward for env in environments]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # @@@ahoaho XXX
    # args = parse_args()
    parser = TrlParser((GRPOSudokuScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    training_args.chat_template_kwargs = {"enable_thinking": False}

    # @@@ahoaho XXX
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"XXX script_args: {script_args} XXX")
        print(f"XXX training_args: {training_args} XXX")
        print(f"XXX model_args: {model_args} XXX")

    # Setup environment — all modes resolve to env_url
    # @@@ahoaho XXX
    # if args.env_mode == "docker-local":
    #     env_url = f"http://{args.env_host}:{args.env_port}"
    # elif args.env_mode == "docker-image":
    #     provider = LocalDockerProvider()
    #     env_url = provider.start_container(args.env_image)
    #     provider.wait_for_ready(env_url)
    # elif args.env_mode == "space":
    #     env_url = args.env_host
    # else:
    #     raise ValueError(f"Unknown environment mode: {args.env_mode}")

    # print(f"Environment: {args.env_mode} ({env_url})")
    if script_args.env_mode == "docker-local":
        env_url = f"http://{script_args.env_host}:{script_args.env_port}"
    elif script_args.env_mode == "docker-image":
        provider = LocalDockerProvider()
        env_url = provider.start_container(script_args.env_image)
        provider.wait_for_ready(env_url)
    elif script_args.env_mode == "space":
        env_url = script_args.env_host
    else:
        raise ValueError(f"Unknown environment mode: {script_args.env_mode}")

    print(f"Environment: {script_args.env_mode} ({env_url})")

    # @@@ahoaho XXX
    # system_prompt = resolve_system_prompt(args.system_prompt_path)
    # dataset = Dataset.from_dict(
    #     {
    #         "prompt": [
    #             [
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": args.dataset_prompt},
    #             ]
    #         ]
    #         * args.dataset_size
    #     }
    # )
    system_prompt = resolve_system_prompt(script_args.system_prompt_path)
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": script_args.dataset_prompt},
                ]
            ]
            * script_args.dataset_size
        }
    )

    # Capture args for use in the environment class closure
    # @@@ahoaho XXX
    # difficulty = args.difficulty
    # max_turns = args.max_turns
    # api_delay = args.api_delay
    difficulty = script_args.difficulty
    max_turns = script_args.max_turns
    api_delay = script_args.api_delay

    class SudokuEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url)
            # self.client = TextArenaEnv(base_url=env_url).sync()

            # @@@ahoaho XXX `.sync()` will make self.client.reset() return StepResult rather than coroutine. This is probably a right fix, but it triggers an another error.

            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/examples_mtake/grpo_sudoku/grpo_sudoku_mtake.py", line 627, in reset
            # [rank0]:     result = self.client.reset()
            # [rank0]:              ^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/openenv/core/sync_client.py", line 178, in reset
            # [rank0]:     return self._run(self._async.reset(**kwargs))
            # [rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/openenv/core/sync_client.py", line 132, in _run
            # [rank0]:     return future.result()
            # [rank0]:            ^^^^^^^^^^^^^^^
            # [rank0]:   File "/u/mtake/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 456, in result
            # [rank0]:     return self.__get_result()
            # [rank0]:            ^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/u/mtake/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
            # [rank0]:     raise self._exception
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/openenv/core/env_client.py", line 389, in reset
            # [rank0]:     response = await self._send_and_receive(message)
            # [rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/openenv/core/env_client.py", line 226, in _send_and_receive
            # [rank0]:     await self._send(message)
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/openenv/core/env_client.py", line 216, in _send
            # [rank0]:     await self._ws.send(json.dumps(message))
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/websockets/asyncio/connection.py", line 473, in send
            # [rank0]:     async with self.send_context():
            # [rank0]:                ^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/u/mtake/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
            # [rank0]:     return await anext(self.gen)
            # [rank0]:            ^^^^^^^^^^^^^^^^^^^^^
            # [rank0]:   File "/proj/dmfexp/granite_ja/mtake/w/trl-command/trl/.venv/lib/python3.12/site-packages/websockets/asyncio/connection.py", line 960, in send_context
            # [rank0]:     raise self.protocol.close_exc from original_exc
            # [rank0]: websockets.exceptions.ConnectionClosedOK: received 1000 (OK); then sent 1000 (OK)

            self._difficulty = difficulty
            self._max_turns = max_turns
            self._api_delay = api_delay
            self._reset_state()

        def _reset_state(self):
            self._move_counts: defaultdict[str, int] = defaultdict(int)
            self._successful_moves: list[str] = []
            self._failed_moves: list[str] = []
            self._valid_move_scores: list[float] = []
            self._empty_cell_scores: list[float] = []
            self._correct_scores: list[float] = []
            self._repetition_scores: list[float] = []
            self._last_board_state = ""
            self._initial_filled = 0
            self._max_filled = 0
            self._turn = 0
            self._done = False

        def reset(self, **kwargs) -> str:
            self._reset_state()
            result = self.client.reset()
            time.sleep(self._api_delay)
            observation = result.observation
            self._done = result.done

            # Store full message content for diffing (messages are cumulative)
            self._last_full_content = observation.messages[0].content if observation.messages else ""

            if is_valid_board_state(self._last_full_content):
                self._last_board_state = self._last_full_content
                self._initial_filled = count_filled_cells(self._last_board_state)
            self._max_filled = self._initial_filled

            board = extract_board_only(self._last_board_state) if self._last_board_state else "No board available."
            hints = self._format_hints()
            return f"Step 0. Progress: 0 cells filled.\n\nBoard:\n{board}{hints}"

        def place(self, row: int, col: int, number: int) -> str:
            """Place a number on the Sudoku board.

            Args:
                row: Row number (1-9).
                col: Column number (1-9).
                number: Number to place (1-9).

            Returns:
                The result of the move and updated board state.
            """
            if self._done:
                raise ValueError("Game is over. No more moves allowed.")

            self._turn += 1
            move = f"[{row} {col} {number}]"

            # Step environment
            result = self.client.step(TextArenaAction(message=move))
            time.sleep(self._api_delay)
            observation = result.observation
            correct_score = float(result.reward or 0.0)
            self._done = result.done

            # Only check the NEW content for feedback (messages are cumulative)
            full_content = observation.messages[0].content if observation.messages else ""
            new_content = full_content[len(self._last_full_content) :]
            self._last_full_content = full_content

            new_content_lower = new_content.lower()
            env_says_invalid = any(
                kw in new_content_lower for kw in ["invalid", "error", "cannot", "already", "violation", "lost"]
            )
            got_warning = "please resubmit" in new_content_lower or "avoid penalties" in new_content_lower

            # Also verify against our own board state: placing on a non-empty cell is always invalid
            if self._last_board_state:
                empty_cells = extract_empty_cells(self._last_board_state)
                targets_empty = (row, col) in empty_cells
            else:
                empty_cells = []
                targets_empty = True  # Can't verify, assume valid

            is_valid = not env_says_invalid and targets_empty

            # Empty cell score: did the model target an empty cell?
            empty_cell_score = 1.0 if targets_empty else -1.0

            # Repetition tracking
            is_new_move = self._move_counts[move] == 0
            repetition_count = self._move_counts[move]
            self._move_counts[move] += 1
            repetition_score = -min(2 ** (repetition_count - 1), 10.0) if repetition_count > 0 else 0.0

            # Valid move score
            if is_valid and is_new_move:
                valid_move_score = 1.0
                self._successful_moves.append(move)
            elif got_warning:
                valid_move_score = -0.5
                self._failed_moves.append(move)
            else:
                valid_move_score = 0.0

            # Update board state from new content
            if is_valid and is_valid_board_state(new_content):
                self._last_board_state = new_content
                current_filled = count_filled_cells(self._last_board_state)
                if current_filled > self._max_filled:
                    self._max_filled = current_filled

            self._valid_move_scores.append(valid_move_score)
            self._empty_cell_scores.append(empty_cell_score)
            self._correct_scores.append(correct_score)
            self._repetition_scores.append(repetition_score)

            # Enforce max turns
            if self._turn >= self._max_turns:
                self._done = True

            # Build response
            board = extract_board_only(self._last_board_state) if self._last_board_state else "No board available."
            status = "valid" if is_valid else "invalid"
            cells_filled = len(self._successful_moves)
            progress = f"Step {self._turn}. Progress: {cells_filled} cells filled."
            hints = self._format_hints()

            if self._done:
                return f"Move {move}: {status}. Game over.\n{progress}\n\nFinal board:\n{board}"
            return f"Move {move}: {status}\n{progress}\n\nBoard:\n{board}{hints}"

        def _format_hints(self) -> str:
            parts = []

            # Already tried moves (avoid repetitions)
            all_tried = self._successful_moves + self._failed_moves
            if all_tried:
                parts.append(f"\nMOVES ALREADY TRIED (do not repeat): {', '.join(all_tried)}")

            if not self._last_board_state:
                return "\n".join(parts)

            if self._difficulty == "easy":
                cells = extract_empty_cells_with_candidates(self._last_board_state, sort_by_difficulty=True)
                if cells:
                    guaranteed = []
                    other = []
                    for r, c, candidates in cells[:10]:
                        if len(candidates) == 1:
                            guaranteed.append(f"[{r} {c} {list(candidates)[0]}]")
                        elif len(candidates) <= 3:
                            nums = ",".join(str(n) for n in sorted(candidates))
                            other.append(f"({r},{c})->{nums}")
                    if guaranteed:
                        parts.append(f"\nGUARANTEED MOVES: {', '.join(guaranteed[:5])}")
                    if other:
                        parts.append(f"Other options: {' | '.join(other[:5])}")

            elif self._difficulty == "medium":
                cells = extract_empty_cells_with_candidates(self._last_board_state, sort_by_difficulty=False)
                if cells:
                    cell_hints = []
                    for r, c, candidates in cells[:10]:
                        nums = ",".join(str(n) for n in sorted(candidates))
                        cell_hints.append(f"({r},{c})->{nums}")
                    parts.append(f"\nEmpty cells: {' | '.join(cell_hints)}")

            return "\n".join(parts)

        # Reward properties — properties are not detected by inspect.ismethod,
        # so they won't be exposed as tools.

        @property
        def correct_reward(self) -> float:
            return self._correct_scores[-1] if self._correct_scores else 0.0

        @property
        def valid_move_reward(self) -> float:
            return sum(self._valid_move_scores) / len(self._valid_move_scores) if self._valid_move_scores else 0.0

        @property
        def empty_cell_reward(self) -> float:
            return sum(self._empty_cell_scores) / len(self._empty_cell_scores) if self._empty_cell_scores else 0.0

        @property
        def repetition_reward(self) -> float:
            return sum(self._repetition_scores) / len(self._repetition_scores) if self._repetition_scores else 0.0

        @property
        def progress_reward(self) -> float:
            remaining = 81 - self._initial_filled
            if remaining > 0:
                return (self._max_filled - self._initial_filled) / remaining
            return 1.0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # @@@ahoaho XXX
    # output_dir = Path(args.output_dir or f"outputs/sudoku-grpo-{sanitize_name(args.model_id)}-{timestamp}")
    # output_dir = Path(training_args.output_dir or f"outputs/sudoku-grpo-{sanitize_name(model_args.model_name_or_path)}-{timestamp}")

    # @@@ahoaho XXX
    # grpo_config = GRPOConfig(
    #     use_vllm=True,
    #     vllm_mode=args.vllm_mode,
    #     vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
    #     vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization or 0.2,
    #     output_dir=str(output_dir),
    #     num_train_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     per_device_train_batch_size=args.per_device_batch_size,
    #     warmup_steps=args.warmup_steps,
    #     num_generations=args.num_generations,
    #     max_completion_length=args.max_completion_length,
    #     logging_steps=args.logging_steps,
    #     save_strategy="steps",
    #     save_steps=args.save_interval,
    #     save_total_limit=args.save_total_limit,
    #     temperature=args.temperature,
    #     top_k=args.top_k,
    #     top_p=args.top_p,
    #     report_to="trackio",
    #     log_completions=True,
    #     num_completions_to_print=1,
    #     chat_template_kwargs={"enable_thinking": False},
    # )

    # @@@ahoaho XXX
    # grpo_config.run_name = args.run_name or f"run-{timestamp}"
    # grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"
    # grpo_config.trackio_space_id = args.trackio_space_id
    # grpo_config.gradient_checkpointing = args.gradient_checkpointing

    peft_config = None
    # @@@ahoaho XXX
    # if args.use_lora:
    #     from peft import LoraConfig

    #     peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, task_type="CAUSAL_LM")
    if model_args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(r=model_args.lora_r, lora_alpha=model_args.lora_alpha, task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        # @@@ahoaho XXX
        # model=args.model_id,
        model=model_args.model_name_or_path,
        reward_funcs=[
            reward_empty_cell,  # Learn to pick empty cells
            reward_valid_moves,  # Learn valid numbers
            reward_repetition,  # Penalize repeating moves
            reward_progress,  # Reward filling more cells
            reward_correct,  # Solve the puzzle
        ],
        peft_config=peft_config,
        train_dataset=dataset,
        # @@@ahoaho XXX
        # args=grpo_config,
        args=training_args,
        environment_factory=SudokuEnv,
        callbacks=[RichProgressCallback()],
    )

    # @@@ahoaho XXX
    # print(f"Starting GRPO training: {args.num_generations} generations, {args.max_turns} max turns")
    print(f"Starting GRPO training: {training_args.num_generations} generations, {script_args.max_turns} max turns")
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
