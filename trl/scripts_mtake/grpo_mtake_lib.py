# from collections.abc import Callable
from trl.rewards import accuracy_reward


# def my_accuracy_reward(
#     completions: list[list[dict[str, str]]],
#     solution: list[str],
#     log_extra: Callable[[str, list], None] | None = None,
#     **kwargs,
# ) -> list[float | None]:
#     return accuracy_reward(
#         completions=completions,
#         solution=solution,
#         log_extra=log_extra,
#         **kwargs,
#     )

def my_accuracy_reward(**kwargs):
    return accuracy_reward(**kwargs)
