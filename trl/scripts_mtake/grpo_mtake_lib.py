from trl.rewards import accuracy_reward


def my_accuracy_reward(**kwargs):
    return accuracy_reward(**kwargs)
