import gym

from baselines import deepq
from baselines import logger
from baselines.common import models


def main():
    env = gym.make("Acrobot-v1")
    logger.configure("logs/Acrobot")
    # Enabling layer_norm here is import for parameter space noise!
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        batch_size=32,
        print_freq=10,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=True
        callback=callback
    )
    print("Saving model to Acrobot_model.pkl")
    act.save("models/Acrobot_model.pkl")


if __name__ == '__main__':
    main()