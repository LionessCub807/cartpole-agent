import torch
import gymnasium
from dqn import DQN

class Agent:
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions)

        obs, _ = env.reset()
        while True:
            action = env.action_space.sample()

            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                break

        env.close()