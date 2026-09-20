from collections import defaultdict
import gymnasium as gym
import numpy as np

# This is a reinforcement learning agent
class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):

        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # track learning progress
        self.training_error = []

    def _state_key(self, obs: np.ndarray) -> tuple[int, ...]:
        bins = (
            np.linspace(-2.4, 2.4, 20),
            np.linspace(-3.0, 3.0, 20),
            np.linspace(-0.2095, 0.2095, 20),
            np.linspace(-3.5, 3.5, 20),
        )

        return tuple(
            int(np.digitize(value, state_bins))
            for value, state_bins in zip(obs, bins)
        )

    def get_action(self, obs: np.ndarray) -> int:
        state = self._state_key(obs)

        # probability epsilon to explore random actions
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return int(np.argmax(self.q_values[state]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)

        # what's the best we could do from the next state?
        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        # what should the q-value be?
        target = reward + self.discount_factor * future_q_value

        # how wrong was our current estimate?
        temporal_difference = target - self.q_values[state][action]

        # update our estimate in the direction of the error
        # learning rate controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        # reduce exploration rate after each episode
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        