import gymnasium as gym
from agent import CartPoleAgent

# training hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# create environment and agent
env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = CartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        

        # choose an action
        action = agent.get_action(obs)

        # take the action and see what happens
        next_obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

print(f"Episode finished! Total reward: {total_reward}")
env.close()