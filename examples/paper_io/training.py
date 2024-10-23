import os
from time import sleep
import matplotlib.ticker as mticker  # For better tick formatting
import matplotlib.pyplot as plt
import numpy as np
import sys
import pygame  # Import pygame for rendering only if necessary

from Paper_io_develop import PaperIoEnv
# from Paper_io_noSE import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLearningAgent

# Set the flag for rendering the environment
render_game = False  # Set to True if you want to render the game during training
# Create the environment
env = PaperIoEnv(render=render_game)

# Choose the policy
agent = QLearningAgent(env)
policy_name = 'q_learning'

# Training variables
num_episodes = 50000  # You may need more episodes for learning
steps_per_episode = 300  # Adjust as needed
episode_rewards = []  # Store rewards per episode
moving_avg_rewards = []  # Moving average of rewards
steps_per_episode_list = []  # Store steps per episode
epsilon_values = []  # Store epsilon values per episode
episodes = []  # Store episode numbers for plotting 
win_loss_rates = []  # Track win/loss rate (1 for win, 0 for loss)

# Win/loss tracking for two agents
agent_wins = 0
agent_losses = 0

window_size = 50  # Increased window size for smoothing graphs

# Initialize the fixed output format
loading_bar_length = 20  # Length of the loading bar

# Function to find the next available folder index
def get_next_model_index(models_dir, policy_name):
    existing_folders = [d for d in os.listdir(models_dir) if policy_name in d]
    if existing_folders:
        # Extract numeric index from folder names and find the max
        indices = [int(folder.split('_')[-1]) for folder in existing_folders if folder.split('_')[-1].isdigit()]
        if indices:
            return max(indices) + 1
        else:
            return 1  # Default to 1 if no numeric index is found
    else:
        return 1

# Initialize model count based on policy name
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Dynamically determine the next folder name
model_count = get_next_model_index(models_dir, policy_name)
model_folder_name = f"{policy_name}_{model_count}"

# Create directories for the new model
model_folder = os.path.join(models_dir, model_folder_name)
trained_model_folder = os.path.join(model_folder, 'trained_model')
plots_folder = os.path.join(model_folder, 'plots')

# Create directories if they don't exist
os.makedirs(trained_model_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# File to save training details
training_info_file = os.path.join(model_folder, 'training_info.txt')

# Function to save training information
def save_training_info(file_path, num_episodes, steps_per_episode, agent, reward_config):
    with open(file_path, 'w') as f:
        f.write(f"Q-Learning Training Information\n")
        f.write(f"Policy Name: {policy_name}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Steps per Episode: {steps_per_episode}\n")
        f.write(f"Learning Rate: {agent.learning_rate}\n")
        f.write(f"Discount Factor: {agent.discount_factor}\n")
        f.write(f"Initial Epsilon: {1.0}\n")
        f.write(f"Final Epsilon: {agent.epsilon}\n")
        f.write(f"Epsilon Decay Rate: {agent.epsilon_decay}\n")
        f.write(f"Minimum Epsilon: {agent.min_epsilon}\n")
        f.write(f"Total Wins: {agent_wins}\n")
        f.write(f"Total Losses: {agent_losses}\n")
        f.write(f"Final Q-Table Path: {q_table_path}\n")
        # Writing reward information
        f.write("\nReward Information:\n")
        f.write(f"Penalty for Self-Elimination: {env.reward_config['self_elimination_penalty']}\n")
        f.write(f"Reward for Trail Creation (Per 5 Steps): {env.reward_config['trail_reward']}\n")
        f.write(f"Max Trail Reward: {env.reward_config['max_trail_reward']}\n")
        f.write(f"Territory Conversion Multiplier: {env.reward_config['territory_conversion_multiplier']}\n")
        f.write(f"Opponent Elimination Reward: {env.reward_config['opponent_elimination_reward']}\n")
        f.write(f"Opponent Elimination Penalty: {env.reward_config['opponent_elimination_penalty']}\n")

    print(f"Training information saved at {file_path}")

# Print the initial header
print(f"{'Epoch':<6} {'Progress':<23}")
print("=" * 30)

# Train the agent
episode_num = 0

while episode_num < num_episodes:
    obs = env.reset()
    episode_reward = 0
    agent_alive_at_end = False  # Track if agent survives at the end

    # Initialize loading progress
    for step in range(steps_per_episode):
        # Optionally render the game if the flag is True
        if render_game:
            env.render()  # Call the environment's render method

        # Get actions from the agent
        actions = agent.get_actions(obs)

        # Record the current state for each player
        states = []
        for i in range(env.num_players):
            if not env.alive[i]:
                states.append(None)
                continue
            state = agent.get_state(obs, i)
            states.append(state)

        # Take a step in the environment
        next_obs, rewards, done, _ = env.step(actions)
        # sleep(0.2)
        # Update total reward for the episode
        episode_reward += sum(rewards)

        # Record the next state for each player and update Q-values
        for i in range(env.num_players):
            if not env.alive[i]:
                continue
            next_state = agent.get_state(next_obs, i)
            agent.update_q_values(states[i], actions[i], rewards[i], next_state, done, i)

        obs = next_obs

        # Calculate and display loading progress
        if (step + 1) % (steps_per_episode // loading_bar_length) == 0:
            progress_percentage = (step + 1) / steps_per_episode * 100
            loading_bar = "|" + "-" * int((step + 1) / (steps_per_episode / loading_bar_length)) + " " * (loading_bar_length - int((step + 1) / (steps_per_episode / loading_bar_length))) + "|"
            sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} {int(progress_percentage)}%")
            sys.stdout.flush()

        # If the environment signals the end of the episode, break out of the loop
        if done:
            agent_alive_at_end = env.alive[0]  # Assuming agent 0 is the one being trained
            break

    # Finish the loading bar at the end of the episode
    loading_bar = "|" + "-" * loading_bar_length + "|"
    progress_percentage = 100
    sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} {progress_percentage}%\n")
    sys.stdout.flush()

    # Track win/loss
    if agent_alive_at_end:
        agent_wins += 1
        win_loss_rates.append(1)  # Win
    else:
        agent_losses += 1
        win_loss_rates.append(0)  # Loss

    # Store rewards and episode data after finishing the episode
    episode_rewards.append(episode_reward)
    episodes.append(episode_num)
    steps_per_episode_list.append(step + 1)  # Track the number of steps taken

    # Track epsilon value
    epsilon_values.append(agent.epsilon)

    # Calculate moving average reward
    if len(episode_rewards) >= window_size:
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
    else:
        moving_avg_rewards.append(np.mean(episode_rewards))

    # Decay epsilon after each episode
    agent.decay_epsilon()

    episode_num += 1

# Plotting the training progress (Episode Rewards)
plt.figure(figsize=(10, 5))
plt.plot(episodes, episode_rewards, label='Episode Reward', linewidth=0.75)
plt.plot(episodes, moving_avg_rewards, label=f'Moving Average Reward (window={window_size})', color='orange', linewidth=2)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress: Reward over Episodes')
plt.legend()
plt.grid(True)

# Save the plot to a file in the plots folder
plot_path = os.path.join(plots_folder, 'training_progress.png')
plt.savefig(plot_path)
print(f"Training progress graph saved at {plot_path}")

# Plotting Steps Per Episode
plt.figure(figsize=(10, 5))
plt.scatter(episodes, steps_per_episode_list, label='Steps Per Episode', color='green', s=3)  # s=1 for small dots
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.title('Steps Taken Per Episode')
plt.legend()
plt.grid(True)

# Save the plot to a file in the plots folder
plot_path_steps = os.path.join(plots_folder, 'steps_per_episode.png')
plt.savefig(plot_path_steps)
print(f"Steps per episode graph saved at {plot_path_steps}")

# Plotting Epsilon Decay
plt.figure(figsize=(10, 5))
plt.plot(episodes, epsilon_values, label='Epsilon Decay', color='purple', linewidth=0.75)
plt.xlabel('Episodes')
plt.ylabel('Epsilon Value')
plt.title('Epsilon Decay Over Time')
plt.legend()
plt.grid(True)

# Save the plot to a file in the plots folder
plot_path_epsilon = os.path.join(plots_folder, 'epsilon_decay.png')
plt.savefig(plot_path_epsilon)
print(f"Epsilon decay graph saved at {plot_path_epsilon}")

# Plotting the TD Error over Time
plt.figure(figsize=(10, 5))
plt.plot(range(len(agent.td_errors)), agent.td_errors, label='TD Error', linewidth=0.75)
plt.xlabel('Steps')
plt.ylabel('TD Error')
plt.title('TD Error over Training')
plt.legend()
plt.grid(True)

# Save the TD error plot to a file in the plots folder
plot_path_td_error = os.path.join(plots_folder, 'td_error.png')
plt.savefig(plot_path_td_error)
print(f"TD error graph saved at {plot_path_td_error}")

# Plotting Win/Loss Rate
plt.figure(figsize=(10, 5))
plt.plot(episodes, win_loss_rates, label='Win/Loss Rate', color='blue', linewidth=0.75)
plt.xlabel('Episodes')
plt.ylabel('Win/Loss (1=Win, 0=Loss)')
plt.title('Win/Loss Rate Over Episodes')
plt.legend()
plt.grid(True)

# Save the win/loss plot to a file in the plots folder
plot_path_win_loss = os.path.join(plots_folder, 'win_loss_rate.png')
plt.savefig(plot_path_win_loss)
print(f"Win/Loss rate graph saved at {plot_path_win_loss}")

# Save the Q-table after training
q_table_path = os.path.join(trained_model_folder, 'q_table.pkl')
agent.save_q_table(q_table_path)
print(f"Q-table saved at {q_table_path}")

# Save the training information at the end of training
save_training_info(training_info_file, num_episodes, steps_per_episode, agent, env.reward_config)

# Show the plots
plt.show()

# Cleanup pygame if rendering was enabled
if render_game:
    pygame.quit()  # Ensure pygame quits to close the window properly

print("Training was Completed.")
