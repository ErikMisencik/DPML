import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import time  
from examples.paper_io.utils.agent_colors import assign_agent_colors
from examples.paper_io.utils.plots import (
    plot_average_self_eliminations, plot_cumulative_rewards, plot_cumulative_self_eliminations, plot_epsilon_decay, plot_td_error,
    plot_training_progress, plot_agent_wins, plot_agent_eliminations, plot_average_eliminations, plot_territory_gained
)
import pygame  # Import pygame for rendering only if necessary

from Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent


# Choose algorithm and initialize agents
num_agents = 2 
render_game = False  # Set to True if you want to render the game during training 
load_existing_model = False  # Set to True to load an existing model
steps_per_episode = 350

window_size = 50               # For smoothing graphs
loading_bar_length = 20;       # Length of the loading bar

# Training variables
discount_factor = 0.99  # Typically remains the same for both training and retraining

# Set parameters based on whether we are training from scratch or retraining
if load_existing_model:
    # Parameters for retraining (fine-tuning)
    num_episodes = 7500           # Fewer episodes for retraining
    epsilon = 0.35                  # Lower initial exploration rate for retraining
    learning_rate = 0.003          # Smaller learning rate for fine-tuning
    epsilon_reset = False          
    epsilon_reset_value = 0.15      # Mid-range value for epsilon reset
    epsilon_reset_interval = 3000  # More frequent exploration resets
    epsilon_decay = 0.9994          # Keep similar decay rate
    min_epsilon = 0.1              # Minimum exploration rate remains the same

else:
    # Parameters for initial training
    num_episodes = 20000           # Full training length
    epsilon = 1.0                  # High exploration at start
    learning_rate = 0.002          # Standard learning rate for initial training
    epsilon_reset = True          # No epsilon reset for initial training
    epsilon_reset_value = 0.50     # Not used if epsilon_reset is False
    epsilon_reset_interval = 5000  # Not used if epsilon_reset is False
    epsilon_decay = 0.9995         # Standard decay rate
    min_epsilon = 0.05              # Minimum exploration rate


env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=num_agents)

agents = [QLAgent(env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon) for _ in range(num_agents)]

agent_type = agents[0].__class__.__name__  # Assumes all agents are of the same type

policy_name = (
    f"{'PreTrained' if load_existing_model else 'New'}_"
    f"{'M' if num_agents > 1 else 'S'}_"
    f"{num_agents}_" if num_agents > 1 else ""
) + f"{agent_type}"

# Function to find the next available folder index
def get_next_model_index(models_dir, policy_name):
    existing_folders = [d for d in os.listdir(models_dir) if policy_name in d]
    if existing_folders:
        indices = [int(folder.split('_')[-1]) for folder in existing_folders if folder.split('_')[-1].isdigit()]
        return max(indices) + 1 if indices else 1
    else:
        return 1

# Initialize model count based on policy name
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Determine the next folder name
model_count = get_next_model_index(models_dir, policy_name)
model_folder_name = f"{policy_name}_{model_count}"

# Create directories for the new model
model_folder = os.path.join(models_dir, model_folder_name)
trained_model_folder = os.path.join(model_folder, 'trained_model')
plots_folder = os.path.join(model_folder, 'plots')
os.makedirs(trained_model_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# File to save training details
training_info_file = os.path.join(model_folder, 'training_info.txt')
q_table_file = os.path.join('models', 'PreTrained_S_QLAgent_4', 'trained_model', 'q_table_ag_0_end.pkl')

# Assign random colors to agents
color_info = assign_agent_colors(env.num_players)
agent_colors = [info[0] for info in color_info]  # RGB values for rendering
agent_color_names = [info[1] for info in color_info]  # Color names for logging

# Initialize agents and load model if needed
agents = []
for i in range(num_agents):
    agent = QLAgent(env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon)
    if load_existing_model and os.path.exists(q_table_file):
        agent.load(q_table_file)
    agents.append(agent)
policy_name = 'q_learning'

episode_rewards = []
moving_avg_rewards = []
steps_per_episode_list = []
epsilon_values = []
episodes = []
eliminations_per_episode = []
self_eliminations_per_episode = []
  
# Initialize cumulative counts
agent_wins = [0 for _ in range(env.num_players)]
agent_eliminations = [0 for _ in range(env.num_players)]
agent_self_eliminations = [0 for _ in range(env.num_players)]
cumulative_rewards_per_agent = [[] for _ in range(env.num_players)]
territory_per_agent = [[] for _ in range(env.num_players)]  # Track territory per agent

# Function to save training information
def save_training_info(file_path, num_episodes, steps_per_episode, agent, reward_config):
    with open(file_path, 'w') as f:
        # General Training Info
        f.write("=== Q-Learning Training Information ===\n")
        f.write(f"Policy Name         : {policy_name}\n")
        f.write(f"Number of Episodes  : {num_episodes}\n")
        f.write(f"Max Steps per Ep.   : {env.max_steps}\n")
        f.write(f"Learning Rate       : {agent.learning_rate}\n")
        f.write(f"Discount Factor     : {agent.discount_factor}\n")
        f.write(f"Initial Epsilon     : {epsilon}\n")
        f.write(f"Final Epsilon       : {agent.epsilon}\n")
        f.write(f"Epsilon Decay Rate  : {agent.epsilon_decay}\n")
        f.write(f"Minimum Epsilon     : {agent.min_epsilon}\n")
        f.write(f"Epsilon Reset Every : {epsilon_reset_interval} episodes\n")
        f.write(f"Epsilon Reset Value : {epsilon_reset_value}\n")
        f.write("\n")

        # Agent Statistics
        f.write("=== Agent Statistics ===\n")
        for idx in range(env.num_players):
            avg_cumulative_reward = (np.mean(cumulative_rewards_per_agent[idx])
                                     if cumulative_rewards_per_agent[idx] else 0)
            f.write(f"Agent {idx}:\n")
            f.write(f"  - Total Wins           : {agent_wins[idx]}\n")
            f.write(f"  - Total Eliminations   : {agent_eliminations[idx]}\n")
            f.write(f"  - Total Self-Eliminations : {agent_self_eliminations[idx]}\n")
            f.write(f"  - Avg Cumulative Reward : {avg_cumulative_reward:.2f}\n\n")

        # Additional Information
        f.write(f"Agent Colors (Names): {', '.join(agent_color_names)}\n")
        if load_existing_model: # Add the loaded model path
            f.write(f"Loaded Q-Table Path : {q_table_file}\n")
        f.write(f"Final Q-Table Path  : {q_table_path}\n")
        f.write("\n")

        # Reward Configuration
        f.write("=== Reward Information ===\n")
        for key, value in env.reward_config.items():
            reward_name = key.replace('_', ' ').capitalize()
            f.write(f"{reward_name:20}: {value}\n")
    
    print(f"Training information saved at {file_path}")

# Start the timer for the entire training process
training_start_time = time.time()

# Print the initial header
print(f"{'Epoch':<6} {'Progress':<23}")
print("=" * 30)

# Train the agent
episode_num = 0

while episode_num < num_episodes:
    obs = env.reset()
    episode_reward = 0
    done = False  # Initialize the done flag for each episode
    step = 0  # Initialize step counter

    while not done:
        if render_game:
            env.render(agent_colors)

        # Get actions for each agent
        actions = [agent.get_action(obs, i) for i, agent in enumerate(agents)]
        states = [agent.get_state(obs, i) for i, agent in enumerate(agents)]

        # Step in the environment with the selected actions
        next_obs, rewards, done, info = env.step(actions)
        episode_reward += sum(rewards)

        # Update each agent with its respective state, action, reward, and next state
        next_states = [agent.get_state(next_obs, i) for i, agent in enumerate(agents)]
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i], done, i)

        # Update observation and step count
        obs = next_obs
        step += 1

            # Calculate and display loading progress
        if step % max(1, (env.max_steps // loading_bar_length)) == 0 or done:
            progress_percentage = min((step / env.max_steps) * 100, 100)
            loading_bar = "|" + "-" * int((step) / (env.max_steps / loading_bar_length)) + \
                          " " * (loading_bar_length - int((step) / (env.max_steps / loading_bar_length))) + "|"
            sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} {int(progress_percentage)}%")
            sys.stdout.flush()

    # Finish the loading bar at the end of the episode
    loading_bar = "|" + "-" * loading_bar_length + "|"
    sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} 100%\n")
    sys.stdout.flush()

    # Process info to update cumulative counts
    winners = info.get('winners', [])
    if winners:
        for winner in winners:
            agent_wins[winner] += 1

    eliminations = info.get('eliminations_by_agent', [0] * env.num_players)
    eliminations_per_episode.append(info.get('eliminations_by_agent', [0] * env.num_players))
    self_eliminations = info.get('self_eliminations_by_agent', [0] * env.num_players)
    self_eliminations_per_episode.append(info.get('self_eliminations_by_agent', [0] * env.num_players))

    for i in range(env.num_players):
        agent_eliminations[i] += eliminations[i]
        agent_self_eliminations[i] += self_eliminations[i]

    # Update cumulative rewards per agent
    cumulative_rewards = env.cumulative_rewards.copy()
    for i in range(env.num_players):
        cumulative_rewards_per_agent[i].append(cumulative_rewards[i])

    # Update territory information per agent
    territory_info = info.get('territory_by_agent', [0] * env.num_players)
    for i in range(env.num_players):
        territory_per_agent[i].append(territory_info[i])

    # Store episode data
    episode_rewards.append(episode_reward)
    episodes.append(episode_num)
    steps_per_episode_list.append(step)
    epsilon_values.append(agent.epsilon)

    # Calculate moving average reward
    if len(episode_rewards) >= window_size:
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
    else:
        moving_avg_rewards.append(np.mean(episode_rewards))

    # Periodic epsilon reset
    if ((episode_num + 1) % epsilon_reset_interval == 0) and epsilon_reset == True:
        agent.epsilon = epsilon_reset_value
        print(f"\nEpsilon reset to {epsilon_reset_value} at episode {episode_num + 1}")

    # Save Q-table periodically
    if (episode_num + 1) % 5000 == 0:
        for idx, agent in enumerate(agents):
            q_table_path = os.path.join(trained_model_folder, f'q_table_ag_{idx}_{episode_num + 1}.pkl')
            agent.save(q_table_path)
            print(f"Q-table for agent {idx} saved at {q_table_path}")

    # Decay epsilon after each episode
    agent.decay_epsilon()

    episode_num += 1

total_training_time = (time.time() - training_start_time) / 60

# Plotting (including the new cumulative rewards plot)
plot_training_progress(episodes, episode_rewards, moving_avg_rewards, plots_folder)
plot_epsilon_decay(episodes, epsilon_values, plots_folder)
plot_td_error(agent.td_errors, plots_folder)
plot_agent_wins(agent_wins, plots_folder)
plot_agent_eliminations(agent_eliminations, plots_folder)
plot_cumulative_self_eliminations(episodes, self_eliminations_per_episode, plots_folder)
plot_average_self_eliminations(episodes, self_eliminations_per_episode, plots_folder, window_size=window_size)
plot_cumulative_rewards(episodes, cumulative_rewards_per_agent, plots_folder)
plot_average_eliminations(episodes, eliminations_per_episode, plots_folder, window_size=window_size)
plot_territory_gained(episodes, territory_per_agent, plots_folder)

# Save the Q-table after training
for idx, agent in enumerate(agents):
    q_table_path = os.path.join(trained_model_folder, f'q_table_ag_{idx}_end.pkl')
    agent.save(q_table_path)
    print(f"Q-table for agent {idx} saved at {q_table_path}")

# Save the training information
save_training_info(training_info_file, num_episodes, steps_per_episode, agent, env.reward_config)

# Show the plots
plt.show()

# Cleanup pygame if rendering was enabled
if render_game:
    pygame.quit()

print(f"\nTotal Training Time: {total_training_time:.2f} minutes")
print("Training was Completed.")
