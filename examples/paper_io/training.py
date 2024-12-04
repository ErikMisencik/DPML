import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import sys
import time  
from examples.paper_io.algorithm.MonteCarlo.monteCarlo_agent import MCAgent
from examples.paper_io.algorithm.Sarsa.sarsa_agent import SARSAAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors
from examples.paper_io.utils.plots import (
    plot_average_self_eliminations, plot_cumulative_rewards, plot_cumulative_self_eliminations, plot_epsilon_decay, plot_td_error,
    plot_training_progress, plot_agent_wins, plot_agent_eliminations, plot_average_eliminations, plot_territory_gained
)
import pygame  # type: ignore # Import pygame for rendering only if necessary

from Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent
 


# Explicit Q-table paths for LOADING pre-trained models
explicit_q_table_paths = {
    0: os.path.join('models', 'PreTrained_S_QLAgent_4', 'trained_model', 'q_table_qlearning_ag_0_end.pkl'),
    1: os.path.join('models', 'PreTrained_S_SARSA_2', 'trained_model', 'q_table_sarsa_ag_1_end.pkl'),
}

# Selection of algorithms to train
algorithm_config = {
    "Q-Learning": True,   # Train Q-Learning agents
    "SARSA": False,        # Train SARSA agents
    "MonteCarlo": True,  # Train Monte Carlo agents
}

# Choose algorithm and initialize agents
num_agents = 2 
render_game = False  # Set to True if you want to render the game during training 
load_existing_model = False  # Set to True to load an existing model
partial_observability = False  # Set to True for partial observability
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
    num_episodes = 10000         # Full training length
    epsilon = 1.0                  # High exploration at start
    learning_rate = 0.0025          # Standard learning rate for initial training
    epsilon_reset = True          # No epsilon reset for initial training
    epsilon_reset_value = 0.50     # Not used if epsilon_reset is False
    epsilon_reset_interval = 5000  # Not used if epsilon_reset is False
    epsilon_decay = 0.9992         # Standard decay rate
    min_epsilon = 0.05              # Minimum exploration rate


env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=num_agents, partial_observability=partial_observability)

agents = []
enabled_algorithms = [key for key, value in algorithm_config.items() if value]
num_algorithms = len(enabled_algorithms)
agents_per_algorithm = max(1, num_agents // num_algorithms)

for algo in enabled_algorithms:
    if algo == "Q-Learning":
        agents += [QLAgent(env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon)
                   for _ in range(agents_per_algorithm)]
    elif algo == "SARSA":
        agents += [SARSAAgent(env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon)
                   for _ in range(agents_per_algorithm)]
    elif algo == "MonteCarlo":
        agents += [MCAgent(env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon)
                   for _ in range(agents_per_algorithm)]

# Assign each agent its type (for naming and tracking purposes)
agent_types = [agent.__class__.__name__ for agent in agents]

policy_name = "PreTrained" if load_existing_model else "New"  
policy_name += f"{'_P' if partial_observability else ''}"
policy_name += f"_{'M' if num_agents > 1 else 'S'}_{num_agents}_"   
policy_name += f"{'_'.join(enabled_algorithms)}"


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


# Assign random colors to agents
color_info = assign_agent_colors(env.num_players)
agent_colors = [info[0] for info in color_info]  # RGB values for rendering
agent_color_names = [info[1] for info in color_info]  # Color names for logging

# File to save training details
training_info_file = os.path.join(model_folder, 'training_info.txt')

# Load models if needed
loaded_q_paths = []
for idx, agent in enumerate(agents):
    q_table_path = explicit_q_table_paths.get(idx, None)

    if q_table_path and os.path.exists(q_table_path):
        agent.load(q_table_path)
        print(f"Loaded Q-table for {agent.__class__.__name__} agent {idx} from {q_table_path}")
    else:
        print(f"No Q-table found for {agent.__class__.__name__} agent {idx}. Starting from scratch.")

    # Append path (or None) to loaded_q_paths for tracking
    loaded_q_paths.append(q_table_path)

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
def save_training_info(file_path, num_episodes, steps_per_episode, agents, reward_config, loaded_q_paths):
    with open(file_path, 'w') as f:
        # General Training Info
        f.write("=== Training Information ===\n")
        f.write(f"Objective: The agents aim to maximize territory gain.\n")
        f.write(f"Policy Name           : {policy_name}\n")
        f.write(f"Partial Observability : {partial_observability}\n")
        f.write(f"Number of Episodes    : {num_episodes}\n")
        f.write(f"Max Steps per Episode : {steps_per_episode}\n")
        f.write("\n")

        # Training Parameters
        f.write("=== Training Parameters ===\n")
        f.write(f"Learning Rate         : {agents[0].learning_rate if agents else 'N/A'}\n")
        f.write(f"Discount Factor       : {agents[0].discount_factor if agents else 'N/A'}\n")
        f.write(f"Initial Epsilon       : {epsilon}\n")
        f.write(f"Final Epsilon         : {agents[0].epsilon if agents else 'N/A'}\n")
        f.write(f"Epsilon Decay Rate    : {epsilon_decay}\n")
        f.write(f"Minimum Epsilon       : {min_epsilon}\n")
        f.write(f"Epsilon Reset Every   : {epsilon_reset_interval if epsilon_reset else 'N/A'} episodes\n")
        f.write(f"Epsilon Reset Value   : {epsilon_reset_value if epsilon_reset else 'N/A'}\n")
        f.write("\n")

        f.write("=== Agent-Specific Information ===\n")
        for idx, agent in enumerate(agents):
            avg_cumulative_reward = (np.mean(cumulative_rewards_per_agent[idx])
                                     if cumulative_rewards_per_agent[idx] else 0)
            f.write(f"Agent {idx} ({agent.__class__.__name__}):\n")
            f.write(f"  - Avg Cumulative Reward : {avg_cumulative_reward:.2f}\n")
            f.write(f"  - Wins                  : {agent_wins[idx]}\n")
            f.write(f"  - Eliminations          : {agent_eliminations[idx]}\n")
            f.write(f"  - Self-Eliminations     : {agent_self_eliminations[idx]}\n")
            f.write(f"  - Color Assigned        : {agent_color_names[idx]}\n")
            if load_existing_model and loaded_q_paths[idx]:
                f.write(f"  - Loaded Q-Table Path   : {loaded_q_paths[idx]}\n")
            else:
                f.write("  - Loaded Q-Table Path   : None (Training from scratch)\n")
            f.write("\n")

        # Reward Configuration
        f.write("=== Reward Information ===\n")
        for key, value in reward_config.items():
            reward_name = key.replace('_', ' ').capitalize()
            f.write(f"{reward_name:25}: {value}\n")
    
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
         # Update each agent with its respective state, action, reward, and next state
        next_states = [agent.get_state(next_obs, i) for i, agent in enumerate(agents)]
        next_actions = [agent.get_action(next_obs, i) for i, agent in enumerate(agents)]

        for i, agent in enumerate(agents):
            if isinstance(agent, SARSAAgent):
                agent.update(states[i], actions[i], rewards[i], next_states[i], next_actions[i], done, i)
            elif not isinstance(agent, MCAgent):  # Q-Learning and other step-based algorithms
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
            q_table_path = os.path.join(trained_model_folder, f'{agent.__class__.__name__.lower()}_ag_{idx}_{episode_num + 1}.pkl')
            agent.save(q_table_path)
            print(f"Q-table for {agent.__class__.__name__} agent {idx} saved at {q_table_path}")

    # Perform Monte Carlo updates after the episode ends
    for agent in agents:
        if isinstance(agent, MCAgent):
            agent.update()
    # Decay epsilon after each episode
    agent.decay_epsilon()

    episode_num += 1

total_training_time = (time.time() - training_start_time) / 60

agent_names = [agent.__class__.__name__ for agent in agents]
td_errors = []
for agent in agents:
    if hasattr(agent, "td_errors"):
        td_errors.extend(agent.td_errors)

plot_training_progress(episodes, episode_rewards, moving_avg_rewards, plots_folder)
plot_epsilon_decay(episodes, epsilon_values, plots_folder)
if td_errors:
    plot_td_error(td_errors, plots_folder)
plot_agent_wins(agent_wins, plots_folder, agent_names)
plot_agent_eliminations(agent_eliminations, plots_folder, agent_names)
plot_cumulative_self_eliminations(episodes, self_eliminations_per_episode, plots_folder, agent_names)
plot_average_self_eliminations(episodes, self_eliminations_per_episode, plots_folder, window_size=window_size)
plot_cumulative_rewards(episodes, cumulative_rewards_per_agent, plots_folder, agent_names)
plot_average_eliminations(episodes, eliminations_per_episode, plots_folder, agent_names, window_size=window_size)
plot_territory_gained(episodes, territory_per_agent, plots_folder, agent_names)

# Save models dynamically
for idx, agent in enumerate(agents):
    model_path = os.path.join(trained_model_folder, f'{agent.__class__.__name__.lower()}_ag_{idx}_end.pkl')
    agent.save(model_path)
    print(f"Model for {agent.__class__.__name__} agent {idx} saved at {model_path}")

# Save the training information
save_training_info(training_info_file, num_episodes, steps_per_episode, agents, env.reward_config, loaded_q_paths)

# Show the plots
plt.show()

# Cleanup pygame if rendering was enabled
if render_game:
    pygame.quit()

print(f"\nTotal Training Time: {total_training_time:.2f} minutes")
print("Training was Completed.")
