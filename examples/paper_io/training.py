import os
import matplotlib.pyplot as plt
import numpy as np
import sys


from examples.paper_io.Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Random.random_policy import RandomPolicy  # Import the Random Policy
from examples.paper_io.algorithm.Greedy.greedy_policy import GreedyPolicy  # Import the Greedy Policy


# Create the environment
env = PaperIoEnv()

# Choose the policy (You can comment/uncomment the policy you want to use)


# policy = RandomPolicy(env)
# policy_name = 'random_policy'

policy = GreedyPolicy(env)
policy_name = 'greedy_policy'

# Training variables
num_episodes = 30  # Set the number of episodes (epochs)
steps_per_episode = 10000  # Set the number of steps for each episode
episode_rewards = []  # Store rewards per episode
moving_avg_rewards = []  # Moving average of rewards
episodes = []  # Store episode numbers for plotting
window_size = 10  # Window size for moving average

# Initialize the fixed output format
loading_bar_length = 20  # Length of the loading bar

# Initialize model count based on policy name
model_count = len([d for d in os.listdir('models') if policy_name in d])  # Count models matching the policy name
model_folder_name = f"{policy_name}_{model_count + 1}"

# Create directories for the new model
models_dir = 'models'
model_folder = os.path.join(models_dir, model_folder_name)
trained_model_folder = os.path.join(model_folder, 'trained_model')
plots_folder = os.path.join(model_folder, 'plots')

# Create directories if they don't exist
os.makedirs(trained_model_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# Print the initial header
print(f"{'Epoch':<6} {'Progress':<23}")
print("=" * 30)

# Train the model with a callback to collect data
episode_num = 0

while episode_num < num_episodes:
    obs = env.reset()
    episode_reward = 0

    # Initialize loading progress
    for step in range(steps_per_episode):
        # Use the correct list of action spaces for each player
        actions = [env.action_spaces[i].sample() for i in range(env.num_players)]
        obs, rewards, done, _ = env.step(actions)

        # Update total reward for the episode
        episode_reward += sum(rewards)

        # Calculate and display loading progress
        if (step + 1) % (steps_per_episode // loading_bar_length) == 0:
            progress_percentage = (step + 1) / steps_per_episode * 100
            loading_bar = "|" + "-" * int((step + 1) / (steps_per_episode / loading_bar_length)) + " " * (loading_bar_length - int((step + 1) / (steps_per_episode / loading_bar_length))) + "|"
            sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} {int(progress_percentage)}%")
            sys.stdout.flush()

        # If the environment signals the end of the episode, break out of the loop
        if done:
            break

    # Finish the loading bar at the end of the episode
    loading_bar = "|" + "-" * loading_bar_length + "|"
    progress_percentage = 100
    sys.stdout.write(f"\rEpoch {episode_num + 1:<5} {loading_bar} {progress_percentage}%\n")
    sys.stdout.flush()

    # Store rewards and episode data after finishing the episode
    episode_rewards.append(episode_reward)
    episodes.append(episode_num)

    # Calculate moving average reward
    if len(episode_rewards) >= window_size:
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
    else:
        moving_avg_rewards.append(np.mean(episode_rewards))

    episode_num += 1

# Plotting the training progress
plt.figure(figsize=(10, 5))
plt.plot(episodes, episode_rewards, label='Episode Reward')
plt.plot(episodes, moving_avg_rewards, label=f'Moving Average Reward (window={window_size})', color='orange')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress: Reward over Episodes')
plt.legend()
plt.grid(True)

# Save the plot to a file in the plots folder
plot_path = os.path.join(plots_folder, 'training_progress.png')
plt.savefig(plot_path)
print(f"Training progress graph saved at {plot_path}")

# Show the plot
plt.show()

print("Training was Completed.")
