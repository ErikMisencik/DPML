import os
import matplotlib.pyplot as plt
import numpy as np

def save_plot(episodes, data, xlabel, ylabel, title, file_name, plot_type='line', color=None, label=None, moving_avg=False, window_size=50):
    plt.figure(figsize=(10, 5))

    if plot_type == 'line':
        plt.plot(episodes, data, label=label if label else ylabel, linewidth=0.75, color=color)
    elif plot_type == 'scatter':
        plt.scatter(episodes, data, label=label if label else ylabel, color=color, s=3)

    if moving_avg:
        if len(data) >= window_size:
            moving_avg_data = [np.mean(data[max(0, i - window_size):i+1]) for i in range(len(data))]
            plt.plot(episodes, moving_avg_data, label=f'Moving Average (window={window_size})', color='orange', linewidth=2)
        else:
            moving_avg_data = [np.mean(data)] * len(data)
            plt.plot(episodes, moving_avg_data, label=f'Moving Average', color='orange', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(file_name)
    print(f"Plot saved at {file_name}")
    plt.close()

def plot_training_progress(episodes, episode_rewards, moving_avg_rewards, plots_folder, window_size=50):
    plot_path = os.path.join(plots_folder, 'training_progress.png')
    save_plot(episodes, episode_rewards, 'Episodes', 'Total Reward', 'Training Progress: Reward over Episodes', plot_path, moving_avg=True, window_size=window_size)

def plot_steps_per_episode(episodes, steps_per_episode_list, plots_folder):
    plot_path_steps = os.path.join(plots_folder, 'steps_per_episode.png')
    save_plot(episodes, steps_per_episode_list, 'Episodes', 'Steps', 'Steps Taken Per Episode', plot_path_steps, plot_type='scatter', color='green')

def plot_epsilon_decay(episodes, epsilon_values, plots_folder):
    plot_path_epsilon = os.path.join(plots_folder, 'epsilon_decay.png')
    save_plot(episodes, epsilon_values, 'Episodes', 'Epsilon Value', 'Epsilon Decay Over Time', plot_path_epsilon, color='purple')

def plot_td_error(td_errors, plots_folder):
    plot_path_td_error = os.path.join(plots_folder, 'td_error.png')
    save_plot(range(len(td_errors)), td_errors, 'Steps', 'TD Error', 'TD Error over Training', plot_path_td_error)

def plot_self_eliminations_per_episode(episodes, self_elims_list, plots_folder):
    plot_path = os.path.join(plots_folder, 'self_eliminations_per_episode.png')
    save_plot(episodes, self_elims_list, 'Episodes', 'Self-Eliminations', 'Self-Eliminations per Episode', plot_path)

def plot_agent_wins(agent_wins, plots_folder):
    labels = [f'Agent {i}' for i in range(len(agent_wins))]
    wins = agent_wins
    colors = plt.cm.tab20.colors[:len(agent_wins)]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, wins, color=colors)
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('Total Wins per Agent')
    plt.grid(axis='y')

    # Save the plot
    plot_path = os.path.join(plots_folder, 'agent_wins.png')
    plt.savefig(plot_path)
    print(f"Agent wins plot saved at {plot_path}")
    plt.close()

def plot_agent_eliminations(agent_eliminations, plots_folder):
    labels = [f'Agent {i}' for i in range(len(agent_eliminations))]
    eliminations = agent_eliminations
    colors = plt.cm.tab20.colors[:len(agent_eliminations)]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, eliminations, color=colors)
    plt.xlabel('Agents')
    plt.ylabel('Number of Eliminations')
    plt.title('Total Eliminations per Agent')
    plt.grid(axis='y')

    # Save the plot
    plot_path = os.path.join(plots_folder, 'agent_eliminations.png')
    plt.savefig(plot_path)
    print(f"Agent eliminations plot saved at {plot_path}")
    plt.close()

