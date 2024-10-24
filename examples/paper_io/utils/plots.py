# plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

def save_plot(episodes, data, xlabel, ylabel, title, file_name, plot_type='line', color=None, label=None, moving_avg=False, window_size=50):
        plt.figure(figsize=(10, 5))
    
        if plot_type == 'line':
            plt.plot(episodes, data, label=label if label else ylabel, linewidth=0.75, color=color)
        elif plot_type == 'scatter':
            plt.scatter(episodes, data, label=label if label else ylabel, color=color, s=3)

        # Correct moving average calculation
        if moving_avg:
            if len(data) >= window_size:
                moving_avg_data = [np.mean(data[max(0, i - window_size):i+1]) for i in range(len(data))]
                plt.plot(episodes, moving_avg_data, label=f'Moving Average (window={window_size})', color='orange', linewidth=2)
            else:
                moving_avg_data = [np.mean(data)] * len(data)
                plt.plot(episodes, moving_avg_data, label=f'Moving Average (window={window_size})', color='orange', linewidth=2)

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

def plot_win_loss_pie(win_loss_rates, plots_folder):
    # Calculate the total wins and losses
    total_wins = win_loss_rates.count(1)
    total_losses = win_loss_rates.count(0)
    total_games = total_wins + total_losses

    # Data for pie chart
    labels = ['Wins', 'Losses']
    sizes = [total_wins, total_losses]
    colors = ['green', 'red']  # You can choose different colors if needed
    explode = (0.1, 0)  # Slightly "explode" the win section for emphasis

    # Create a pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is drawn as a circle.

    # Add a title with the exact numbers of wins and losses
    plt.title(f'Win/Loss Ratio\nWins: {total_wins}, Losses: {total_losses}, Total: {total_games}')

    # Save the pie chart to a file
    plot_path_pie = os.path.join(plots_folder, 'win_loss_pie_chart.png')
    plt.savefig(plot_path_pie, bbox_inches='tight')
    print(f"Win/Loss pie chart saved at {plot_path_pie}")

    plt.close()
# Example Usage:
# plot_training_progress(episodes, episode_rewards, moving_avg_rewards, plots_folder)
# plot_steps_per_episode(episodes, steps_per_episode_list, plots_folder)
# plot_epsilon_decay(episodes, epsilon_values, plots_folder)
# plot_td_error(agent.td_errors, plots_folder)
# plot_win_loss_rate(episodes, win_loss_rates, plots_folder)
