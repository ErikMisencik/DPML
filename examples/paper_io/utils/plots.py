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
            plt.plot(episodes, moving_avg_data, label=f'Kĺzavý priemer (okno={window_size})', color='orange', linewidth=2)
        else:
            moving_avg_data = [np.mean(data)] * len(data)
            plt.plot(episodes, moving_avg_data, label='Kĺzavý priemer', color='orange', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(file_name)
    print(f"Plot saved at {file_name}")
    plt.close()


# Update plot functions to include algorithm names and Slovak translations
def plot_training_progress(episodes, episode_rewards, moving_avg_rewards, plots_folder, window_size=50):
    plot_path = os.path.join(plots_folder, 'training_progress.png')
    save_plot(episodes, episode_rewards, 'Epizódy', 'Celková odmena', 'Vývoj tréningu: Odmena počas epizód', plot_path, moving_avg=True, window_size=window_size)

def plot_epsilon_decay(episodes, epsilon_values, plots_folder):
    plot_path_epsilon = os.path.join(plots_folder, 'epsilon_decay.png')
    save_plot(episodes, epsilon_values, 'Epizódy', 'Hodnota epsilonu', 'Úbytok epsilonu počas času', plot_path_epsilon, color='purple')

def plot_td_error(td_errors, plots_folder):
    plot_path_td_error = os.path.join(plots_folder, 'td_error.png')
    save_plot(range(len(td_errors)), td_errors, 'Kroky', 'TD chyba', 'TD chyba počas tréningu', plot_path_td_error)

def plot_cumulative_self_eliminations(episodes, self_eliminations_per_episode, plots_folder, agent_names):
    self_elims_array = np.array(self_eliminations_per_episode)
    num_agents = self_elims_array.shape[1]

    cumulative_self_elims = np.cumsum(self_elims_array, axis=0)

    plt.figure(figsize=(10, 6))
    for agent_idx in range(num_agents):
        plt.plot(episodes, cumulative_self_elims[:, agent_idx], label=agent_names[agent_idx])
    plt.xlabel('Epizódy')
    plt.ylabel('Kumulatívne sebazlikvidácie')
    plt.title('Kumulatívne sebazlikvidácie agentov počas epizód')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, 'cumulative_self_eliminations.png')
    plt.savefig(plot_path)
    print(f"Cumulative self-eliminations plot saved at {plot_path}")
    plt.close()

def plot_average_self_eliminations(episodes, self_eliminations_per_episode, plots_folder, window_size=50):
    self_elims_array = np.array(self_eliminations_per_episode)
    total_self_elims_per_episode = self_elims_array.sum(axis=1)

    moving_avg_self_elims = np.convolve(total_self_elims_per_episode, np.ones(window_size) / window_size, mode='same')

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_self_elims_per_episode, label='Celkové sebazlikvidácie', linewidth=0.75)
    plt.plot(episodes, moving_avg_self_elims, label=f'Kĺzavý priemer (okno={window_size})', color='orange', linewidth=2)
    plt.xlabel('Epizódy')
    plt.ylabel('Seba-eliminácie')
    plt.title('Seba-eliminácie počas epizód s kĺzavým priemerom')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, 'average_self_eliminations.png')
    plt.savefig(plot_path)
    print(f"Average self-eliminations plot saved at {plot_path}")
    plt.close()

def plot_agent_wins(agent_wins, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, agent_wins, color=plt.cm.tab20.colors[:len(agent_wins)])
    plt.xlabel('Agenti')
    plt.ylabel('Počet víťazstiev')
    plt.title('Celkové víťazstvá agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'agent_wins.png')
    plt.savefig(plot_path)
    print(f"Agent wins plot saved at {plot_path}")
    plt.close()

def plot_agent_eliminations(agent_eliminations, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, agent_eliminations, color=plt.cm.tab20.colors[:len(agent_eliminations)])
    plt.xlabel('Agenti')
    plt.ylabel('Počet eliminácií')
    plt.title('Celkové eliminácie agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'agent_eliminations.png')
    plt.savefig(plot_path)
    print(f"Agent eliminations plot saved at {plot_path}")
    plt.close()

def plot_cumulative_rewards(episodes, cumulative_rewards_per_agent, plots_folder, agent_names):
    plt.figure(figsize=(12, 6))

    # Line styles and markers for better differentiation
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']

    for i in range(len(cumulative_rewards_per_agent)):
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]

        # Plot with reduced marker density
        plt.plot(
            episodes,
            cumulative_rewards_per_agent[i],
            label=agent_names[i],
            linestyle=style,
            marker=marker,
            markevery=200,  # Show markers every 200 episodes
            linewidth=2,
            alpha=0.6  # More transparency
        )

    plt.xlabel('Epizódy')
    plt.ylabel('Kumulatívna odmena')
    plt.title('Kumulatívne odmeny agentov')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, 'cumulative_rewards_per_agent_combined.png'))
    plt.close()

def plot_average_eliminations(episodes, eliminations_per_episode, plots_folder, agent_names, window_size=50):
    elims_array = np.array(eliminations_per_episode)
    moving_avg_elims = np.array([
        np.convolve(elims_array[:, i], np.ones(window_size) / window_size, mode='same')
        for i in range(elims_array.shape[1])
    ])

    plt.figure(figsize=(10, 6))
    for i in range(elims_array.shape[1]):
        plt.plot(episodes, moving_avg_elims[i], label=f'{agent_names[i]} (kĺzavý priemer)')

    plt.xlabel('Epizódy')
    plt.ylabel('Priemerné eliminácie')
    plt.title('Priemerné eliminácie agentov s kĺzavým priemerom')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, 'average_eliminations_per_agent.png')
    plt.savefig(plot_path)
    print(f"Average eliminations plot saved at {plot_path}")
    plt.close()

def plot_territory_gained(episodes, territory_per_agent, plots_folder, agent_names):
    plt.figure(figsize=(12, 6))

    # Line styles and markers for better differentiation
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']

    territory_array = np.array(territory_per_agent)

    for i in range(territory_array.shape[0]):
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]

        # Plot with reduced marker density
        plt.plot(
            episodes,
            territory_array[i],
            label=agent_names[i],
            linestyle=style,
            marker=marker,
            markevery=200,  # Show markers every 200 episodes
            linewidth=2,
            alpha=0.6  # More transparency
        )

    plt.xlabel('Epizódy')
    plt.ylabel('Získané územie')
    plt.title('Získané územie agentov počas epizód')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_folder, 'territory_gained_per_agent_combined.png')
    plt.savefig(plot_path)
    plt.close()

def plot_average_trail(episodes, avg_trail_data, directory, agent_names):
    plt.figure()
    skip = 5  
    for i in range(len(agent_names)):
        plt.plot(episodes[::skip], 
                 avg_trail_data[i][::skip], 
                 label=f'Agent {i} - {agent_names[i]}', 
                 linewidth=1, 
                 alpha=0.6)
    plt.xlabel('Epizódy')
    plt.ylabel('Priemerná Dĺžka Trasy')
    plt.legend()
    plt.title('Priemerná Dĺžka Trasy za Epizódu')
    plt.savefig(os.path.join(directory, 'average_trail_by_agent.png'))
    plt.close()   

def plot_average_territory_increase(episodes, territory_increase_data, directory, agent_names):
    plt.figure()
    for i in range(len(agent_names)):
        plt.plot(episodes, territory_increase_data[i], label=f'Agent {i} - {agent_names[i]}', linewidth=1, alpha=0.6)
    plt.xlabel('Epizódy')
    plt.ylabel('Zvýšenie územia')
    plt.title('Zvýšenie územia na epizódu')
    plt.legend()
    plt.savefig(os.path.join(directory, 'average_territory_increase_by_agent.png'))
    plt.close()