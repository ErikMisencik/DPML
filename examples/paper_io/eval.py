import os
import random
import sys
from time import sleep, time
import pygame  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from examples.paper_io.Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent
from examples.paper_io.algorithm.Sarsa.sarsa_agent import SARSAAgent
from examples.paper_io.algorithm.MonteCarlo.monteCarlo_agent import MCAgent
from examples.paper_io.algorithm.TD_Learning.td_learning_agent import TDAgent
from examples.paper_io.algorithm.ActorCritic.ac_agent import ACAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors

# Ak chcete počas hodnotenia zobraziť hru, nastavte render_game na True
render_game = False

def get_next_eval_folder(base_folder="evaluation_results"):
    """
    Vráti cestu k novej evaluácii. V priečinku base_folder hľadá podpriečinky vo formáte eval_<index>
    a vygeneruje novú cestu s indexom o jedna vyšším, ako je aktuálny najvyšší.
    """
    os.makedirs(base_folder, exist_ok=True)
    eval_indices = []
    for entry in os.listdir(base_folder):
        if entry.startswith("eval_") and os.path.isdir(os.path.join(base_folder, entry)):
            try:
                idx = int(entry.split("_")[-1])
                eval_indices.append(idx)
            except ValueError:
                continue
    next_index = max(eval_indices) + 1 if eval_indices else 1
    new_folder = os.path.join(base_folder, f"eval_{next_index}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def get_agent_name_from_path(path: str) -> str:
    """
    Vráti čitateľné meno agenta na základe kľúčových slov v ceste k modelu.
    """
    path_lower = path.lower()
    if "qlagent" in path_lower:
        return "Q Learning Agent"
    elif "sarsaagent" in path_lower:
        return "Sarsa Agent"
    elif "mcagent" in path_lower:
        return "MonteCarlo Agent"
    elif "tdagent" in path_lower:
        return "TD Agent"
    elif "acagent" in path_lower:
        return "ActorCritic Agent"
    else:
        return "Neznámy Agent"

def load_agent(env, model_path):
    """
    Načíta trénovaného agenta podľa cesty k modelu.
    """
    path_lower = model_path.lower()
    if "qlagent" in path_lower:
        agent = QLAgent(env, learning_rate=0, epsilon=0.0, load_only=True)
    elif "sarsaagent" in path_lower:
        agent = SARSAAgent(env, learning_rate=0, epsilon=0.0, load_only=True)
    elif "mcagent" in path_lower:
        agent = MCAgent(env, learning_rate=0, epsilon=0.0, load_only=True)
    elif "tdagent" in path_lower:
        agent = TDAgent(env, learning_rate=0, epsilon=0.0, load_only=True)
    elif "acagent" in path_lower:
        agent = ACAgent(env, learning_rate=0, epsilon=0.0, load_only=True)
    else:
        raise ValueError("Neznámy typ agenta v ceste k modelu.")
    agent.load(model_path)
    return agent

def save_evaluation_info(file_path, num_games, agent_names, agent_wins,
                         cumulative_rewards_all, territory_all, trail_all,
                         eliminations_all, self_eliminations_all, total_eliminations, total_self_eliminations):
    """
    Uloží hodnotiace štatistiky do textového súboru.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=== Hodnotiace Informácie ===\n")
        f.write(f"Počet hier: {num_games}\n\n")
        for i, name in enumerate(agent_names):
            wins = agent_wins[i]
            avg_reward = (sum(cumulative_rewards_all[i]) / len(cumulative_rewards_all[i])
                          if cumulative_rewards_all[i] else 0)
            avg_territory = (sum(territory_all[i]) / len(territory_all[i])
                             if territory_all[i] else 0)
            avg_trail = (sum(trail_all[i]) / len(trail_all[i])
                         if trail_all[i] else 0)
            avg_eliminations = (sum(eliminations_all[i]) / len(eliminations_all[i])
                                if eliminations_all[i] else 0)
            avg_self_elims = (sum(self_eliminations_all[i]) / len(self_eliminations_all[i])
                              if self_eliminations_all[i] else 0)
            f.write(f"Agent {i}: {name}\n")
            f.write(f"  - Celkové víťazstvá              : {wins}\n")
            f.write(f"  - Celkové eliminácie             : {total_eliminations[i]}\n")
            f.write(f"  - Celkové vlastné eliminácie     : {total_self_eliminations[i]}\n")
            f.write(f"  - Priemerná kumulatívna odmena     : {avg_reward:.2f}\n")
            f.write(f"  - Priemerná dĺžka stopy           : {avg_trail:.2f}\n")
            f.write(f"  - Priemerný zisk teritórií        : {avg_territory:.2f}\n")
            f.write(f"  - Priemerný počet eliminácií      : {avg_eliminations:.2f}\n")
            f.write(f"  - Priemerný počet vlastných eliminácií: {avg_self_elims:.2f}\n\n")
    print(f"Hodnotiace informácie uložené v {file_path}")

def plot_agent_wins(agent_wins, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, agent_wins, color=plt.cm.tab20.colors[:len(agent_wins)])
    plt.ylabel('Počet víťazstiev')
    plt.title('Celkové víťazstvá agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'agent_wins.png')
    plt.savefig(plot_path)
    print(f"Graf víťazstiev agentov uložený v {plot_path}")
    plt.close()

def plot_average_trail(avg_trails, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, avg_trails, color=plt.cm.tab20.colors[:len(avg_trails)])
    plt.ylabel('Priemerná dĺžka stopy')
    plt.title('Priemerná dĺžka stopy agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'average_trail.png')
    plt.savefig(plot_path)
    print(f"Graf priemernej dĺžky stopy uložený v {plot_path}")
    plt.close()

def plot_average_cumulative_reward(avg_rewards, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, avg_rewards, color=plt.cm.tab20.colors[:len(avg_rewards)])
    plt.ylabel('Priemerná kumulatívna odmena')
    plt.title('Priemerná kumulatívna odmena agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'average_cumulative_reward.png')
    plt.savefig(plot_path)
    print(f"Graf priemernej kumulatívnej odmeny uložený v {plot_path}")
    plt.close()

def plot_average_eliminations(avg_eliminations, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, avg_eliminations, color=plt.cm.tab20.colors[:len(avg_eliminations)])
    plt.ylabel('Priemerný počet eliminácií')
    plt.title('Priemerný počet eliminácií agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'average_eliminations.png')
    plt.savefig(plot_path)
    print(f"Graf priemerného počtu eliminácií uložený v {plot_path}")
    plt.close()

def plot_average_self_eliminations(avg_self_elims, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, avg_self_elims, color=plt.cm.tab20.colors[:len(avg_self_elims)])
    plt.ylabel('Priemerný počet seba eliminácií')
    plt.title('Priemerný počet seba eliminácií agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'average_self_eliminations.png')
    plt.savefig(plot_path)
    print(f"Graf priemerného počtu vlastných eliminácií uložený v {plot_path}")
    plt.close()

def plot_average_territory(avg_territory, plots_folder, agent_names):
    plt.figure(figsize=(10, 5))
    plt.bar(agent_names, avg_territory, color=plt.cm.tab20.colors[:len(avg_territory)])
    plt.xlabel('Agenti')
    plt.ylabel('Priemerný zisk teritórií')
    plt.title('Priemerný zisk teritórií agentov')
    plt.grid(axis='y')
    plot_path = os.path.join(plots_folder, 'average_territory.png')
    plt.savefig(plot_path)
    print(f"Graf priemerného zisku teritórií uložený v {plot_path}")
    plt.close()

def evaluate(agents, agent_names, num_games=10):
    """
    Spustí hodnotiace hry, zaznamená štatistiky a uloží súhrnné údaje do súboru v novej priečinku eval_<index>.
    """
    num_players = len(agents)
    color_info = assign_agent_colors(num_players)
    agent_colors = [info[0] for info in color_info]
    agent_color_names = [info[1] for info in color_info]

    # Vypíše základné informácie o hodnotiacom nastavení
    print("\nHodnotiace nastavenie:")
    for i in range(num_players):
        print(f"{agent_names[i]} má priradenú farbu {agent_color_names[i]}")
    print()

    # Inicializácia štatistík pre každého agenta
    agent_wins = [0] * num_players
    cumulative_rewards_all = [[] for _ in range(num_players)]
    territory_all = [[] for _ in range(num_players)]
    trail_all = [[] for _ in range(num_players)]
    eliminations_all = [[] for _ in range(num_players)]
    self_eliminations_all = [[] for _ in range(num_players)]
    total_eliminations = [0] * num_players
    total_self_eliminations = [0] * num_players

    # Hodnotiaca slučka
    for game_num in range(num_games):
        obs = env.reset()
        done = False
        agent_game_rewards = [0] * num_players

        while not done:
            if render_game:
                env.render(agent_colors)
            actions = [agent.get_action(obs, i) for i, agent in enumerate(agents)]
            obs, rewards, done, info = env.step(actions)
            # Akumulácia odmien počas hry
            for i in range(num_players):
                agent_game_rewards[i] += rewards[i]
            if render_game:
                sleep(0.03)

        # Získanie informácií z env.info (ak sú k dispozícii)
        winners = info.get('winners', [])
        cumulative_rewards = info.get('cumulative_rewards', agent_game_rewards)
        territory = info.get('territory_by_agent', [0] * num_players)
        average_trail = info.get('average_trail_by_agent', [0] * num_players)
        eliminations = info.get('eliminations_by_agent', [0] * num_players)
        self_elims = info.get('self_eliminations_by_agent', [0] * num_players)

        # Aktualizácia štatistík pre každého agenta
        for i in range(num_players):
            cumulative_rewards_all[i].append(cumulative_rewards[i])
            territory_all[i].append(territory[i])
            trail_all[i].append(average_trail[i])
            eliminations_all[i].append(eliminations[i])
            self_eliminations_all[i].append(self_elims[i])
            total_eliminations[i] += eliminations[i]
            total_self_eliminations[i] += self_elims[i]
        for winner in winners:
            agent_wins[winner] += 1

        # Vypísanie priebežných údajov o hre
        print(f"Hra {game_num + 1}:")
        for i in range(num_players):
            print(f"  {agent_names[i]}: kumulatívna odmena {cumulative_rewards[i]:.2f}, "
                  f"eliminácie {eliminations[i]}, vlastné eliminácie {self_elims[i]}")
        if winners:
            winner_names = [agent_names[i] for i in winners]
            print(f"  Víťazi: {', '.join(winner_names)}\n")
        else:
            print("  V tejto hre nebol víťaz.\n")

    # Vytvorenie novej priečinky pre túto evaluáciu
    evaluation_folder = get_next_eval_folder("evaluation_results")
    evaluation_file_path = os.path.join(evaluation_folder, "evaluation_info.txt")
    save_evaluation_info(evaluation_file_path, num_games, agent_names, agent_wins,
                         cumulative_rewards_all, territory_all, trail_all,
                         eliminations_all, self_eliminations_all,
                         total_eliminations, total_self_eliminations)

    # Výpočet priemerov pre jednotlivé metriky
    avg_cum_rewards = [sum(lst) / len(lst) if lst else 0 for lst in cumulative_rewards_all]
    avg_trails = [sum(lst) / len(lst) if lst else 0 for lst in trail_all]
    avg_eliminations = [sum(lst) / len(lst) if lst else 0 for lst in eliminations_all]
    avg_self_elims = [sum(lst) / len(lst) if lst else 0 for lst in self_eliminations_all]
    avg_territory = [sum(lst) / len(lst) if lst else 0 for lst in territory_all]

    # Vypísanie súhrnných údajov do terminálu
    print(f"\nHodnotiace výsledky (po {num_games} hrách):")
    for i in range(num_players):
        print(f"{agent_names[i]} (Farba: {agent_color_names[i]})")
        print(f"  - Víťazstvá: {agent_wins[i]}")
        print(f"  - Priemerná kumulatívna odmena: {avg_cum_rewards[i]:.2f}")
        print(f"  - Priemerná dĺžka stopy: {avg_trails[i]:.2f}")
        print(f"  - Priemerný zisk teritórií: {avg_territory[i]:.2f}")
        print(f"  - Priemerný počet eliminácií: {avg_eliminations[i]:.2f}")
        print(f"  - Priemerný počet vlastných eliminácií: {avg_self_elims[i]:.2f}\n")

    # Vytvorenie priečinka pre grafy v rámci aktuálnej evaluácie
    plots_folder = os.path.join(evaluation_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Generovanie grafov
    plot_agent_wins(agent_wins, plots_folder, agent_names)
    plot_average_trail(avg_trails, plots_folder, agent_names)
    plot_average_cumulative_reward(avg_cum_rewards, plots_folder, agent_names)
    plot_average_eliminations(avg_eliminations, plots_folder, agent_names)
    plot_average_self_eliminations(avg_self_elims, plots_folder, agent_names)
    plot_average_territory(avg_territory, plots_folder, agent_names)

def main():
    print("Spúšťam hodnotenie...")

    base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/best_models/"
    # Špecifikujte cesty k modelom (upravte alebo odkomentujte podľa potreby)
    model_paths = [
        # "New_BEST_S_1_Q-Learning_1/trained_model/qlagent_ag_0_end.pkl",
        # "New_BEST_S_1_SARSA_1/trained_model/sarsaagent_ag_0_end.pkl",
        # "New_BEST_S_1_MonteCarlo_1/trained_model/mcagent_ag_0_end.pkl",
        # "New_BEST_S_1_TD_1/trained_model/tdagent_ag_0_end.pkl",
        # "New_BEST_S_1_ActorCritic_1/trained_model/acagent_ag_0_end.pkl",

        "PreTrained_UPDATE_M_5/trained_model/qlagent_ag_0_end.pkl",
        "PreTrained_UPDATE_M_5/trained_model/sarsaagent_ag_1_end.pkl",
        "PreTrained_UPDATE_M_5/trained_model/mcagent_ag_2_end.pkl",
        "PreTrained_UPDATE_M_5/trained_model/tdagent_ag_3_end.pkl",
        "PreTrained_UPDATE_M_5/trained_model/acagent_ag_4_end.pkl",
    ]
    model_paths = [p for p in model_paths if p]
    num_agents = len(model_paths)

    # Inicializácia prostredia
    steps_per_episode = 350
    num_games = 10000  # Nastavte počet hodnotiacich hier podľa potreby

    global env
    env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=num_agents)

    # Načítanie agentov a ich čitateľných mien
    agents = []
    agent_names = []
    for path in model_paths:
        full_path = os.path.join(base_models_path, path)
        agents.append(load_agent(env, full_path))
        agent_names.append(get_agent_name_from_path(full_path))

    # Konfigurácia pozorovania pre každého agenta
    for i, agent in enumerate(agents):
        if agent.__class__.__name__ in ["QLAgent", "TDAgent"]:
            env.agent_observation_config[i] = {"trail": "border", "territory": "transformed"}
        elif agent.__class__.__name__ in ["SARSAAgent", "MCAgent"]:
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}
        elif agent.__class__.__name__ == "ACAgent":
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}
        else:
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}

    evaluate(agents, agent_names, num_games)

    if render_game:
        pygame.quit()

if __name__ == "__main__":
    main()
