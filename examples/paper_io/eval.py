import os 
import random
import sys
from time import sleep
import pygame  # type: ignore

from examples.paper_io.Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent
from examples.paper_io.algorithm.Sarsa.sarsa_agent import SARSAAgent
from examples.paper_io.algorithm.MonteCarlo.monteCarlo_agent import MCAgent
from examples.paper_io.algorithm.TD_Learning.td_learning_agent import TDAgent
from examples.paper_io.algorithm.ActorCritic.ac_agent import ACAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors

# Set up the rendering flag for the environment
render_game = True  # Set to True if you want to render the game during evaluation

def get_agent_name_from_path(path: str) -> str:
    """
    Returns a friendly agent name based on filename keywords.
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
        return "Unknown Agent"

# Generic function to load a trained agent based on model path.
def load_agent(env, model_path):
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
        raise ValueError("Unknown agent type in model path.")
    agent.load(model_path)
    return agent

# Function to evaluate agents in the environment (dynamic version)
def evaluate(agents, agent_names, num_games=10):
    num_players = len(agents)
    color_info = assign_agent_colors(num_players)
    agent_colors = [info[0] for info in color_info]      
    agent_color_names = [info[1] for info in color_info]   

    print("\nEvaluation Setup:")
    for i in range(num_players):
        print(f"{agent_names[i]} is assigned color {agent_color_names[i]}")
    print()

    agent_wins = [0 for _ in range(num_players)]  

    for game_num in range(num_games):
        obs = env.reset()  
        done = False
        agent_game_rewards = [0 for _ in range(num_players)]

        while not done:
            if render_game:
                env.render(agent_colors)

            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.get_action(obs, i))
            obs, rewards, done, info = env.step(actions)

            for i in range(num_players):
                agent_game_rewards[i] += rewards[i]

            if render_game:
                sleep(0.01)

        winners = info.get('winners', [])
        cumulative_rewards = info.get('cumulative_rewards', [0] * num_players)
        for i in winners:
            agent_wins[i] += 1

        print(f"Game {game_num + 1}:")
        for i in range(num_players):
            print(f"{agent_names[i]} cumulative reward: {cumulative_rewards[i]:.2f}")
        if winners:
            winner_names = [agent_names[i] for i in winners]
            print(f"Winner(s): {', '.join(winner_names)}\n")
        else:
            print("No winner this game.\n")

    print(f"\nEvaluation Results (over {num_games} games):")
    for i in range(num_players):
        print(f"{agent_names[i]} (Color: {agent_color_names[i]}) wins: {agent_wins[i]}")

def main():
    print("Starting evaluation...")

    base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/"

    # Specify model paths.
    model_paths = [
        # "New_UPDATE_S_1_MonteCarlo_1/trained_model/mcagent_ag_0_end.pkl",
      
        # "New_UPDATE_S_1_TD_1/trained_model/tdagent_ag_0_end.pkl",
        "New_UPDATE_S_1_ActorCritic_1/trained_model/acagent_ag_0_end.pkl",
        # "New_UPDATE_S_1_Q-Learning_1/trained_model/qlagent_ag_0_5000.pkl",


        # BEST
        #   "New_UPDATE_S_1_ActorCritic_1/trained_model/acagent_ag_0_end.pkl",
        # "New_X_BORDERTRAIL_TRAILAVG_S_1_TD_2/trained_model/tdagent_ag_0_end.pkl",
        # "New_UPDATE_S_1_SARSA_1/trained_model/sarsaagent_ag_0_end.pkl",
        # "New_X_BORDERTRAIL_TRAILAVG_S_1_Q-Learning_2/trained_model/qlagent_ag_0_end.pkl",
 
    ]
    model_paths = [p for p in model_paths if p]
    num_agents = len(model_paths)

    # Initialize the environment.
    steps_per_episode = 500  
    global env
    env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=num_agents)
    
    # Load agents and their names dynamically.
    agents = []
    agent_names = []
    for path in model_paths:
        full_path = os.path.join(base_models_path, path)
        agents.append(load_agent(env, full_path))
        agent_names.append(get_agent_name_from_path(full_path))
    
    # Set per-agent observation configurations based on agent type.
    for i, agent in enumerate(agents):
        if agent.__class__.__name__ in ["QLAgent", "TDAgent"]:
            env.agent_observation_config[i] = {"trail": "border", "territory": "transformed"}
        elif agent.__class__.__name__ in ["SARSAAgent", "MCAgent"]:
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}
        elif agent.__class__.__name__ == "ACAgent":
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}
        else:
            env.agent_observation_config[i] = {"trail": "negative", "territory": "transformed"}
    
    num_games = 10  # Adjust as needed
    evaluate(agents, agent_names, num_games)

    if render_game:
        pygame.quit()

if __name__ == "__main__":
    main()
