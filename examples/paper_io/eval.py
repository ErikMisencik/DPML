import os
import random
import sys
from time import sleep
import pygame  # type: ignore

from examples.paper_io.Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Greedy.greedy_agent import GreedyAgent
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent
from examples.paper_io.algorithm.Random.random_agent import RandomAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors

# Set up the rendering flag for the environment
render_game = True  # Set to True if you want to render the game during evaluation

def get_agent_name_from_path(path: str) -> str:
    """
    Returns a friendly agent name based on filename keywords.
    You can extend this function as needed.
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
    else:
        return "Unknown Agent"

# Function to load a trained Q-learning model (Q-table)
def load_q_learning_model(env, q_table_path):
    agent = QLAgent(env, learning_rate=0, epsilon=0.0, load_only=True)  # Read-only Q-table
    agent.load(q_table_path)  # Load the Q-table from the provided path
    return agent

# Function to evaluate agents in the environment (dynamic version)
def evaluate(agents, agent_names, num_games=10):
    num_players = len(agents)
    # Assign random colors based on the number of players
    color_info = assign_agent_colors(num_players)
    agent_colors = [info[0] for info in color_info]      # For rendering (RGB values)
    agent_color_names = [info[1] for info in color_info]   # For logging (color names)

    # Print the evaluation setup
    print("\nEvaluation Setup:")
    for i in range(num_players):
        print(f"{agent_names[i]} is assigned color {agent_color_names[i]}")
    print()

    agent_wins = [0 for _ in range(num_players)]  # Track wins per agent

    for game_num in range(num_games):
        obs = env.reset()  # Reset the environment for a new game
        done = False
        agent_game_rewards = [0 for _ in range(num_players)]

        while not done:
            if render_game:
                env.render(agent_colors)

            # Dynamically get actions for each agent based on their index
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.get_action(obs, i))
            obs, rewards, done, info = env.step(actions)

            # Accumulate rewards
            for i in range(num_players):
                agent_game_rewards[i] += rewards[i]

            if render_game:
                sleep(0.01)

        # After the game ends, update win counts and print game results
        winners = info.get('winners', [])
        cumulative_rewards = info.get('cumulative_rewards', [0] * num_players)
        for i in winners:
            agent_wins[i] += 1

        print(f"Game {game_num + 1}:")
        for i in range(num_players):
            print(f"{agent_names[i]} cumulative reward: {cumulative_rewards[i]:.2f}")
        if winners:
            winner_names = [agent_names[i] if i < num_players else f"Agent {i}" for i in winners]
            print(f"Winner(s): {', '.join(winner_names)}\n")
        else:
            print("No winner this game.\n")

    # Final evaluation results
    print(f"\nEvaluation Results (over {num_games} games):")
    for i in range(num_players):
        print(f"{agent_names[i]} (Color: {agent_color_names[i]}) wins: {agent_wins[i]}")

def main():
    print("Starting evaluation...")

    base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/"
    # base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/compare_models/"


    # Specify model paths.
    model_paths = [
        # "New_M_2_Q-Learning_TD_5/trained_model/qlagent_ag_0_end.pkl",
        # "New_M_2_Q-Learning_TD_5/trained_model/tdagent_ag_1_end.pkl",

        # "New_M_2_Q-Learning_TD_6_BIG/trained_model/qlagent_ag_0_end.pkl",
        # "New_M_2_Q-Learning_TD_6_BIG/trained_model/tdagent_ag_1_end.pkl",

        # "New_M_2_Q-Learning_TD_8/trained_model/qlagent_ag_0_30000.pkl",
        # "New_M_2_Q-Learning_TD_8/trained_model/tdagent_ag_1_30000.pkl",

        # "New_M_2_Q-Learning_TD_10_BESTGRAPHS/trained_model/qlagent_ag_0_30000.pkl",
        # "New_M_2_Q-Learning_TD_10_BESTGRAPHS/trained_model/tdagent_ag_1_30000.pkl",

        # "New_M_2_Q-Learning_TD_11/trained_model/qlagent_ag_0_30000.pkl",
        # "New_M_2_Q-Learning_TD_11/trained_model/tdagent_ag_1_30000.pkl",

        # "New_S_1_TD_1/trained_model/tdagent_ag_0_30000.pkl",
        # "New_S_1_TD_2/trained_model/tdagent_ag_0_30000.pkl",
        # "New_S_1_TD_4/trained_model/tdagent_ag_0_end.pkl",
        # "New_S_1_SARSA_3/trained_model/sarsaagent_ag_0_end.pkl",


        # "New_M_2_SARSA_MonteCarlo_1/trained_model/mcagent_ag_1_end.pkl",
        # "New_M_2_SARSA_MonteCarlo_1/trained_model/sarsaagent_ag_0_end.pkl",

        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/mcagent_ag_2_end.pkl",
        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/qlagent_ag_0_end.pkl",

        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/mcagent_ag_2_end.pkl",
        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/qlagent_ag_0_end.pkl",
        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/sarsaagent_ag_1_end.pkl",
        # "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/tdagent_ag_3_end.pkl"

        # "TD_BEAST/trained_model/tdagent_ag_1_end.pkl"

        # "New_S_1_ActorCritic_1_LR_0,001/trained_model/acagent_ag_0_end.pkl",
        # "New_S_1_ActorCritic_1_LR_0,001_30k/trained_model/acagent_ag_0_end.pkl",
        # "New_S_1_MonteCarlo_1_LR_0,01/trained_model/mcagent_ag_0_end.pkl",
        # "New_S_1_Q-Learning_1_LR_0,0001/trained_model/qlagent_ag_0_end.pkl",
        #  "New_S_1_SARSA_2_LR_0,00001/trained_model/sarsaagent_ag_0_end.pkl",
        # "New_S_1_TD_5/trained_model/tdagent_ag_0_end.pkl",

        # "New_REWARD_S_1_TD_1/trained_model/tdagent_ag_0_end.pkl",

        #GEN 2
        # "New_GEN2_S_1_TD_14/trained_model/tdagent_ag_0_end.pkl",
        # "New_GEN2_S_1_SARSA_2/trained_model/sarsaagent_ag_0_end.pkl",
        # "New_GEN2_S_1_Q-Learning_3/trained_model/qlagent_ag_0_end.pkl",
        # "New_GEN2_S_1_ActorCritic_1_500P_ELIMINATION/trained_model/acagent_ag_0_end.pkl",
        # "New_GEN2_S_1_MonteCarlo_1/trained_model/mcagent_ag_0_end.pkl",

        #GEN 2 BEST
        # "New_GEN2_S_1_SARSA_X_TOPKA/trained_model/sarsaagent_ag_0_end.pkl",

        #DF DIscount Factor
        # "New_DF_S_1_TD_1/trained_model/tdagent_ag_0_end.pkl",
        # "New_DF_S_2_ActorCritic_2/trained_model/acagent_ag_0_end.pkl",
        # "New_DF_S_1_MonteCarlo_1/trained_model/mcagent_ag_0_end.pkl",
        # "New_DF_S_1_Q-Learning_1/trained_model/qlagent_ag_0_end.pkl",
        # "New_DF_S_1_SARSA_1/trained_model/sarsaagent_ag_0_end.pkl",

        #DF DIscount Factor 20
        # "New_DF20_S_1_MonteCarlo_1/trained_model/mcagent_ag_0_end.pkl",
        "New_DF20_S_1_Q-Learning_1/trained_model/qlagent_ag_0_end.pkl",
        # "New_DF20_S_1_SARSA_3/trained_model/sarsaagent_ag_0_end.pkl",
        # "New_DF20_S_1_TD_1/trained_model/tdagent_ag_0_end.pkl",
        # "New_DF20_S_1_ActorCritic_1/trained_model/acagent_ag_0_end.pkl",


    ]
    # Filter out any empty paths in case you want to use less than 4 models.
    model_paths = [p for p in model_paths if p]

    num_agents = len(model_paths)

    # Initialize the environment with the appropriate number of players
    steps_per_episode = 500  # Use the same max_steps as in training
    global env
    env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=num_agents)

    # Load agents and their names dynamically
    agents = []
    agent_names = []
    for path in model_paths:
        full_path = os.path.join(base_models_path, path)
        agents.append(load_q_learning_model(env, full_path))
        agent_names.append(get_agent_name_from_path(full_path))

    num_games = 10  # Adjust the number of evaluation games as needed

    # Evaluate the agents using the dynamic evaluate function
    evaluate(agents, agent_names, num_games)

    if render_game:
        pygame.quit()

if __name__ == "__main__":
    main()
