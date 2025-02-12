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

# Initialize the environment with rendering enabled and max_steps
steps_per_episode = 300  # Use the same max_steps as in training
env = PaperIoEnv(render=render_game, max_steps=steps_per_episode, num_players=4)

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
def load_q_learning_model(q_table_path):
    agent = QLAgent(env, learning_rate=0, epsilon=0.0, load_only=True)  # Read-only Q-table
    agent.load(q_table_path)  # Load the Q-table from the provided path
    return agent

# Function to evaluate agents in the environment
def evaluate(agent1, agent1_name, agent2, agent2_name, agent3, agent3_name, agent4, agent4_name, num_games=10):
    # Assign random colors to the agents
    color_info = assign_agent_colors(4)
    agent_colors = [info[0] for info in color_info]  # Extract only RGB values for rendering
    agent_color_names = [info[1] for info in color_info]  # Extract color names for logging

    # Print agent colors at the start with both name and RGB tuple
    print("\nEvaluation Setup:")
    print(f"{agent1_name} is assigned color {agent_color_names[0]}")
    print(f"{agent2_name} is assigned color {agent_color_names[1]}")
    print(f"{agent3_name} is assigned color {agent_color_names[2]}")
    print(f"{agent4_name} is assigned color {agent_color_names[3]}\n")

    agent_wins = [0 for _ in range(env.num_players)]  # Track wins per agent

    for game_num in range(num_games):
        obs = env.reset()  # Reset the environment for a new game
        done = False

        # Reset cumulative rewards for this game
        agent_game_rewards = [0 for _ in range(env.num_players)]

        while not done:
            # Render the game, passing player colors
            if render_game:
                env.render(agent_colors)

            # Get actions for each agent
            action_agent1 = agent1.get_action(obs, 0)  # Agent 1 is at index 0
            action_agent2 = agent2.get_action(obs, 1)  # Agent 2 is at index 1
            action_agent3 = agent3.get_action(obs, 2)  # Agent 3 is at index 2
            action_agent4 = agent4.get_action(obs, 3)  # Agent 4 is at index 3

            # Combine actions from all agents into one list
            actions = [action_agent1, action_agent2, action_agent3, action_agent4]

            # Take a step in the environment
            obs, rewards, done, info = env.step(actions)

            # Accumulate rewards for each agent
            for i in range(env.num_players):
                agent_game_rewards[i] += rewards[i]

            if render_game:
                sleep(0.05)  # Adjust the sleep time as needed

        # After the game ends, get the winner from info
        winners = info.get('winners', [])
        cumulative_rewards = info.get('cumulative_rewards', [0] * env.num_players)

        # Update win counts based on winners
        for i in winners:
            agent_wins[i] += 1

        # Print the result of the current game
        print(f"Game {game_num + 1}:")
        for i in range(env.num_players):
            if i == 0:
                agent_name = agent1_name
            elif i == 1:
                agent_name = agent2_name
            elif i == 2:
                agent_name = agent3_name
            elif i == 3:
                agent_name = agent4_name
            else:
                agent_name = f"Agent {i}"
            print(f"{agent_name} cumulative reward: {cumulative_rewards[i]:.2f}")
        if winners:
            winner_names = []
            for i in winners:
                if i == 0:
                    winner_names.append(agent1_name)
                elif i == 1:
                    winner_names.append(agent2_name)
                elif i == 2:
                    winner_names.append(agent3_name)
                elif i == 3:
                    winner_names.append(agent4_name)
                else:
                    winner_names.append(f"Agent {i}")
            print(f"Winner(s): {', '.join(winner_names)}\n")
        else:
            print("No winner this game.\n")

    # Final evaluation results
    print(f"\nEvaluation Results (over {num_games} games):")
    for i in range(env.num_players):
        if i == 0:
            agent_name = agent1_name
        elif i == 1:
            agent_name = agent2_name
        elif i == 2:
            agent_name = agent3_name
        elif i == 3:
            agent_name = agent4_name
        else:
            agent_name = f"Agent {i}"
        print(f"{agent_name} (Color: {agent_color_names[i]}) wins: {agent_wins[i]}")

# Main evaluation function
def main():
    print("Starting evaluation...")

    base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/"

    saved_model_path1 = "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/mcagent_ag_2_end.pkl"
    saved_model_path2 = "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/qlagent_ag_0_end.pkl"
    saved_model_path3 = "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/sarsaagent_ag_1_end.pkl"
    saved_model_path4 = "New_M_4_Q-Learning_SARSA_MonteCarlo_TD_2/trained_model/tdagent_ag_3_end.pkl"

    q_table_path_agent1 = os.path.join(base_models_path, saved_model_path1)
    q_table_path_agent2 = os.path.join(base_models_path, saved_model_path2)
    q_table_path_agent3 = os.path.join(base_models_path, saved_model_path3)
    q_table_path_agent4 = os.path.join(base_models_path, saved_model_path4)

    # Load agents
    agent1 = load_q_learning_model(q_table_path_agent1)
    agent2 = load_q_learning_model(q_table_path_agent2)
    agent3 = load_q_learning_model(q_table_path_agent3)
    agent4 = load_q_learning_model(q_table_path_agent4)

    agent1_name = get_agent_name_from_path(q_table_path_agent1)
    agent2_name = get_agent_name_from_path(q_table_path_agent2)
    agent3_name = get_agent_name_from_path(q_table_path_agent3)
    agent4_name = get_agent_name_from_path(q_table_path_agent4)

    # Number of games to evaluate
    num_games = 10  # Adjust the number of evaluation games as needed

    # Evaluate the agents with their descriptive names
    evaluate(agent1, agent1_name, agent2, agent2_name, agent3, agent3_name, agent4, agent4_name, num_games)

    # Cleanup pygame if rendering was enabled
    if render_game:
        pygame.quit()

# Entry point for the script
if __name__ == "__main__":
    main()

