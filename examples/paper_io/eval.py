import os
import random
import sys
from time import sleep
import pygame # type: ignore

from examples.paper_io.Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Greedy.greedy_agent import GreedyAgent
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLAgent
from examples.paper_io.algorithm.Random.random_agent import RandomAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors

# Set up the rendering flag for the environment
render_game = True  # Set to True if you want to render the game during evaluation

# Initialize the environment with rendering enabled and max_steps
steps_per_episode = 500  # Use the same max_steps as in training
env = PaperIoEnv(render=render_game, max_steps=steps_per_episode)

# -----------------------------------------------------------------------------
# NEW: Helper function to determine agent names from file paths
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------

# Function to load a trained Q-learning model (Q-table)
def load_q_learning_model(q_table_path):
    agent = QLAgent(env, learning_rate=0, epsilon=0.0, load_only=True)  # Read-only Q-table
    agent.load(q_table_path)  # Load the Q-table from the provided path
    return agent

# Function to evaluate agents in the environment
def evaluate(agent1, agent1_name, agent2, agent2_name, num_games=10):
    # Assign random colors to the agents
    color_info = assign_agent_colors(env.num_players)
    agent_colors = [info[0] for info in color_info]  # Extract only RGB values for rendering
    agent_color_names = [info[1] for info in color_info]  # Extract color names for logging

    # Print agent colors at the start with both name and RGB tuple
    print(f"\nEvaluation Setup:")
    print(f"{agent1_name} is assigned color {agent_color_names[0]}")
    print(f"{agent2_name} is assigned color {agent_color_names[1]}\n")

    agent_wins = [0 for _ in range(env.num_players)]  # Track wins per agent

    for game_num in range(num_games):
        obs = env.reset()  # Reset the environment for a new game
        done = False

        # Reset cumulative rewards for this game
        agent_game_rewards = [0 for _ in range(env.num_players)]

        while not done:
            # Render the game, passing player colors
            if render_game:
                env.render(agent_colors)  # Pass the agent colors to the render method

            # Get actions for both agents separately
            action_agent1 = agent1.get_action(obs, 0)  # Agent 1 is at index 0
            action_agent2 = agent2.get_action(obs, 1)  # Agent 2 is at index 1

            # Combine actions from both agents into one list
            actions = [action_agent1, action_agent2]

            # Take a step in the environment
            obs, rewards, done, info = env.step(actions)

            # Accumulate rewards for each agent
            for i in range(env.num_players):
                agent_game_rewards[i] += rewards[i]

            if render_game:   # Slow down the game for rendering
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
            agent_name = agent1_name if i == 0 else agent2_name if i == 1 else f"Agent {i}"
            print(f"{agent_name} cumulative reward: {cumulative_rewards[i]:.2f}")
        if winners:
            winner_names = [agent1_name if i == 0 else agent2_name if i == 1 else f"Agent {i}" for i in winners]
            print(f"Winner(s): {', '.join(winner_names)}\n")
        else:
            print("No winner this game.\n")

    # Final evaluation results
    print(f"\nEvaluation Results (over {num_games} games):")
    for i in range(env.num_players):
        agent_name = agent1_name if i == 0 else agent2_name if i == 1 else f"Agent {i}"
        print(f"{agent_name} (Color: {agent_color_names[i]}) wins: {agent_wins[i]}")

# Main evaluation function
def main():
    print("Starting evaluation...")

    base_models_path = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/"

    saved_model_path1 = "New_M_2_Q-Learning_SARSA_6/trained_model/qlagent_ag_0_end.pkl"
    saved_model_path2 = "New_M_2_Q-Learning_SARSA_6/trained_model/sarsaagent_ag_1_end.pkl"

    q_table_path_agent1 = os.path.join(base_models_path, saved_model_path1)
    q_table_path_agent2 = os.path.join(base_models_path, saved_model_path2)

    # Load agents
    agent1 = load_q_learning_model(q_table_path_agent1)
    agent2 = load_q_learning_model(q_table_path_agent2)


    agent1_name = get_agent_name_from_path(q_table_path_agent1)
    agent2_name = get_agent_name_from_path(q_table_path_agent2)

    # Number of games to evaluate
    num_games = 10  # Adjust the number of evaluation games as needed

    # Evaluate the agents with their descriptive names
    evaluate(agent1, agent1_name, agent2, agent2_name, num_games)

    # Cleanup pygame if rendering was enabled
    if render_game:
        pygame.quit()

# Entry point for the script
if __name__ == "__main__":
    main()
