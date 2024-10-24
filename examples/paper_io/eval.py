import os
import random
import sys
from time import sleep
from examples.paper_io.algorithm.Random.random_agent import RandomAgent
from examples.paper_io.utils import render
import pygame
from Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLearningAgent
from examples.paper_io.utils.agent_colors import assign_agent_colors

# Set up the rendering flag for the environment
render_game = True  # We want to render the environment for evaluation

# Initialize the environment with rendering enabled
env = PaperIoEnv(render=render_game)

# Function to load a trained Q-learning model (Q-table)
def load_q_learning_model(q_table_path):
    agent = QLearningAgent(env)
    agent.load_q_table(q_table_path)  # Load the Q-table from the provided path
    return agent

# Function to evaluate agents in the environment
def evaluate(agent1, agent1_name, agent2, agent2_name, num_games=10):
    # Assign random colors to the agents
    color_info = assign_agent_colors(env.num_players)
    agent_colors = [info[0] for info in color_info]  # Extract only RGB values for rendering
    agent_color_names = [info[1] for info in color_info]  # Extract color names for logging

    # Print agent colors at the start with both name and RGB tuple
    print(f"\nEvaluation Setup:")
    print(f"{agent1_name} is assigned color {agent_color_names[0]} ")
    print(f"{agent2_name} is assigned color {agent_color_names[1]} \n")

    agent1_wins = 0
    agent2_wins = 0

    for game_num in range(num_games):
        obs = env.reset()  # Reset the environment for a new game
        done = False
        agent1_alive = False
        agent2_alive = False

        while not done:
            # Render the game, passing player colors
            if render_game:
                env.render(agent_colors)  # Pass the agent colors to the render method

            # Get actions from both agents based on the current observation
            actions_agent1 = agent1.get_actions(obs)
            actions_agent2 = agent2.get_actions(obs)

            # Combine actions into one list to step through the environment
            actions = [actions_agent1[i] if i == 0 else actions_agent2[i] for i in range(env.num_players)]

            # Take a step in the environment
            obs, rewards, done, _ = env.step(actions)
            sleep(0.2)

        # Determine which agent won
        agent1_alive = env.alive[0]
        agent2_alive = env.alive[1]

        # Print the result of the current game
        print(f"Game {game_num + 1}: {agent1_name} {'won' if agent1_alive else 'lost'}, {agent2_name} {'won' if agent2_alive else 'lost'}")

        # Update win counts
        if agent1_alive:
            agent1_wins += 1
        elif agent2_alive:
            agent2_wins += 1

    # Final evaluation results
    print(f"\nEvaluation Results (over {num_games} games):")
    print(f"{agent1_name} (Color: {agent_color_names[0]}) wins: {agent1_wins}")
    print(f"{agent2_name} (Color: {agent_color_names[1]}) wins: {agent2_wins}")

# Main evaluation function (non-interactive)
def main():
    print("Starting evaluation...")

    # Use absolute path for Agent 1's Q-learning model
    q_table_path_agent1 = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/q_learning_3/trained_model/q_table.pkl"

    agent1 = load_q_learning_model(q_table_path_agent1)
    agent1_name = "Q-Learning Agent"

    # Agent 2: Random agent
    agent2 = RandomAgent(env)
    agent2_name = "Random Agent"

    # Number of games to evaluate
    num_games = 1000  # You can change the number of evaluation games

    # Evaluate the agents with their descriptive names
    evaluate(agent1, agent1_name, agent2, agent2_name, num_games)

    # Cleanup pygame if rendering was enabled
    if render_game:
        pygame.quit()


# Entry point for the script
if __name__ == "__main__":
    main()
