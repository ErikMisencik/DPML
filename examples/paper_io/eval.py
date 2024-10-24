import os
import sys
from examples.paper_io.algorithm.Random.random_agent import RandomAgent
import pygame
from Paper_io_develop import PaperIoEnv
from examples.paper_io.algorithm.Q_Learining.q_learning_agent import QLearningAgent
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
    # Print agent names
    print(f"Evaluating {agent1_name} against {agent2_name}")

    agent1_wins = 0
    agent2_wins = 0

    for game_num in range(num_games):
        obs = env.reset()  # Reset the environment for a new game
        done = False
        agent1_alive = False
        agent2_alive = False

        while not done:
            # Render the game if enabled
            if render_game:
                env.render()

            # Get actions from both agents based on the current observation
            actions_agent1 = agent1.get_actions(obs)  # Actions from agent 1 (could be Q-learning)
            actions_agent2 = agent2.get_actions(obs)  # Actions from agent 2 (could be Random/Greedy)

            # Combine actions into one list to step through the environment
            actions = [actions_agent1[i] if i == 0 else actions_agent2[i] for i in range(env.num_players)]

            # Take a step in the environment
            obs, rewards, done, _ = env.step(actions)

        # Determine which agent won
        agent1_alive = env.alive[0]
        agent2_alive = env.alive[1]

        if agent1_alive:
            agent1_wins += 1
        elif agent2_alive:
            agent2_wins += 1

        print(f"Game {game_num + 1}: {agent1_name} {'won' if agent1_alive else 'lost'}, {agent2_name} {'won' if agent2_alive else 'lost'}")

    # Final evaluation results
    print(f"Evaluation Results (over {num_games} games):")
    print(f"{agent1_name} wins: {agent1_wins}")
    print(f"{agent2_name} wins: {agent2_wins}")

# Main evaluation function (non-interactive)
def main():
    print("Starting evaluation...")

    # Use absolute path for Agent 1's Q-learning model
    q_table_path_agent1 = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/models/q_learning_1/trained_model/q_table.pkl"
    q_table_path_agent2 = "C:/Users/Erik/TUKE/Diplomovka/paper_io/ai-arena/examples/paper_io/archive_models/q_learning_19/trained_model/q_table.pkl"
   
    agent1 = load_q_learning_model(q_table_path_agent1)
    agent1_name = "Q-Learning Agent"

    agent2 = load_q_learning_model(q_table_path_agent1)
    agent2_name = "Q-Learning Agent 2"

    # # Agent 2: Random agent
    # agent2 = RandomAgent(env)
    # agent2_name = "Random Agent"

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
