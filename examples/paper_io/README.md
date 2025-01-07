# Features

## Grid-Based Environment
- The game is played on a grid of configurable size (default is 50x50).
- Each player controls a position on the grid and leaves a trail while moving.
- Players attempt to capture territory by returning to their own base after leaving a trail.

## Players
- The environment supports multiple players (default is 2 players).
- Each player has its own unique ID, starting position, and trail.
- Players can move in four directions: Up, Down, Left, and Right.

## Territory Capturing
- Players can claim territory by completing a trail and returning to their own base.
- The captured trail and enclosed areas are converted into permanent territory.
- Rewards are assigned based on the amount of territory captured.

## Eliminations
- Players can eliminate opponents by crossing their trails.
- When an opponent’s trail is cut off, the trail's owner is eliminated, and their territory and trail are removed from the grid.
- Rewards are assigned for successful eliminations and penalties for being eliminated.

## Reward System
- Players are rewarded for capturing territory (based on the size of the captured area).
- Players are also rewarded for eliminating opponents by cutting off their trails.
- Penalties are applied for being eliminated by another player.

## OpenCV-Based Visualization
- The environment can be rendered using OpenCV to visually display the game grid.
- Each player’s territory and trail are colored uniquely.
- The visualization is updated in real-time as the game progresses.

## Game Loop
- Each game step processes the actions of all players and updates the game state accordingly.
- Players move, claim territory, and can be eliminated based on their actions and interactions with other players.
- The game ends when only one player remains alive, or when a predefined condition is met.

## Installation

Create a conda environment

```` sh
conda create -n arena5 python=3.7 anaconda
conda activate arena5
pip install -e .
pip install stable_baselines tensorflow==1.14.0
conda install -c conda-forge mpi4py
````

## Use for evaluating

python eval.py

## Use for training

python training.py

## RL Policy - CheckList

- Random Policy DONE
- Greedy Policy (Simple improvement over random) DONE
- Q-Learning (Start with simple Q-tables and reward maximization) DONE
- SARSA (A more conservative version of Q-learning) DONE
- Monte Carlo Methods (For episodic learning and environments with long-term rewards) DONE
- Temporal Difference (TD) Learning (More dynamic and works well in continuous learning scenarios)

- Policy Iteration (For deterministic environments with well-defined policies)
- Actor-Critic (without neural networks) (When combining value functions with policy learning)


## Reward Config Strategies Learning

### =========================================================
### Config A: Aggressive Expansion Focus
### Emphasizes territory capture over safety; encourages building larger trails quickly.
### =========================================================
self.reward_config = {
    'self_elimination_penalty': -200,   # Slightly less punishing to allow more risk-taking
    'long_camping_penalty': -250,       # Still penalize camping, but not as harshly
    'trail_reward': 50,                 # More reward for extending trails
    'max_trail_reward': 300,            # Higher cap, encouraging big loops
    'territory_capture_reward_per_cell': 50,  # Significantly increased to push expansion
    'max_trail_length': 15,
    'long_trail_penalty': -10,          # Reduced penalty, bigger loops are less risky
    'opponent_elimination_reward': 150, # Somewhat rewarding, but not the main focus
    'opponent_elimination_penalty': -50,
    'enemy_territory_capture_reward_per_cell': 40,  # Capturing enemy territory is lucrative
    'territory_loss_penalty_per_cell': -50,
    'elimination_reward_modifier': 0.70,
}

### =========================================================
### Config B: Kill-or-Be-Killed Aggression
### Greatly rewards eliminating opponents, encourages direct confrontation and risk.
### =========================================================
self.reward_config = {
    'self_elimination_penalty': -300,   # Still quite punishing for suicidal moves
    'long_camping_penalty': -200,       # Medium penalty for camping
    'trail_reward': 30,                 # Basic reward for building trails
    'max_trail_reward': 200,
    'territory_capture_reward_per_cell': 30,
    'max_trail_length': 15,
    'long_trail_penalty': -20,          # Standard penalty if trail is too long
    'opponent_elimination_reward': 400, # Very high reward to incentivize kills
    'opponent_elimination_penalty': -100, # Victim gets penalized more
    'enemy_territory_capture_reward_per_cell': 25,  # Slightly lowered to keep focus on kills
    'territory_loss_penalty_per_cell': -60,
    'elimination_reward_modifier': 0.70,
}

### =========================================================
### Config C: Defensive / Safe Strategies
### Focuses on holding territory, penalizes self-elimination severely, encourages minimal risk.
### =========================================================
self.reward_config = {
    'self_elimination_penalty': -400,    # Extra harsh to discourage risky behavior
    'long_camping_penalty': -400,        # Also punishes staying still too long
    'trail_reward': 20,                  # Lower trail reward
    'max_trail_reward': 150,             # Cap smaller to reduce risky big loops
    'territory_capture_reward_per_cell': 30,
    'max_trail_length': 10,              # Encourage short, safe expansions
    'long_trail_penalty': -40,           # Bigger penalty for overly long trails
    'opponent_elimination_reward': 150,  # Some reward, but not huge
    'opponent_elimination_penalty': -50,
    'enemy_territory_capture_reward_per_cell': 25,  # Moderately beneficial
    'territory_loss_penalty_per_cell': -80,         # Large penalty for losing territory
    'elimination_reward_modifier': 0.70,
}


### =========================================================
### Config D: Balanced or “Point Control” Approach
### Tries to keep every aspect moderate, no single dimension dominates the strategy.
### =========================================================
self.reward_config = {
    'self_elimination_penalty': -300,    # Medium-level punishment
    'long_camping_penalty': -300,        # Punish inactivity, but not extremely
    'trail_reward': 40,                  # Decent reward for building trails
    'max_trail_reward': 200,
    'territory_capture_reward_per_cell': 30,
    'max_trail_length': 15,
    'long_trail_penalty': -20,           # Minor penalty for excessive loops
    'opponent_elimination_reward': 200,  # Reward is relevant, but not huge
    'opponent_elimination_penalty': -50,
    'enemy_territory_capture_reward_per_cell': 30,  # Balanced approach
    'territory_loss_penalty_per_cell': -50,
    'elimination_reward_modifier': 0.70,
}
