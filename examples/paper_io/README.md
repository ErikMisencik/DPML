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