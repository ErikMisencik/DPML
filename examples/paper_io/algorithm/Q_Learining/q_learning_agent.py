import pickle
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.005, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.996, min_epsilon=0.1):
        self.env = env
        self.alpha = learning_rate        # Learning rate
        self.gamma = discount_factor      # Discount factor
        self.epsilon = epsilon            # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}                 # Q-table initialized as an empty dictionary

    def get_state(self, observation, player_idx):
        # Extract features to create a state representation
        grid = observation[player_idx]
        # print(f"Debug: Type of grid: {type(grid)}, shape: {grid.shape}")
        player = self.env.players[player_idx]
        x, y = player['position']

        # Check if the player is on their territory or trail
        cell_value = grid[x, y]
        if cell_value == player['id']:
            position_status = 'on_territory'
        elif cell_value == -player['id']:
            position_status = 'on_trail'
        else:
            position_status = 'in_neutral'

        # Trail length
        trail_length = len(player['trail'])

        # Find the nearest enemy player
        nearest_enemy_distance = self._get_nearest_enemy_distance(player_idx)
        nearest_enemy_direction = self._get_nearest_enemy_direction(player_idx)

        # Create the state as a tuple of discrete features
        state = (
            position_status,
            min(trail_length, 5),  # Cap trail length at 5 for discretization
            nearest_enemy_direction,
            min(nearest_enemy_distance, 10)  # Cap distance at 10
        )
        return state

    def get_actions(self, observation):
        actions = []
        for i in range(self.env.num_players):
            if not self.env.alive[i]:
                actions.append(None)  # No action for dead players
                continue

            state = self.get_state(observation, i)

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < self.epsilon:
                # Explore: choose a random action
                action = self.env.action_spaces[i].sample()
            else:
                # Exploit: choose the best known action
                num_actions = self.env.action_spaces[i].n
                q_values = [self.q_table.get((state, a), 0) for a in range(num_actions)]
                max_q = max(q_values)
                max_actions = [a for a, q in enumerate(q_values) if q == max_q]
                action = random.choice(max_actions)  # Break ties randomly

            actions.append(action)
        return actions

    def update_q_values(self, state, action, reward, next_state, done, player_idx):
        current_q = self.q_table.get((state, action), 0)

        if done:
            max_future_q = 0
        else:
            num_actions = self.env.action_spaces[player_idx].n
            # Get the max Q value for the next state
            max_future_q = max([self.q_table.get((next_state, a), 0) for a in range(num_actions)])

        # Q-Learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        # Decay epsilon after each episode
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def _get_nearest_enemy_distance(self, player_idx):
        player = self.env.players[player_idx]
        x1, y1 = player['position']
        min_distance = self.env.grid_size * 2  # Initialize with a large number

        for i, other_player in enumerate(self.env.players):
            if i != player_idx and self.env.alive[i]:
                x2, y2 = other_player['position']
                distance = abs(x1 - x2) + abs(y1 - y2)
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def _get_nearest_enemy_direction(self, player_idx):
        player = self.env.players[player_idx]
        x1, y1 = player['position']
        min_distance = self.env.grid_size * 2
        direction = 'none'

        for i, other_player in enumerate(self.env.players):
            if i != player_idx and self.env.alive[i]:
                x2, y2 = other_player['position']
                distance = abs(x1 - x2) + abs(y1 - y2)
                if distance < min_distance:
                    min_distance = distance
                    # Determine direction
                    if x2 > x1:
                        direction = 'down'
                    elif x2 < x1:
                        direction = 'up'
                    elif y2 > y1:
                        direction = 'right'
                    elif y2 < y1:
                        direction = 'left'
        return direction
    
    def save_q_table(self, filepath):
        # Save the Q-table to a file using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        # Load the Q-table from a file using pickle
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filepath}")