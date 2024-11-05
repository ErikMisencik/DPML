import pickle
import numpy as np
import random
from examples.paper_io.algorithm.base_agent import BaseAgent
 # Assuming BaseAgent is saved in base_agent.py

class QLAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.003, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.td_errors = []
        self.load_only = load_only  # New attribute to lock Q-table

    def get_state(self, observation, player_idx):
        grid = observation[player_idx]
        player = self.env.players[player_idx]
        player_id = player['id']
        
        # Determine if the observation is partial or full
        if grid.shape[0] == self.env.grid_size and grid.shape[1] == self.env.grid_size:
            # Full Observability
            x_local, y_local = player['position']
        else:
            # Partial Observability
            x_local = grid.shape[0] // 2
            y_local = grid.shape[1] // 2
        
        cell_value = grid[x_local, y_local]
        position_status = (
            'on_territory' if cell_value == player_id else
            'on_trail' if cell_value == -player_id else
            'in_neutral'
        )
        trail_length = len(player['trail'])
        nearest_enemy_distance = self._get_nearest_enemy_distance(
            grid, player_id, x_local, y_local
        )
        nearest_enemy_direction = self._get_nearest_enemy_direction(
            grid, player_id, x_local, y_local
        )
        state = (
            position_status,
            min(trail_length, 5),
            nearest_enemy_direction,
            min(nearest_enemy_distance, 10)
        )
        return state

    def get_action(self, observation, player_idx):
        if not self.env.alive[player_idx]:
            return None

        state = self.get_state(observation, player_idx)
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_spaces[player_idx].sample()  # Explore
        else:
            num_actions = self.env.action_spaces[player_idx].n
            q_values = [self.q_table.get((state, a), 0) for a in range(num_actions)]
            max_q = max(q_values)
            max_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = random.choice(max_actions)  # Exploit

        return action

    def update(self, state, action, reward, next_state, done, player_idx):
        if self.load_only:
            return  # Do nothing if in load_only mode
        state = tuple(state) if isinstance(state, list) else state
        next_state = tuple(next_state) if isinstance(next_state, list) else next_state
        action = tuple(action) if isinstance(action, list) else action
        current_q = self.q_table.get((state, action), 0)
        max_future_q = 0 if done else max([self.q_table.get((next_state, a), 0) for a in range(self.env.action_spaces[player_idx].n)])
        td_error = reward + self.discount_factor * max_future_q - current_q
        self.td_errors.append(abs(td_error))
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filepath}")

    def _get_nearest_enemy_distance(self, grid, player_id, x_local, y_local):
         # Create a mask for enemy positions
        enemy_mask = (grid > 0) & (grid != player_id)
        enemy_positions = np.argwhere(enemy_mask)
        if enemy_positions.size == 0:
            return grid.shape[0] * 2  # No enemies detected; return max distance
        else:
            # Calculate Manhattan distances to all enemies
            distances = np.abs(enemy_positions - np.array([x_local, y_local])).sum(axis=1)
            min_distance = distances.min()
            return min_distance

    def _get_nearest_enemy_direction(self, grid, player_id, x_local, y_local):
        # Create a mask for enemy positions
        enemy_mask = (grid > 0) & (grid != player_id)
        enemy_positions = np.argwhere(enemy_mask)
        if enemy_positions.size == 0:
            return 'none'  # No enemies detected
        else:
            # Calculate Manhattan distances to all enemies
            differences = enemy_positions - np.array([x_local, y_local])
            distances = np.abs(differences).sum(axis=1)
            min_idx = distances.argmin()
            dx, dy = differences[min_idx]

            if abs(dx) > abs(dy):
                direction = 'down' if dx > 0 else 'up'
            elif abs(dy) > 0:
                direction = 'right' if dy > 0 else 'left'
            else:
                direction = 'none'  # Enemy is at the same position
            return direction
