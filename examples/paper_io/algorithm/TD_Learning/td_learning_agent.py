import pickle
import numpy as np
import random
from examples.paper_io.algorithm.base_agent import BaseAgent  # Assuming BaseAgent is in base_agent.py

class TDAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.003, discount_factor=0.99,
                 lambda_value=0.8, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_value = lambda_value  # λ parameter for eligibility traces
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.e_trace = {}  # Eligibility traces
        self.td_errors = []
        self.load_only = load_only  # To lock Q-table if only loading

    def get_state(self, observation, player_idx):
        """Extracts a compact and hashable state representation."""
        grid = observation[player_idx]
        player = self.env.players[player_idx]
        player_id = player['id']

        # Determine position and observability
        if grid.shape == (self.env.grid_size, self.env.grid_size):
            x_local, y_local = player['position']
        else:
            x_local, y_local = grid.shape[0] // 2, grid.shape[1] // 2

        cell_value = grid[x_local, y_local]
        position_status = (
            'on_territory' if cell_value == player_id else
            'on_trail' if cell_value == -player_id else
            'in_neutral'
        )
        trail_length = min(len(player['trail']), 5)

        # Compute nearest enemy metrics
        enemy_distances, enemy_directions = self._get_enemy_info(grid, player_id, x_local, y_local)
        nearest_enemy_distance = min(enemy_distances, default=10)
        nearest_enemy_direction = enemy_directions[enemy_distances.index(nearest_enemy_distance)] if enemy_distances else 'none'

        # Return compact state
        return (position_status, trail_length, nearest_enemy_direction, nearest_enemy_distance)

    def get_action(self, observation, player_idx):
        """Chooses an action based on the epsilon-greedy policy."""
        if not self.env.alive[player_idx]:
            return None

        state = self.get_state(observation, player_idx)
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_spaces[player_idx].sample()  # Explore

        # Exploit: Choose action with max Q-value
        num_actions = self.env.action_spaces[player_idx].n
        q_values = [self.q_table.get((state, a), 0) for a in range(num_actions)]
        max_q = max(q_values)
        # If multiple actions have the same Q-value, pick one at random
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done, player_idx):
        """Performs the TD(λ) update with eligibility traces."""
        if self.load_only:
            return  # Skip updates if only loading

        # Retrieve current Q-value
        current_q = self.q_table.get((state, action), 0)

        # Choose the next action greedily (for off-policy TD(λ))
        num_actions = self.env.action_spaces[player_idx].n
        next_q_values = [self.q_table.get((next_state, a), 0) for a in range(num_actions)]
        max_next_q = max(next_q_values) if not done else 0

        # Compute TD error
        td_error = reward + self.discount_factor * max_next_q - current_q
        self.td_errors.append(abs(td_error))

        # Update eligibility trace for the current state-action pair
        self.e_trace[(state, action)] = self.e_trace.get((state, action), 0) + 1

        # Update Q-values for all state-action pairs in the trace
        for (s, a), trace_value in list(self.e_trace.items()):
            # Update Q-value
            self.q_table[(s, a)] = self.q_table.get((s, a), 0) + self.learning_rate * td_error * trace_value
            # Decay eligibility trace
            self.e_trace[(s, a)] *= self.discount_factor * self.lambda_value
            # Remove trace if it's too small
            if self.e_trace[(s, a)] < 1e-5:
                del self.e_trace[(s, a)]

        # If episode is done, reset eligibility traces
        if done:
            self.e_trace = {}

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"TD(λ) Q-table saved to {filepath}")

    def load(self, filepath):
        """Loads the Q-table from a file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"TD(λ) Q-table loaded from {filepath}")

    def _get_enemy_info(self, grid, player_id, x_local, y_local):
        """Returns distances and directions to the nearest enemies."""
        enemy_mask = (grid > 0) & (grid != player_id)
        enemy_positions = np.argwhere(enemy_mask)

        if not enemy_positions.size:
            return [], []  # No enemies

        differences = enemy_positions - np.array([x_local, y_local])
        distances = np.abs(differences).sum(axis=1)
        directions = []
        for dx, dy in differences:
            if abs(dx) > abs(dy):
                direction = 'down' if dx > 0 else 'up'
            elif abs(dy) > 0:
                direction = 'right' if dy > 0 else 'left'
            else:
                direction = 'none'  # Enemy is at the same position
            directions.append(direction)

        return distances.tolist(), directions
