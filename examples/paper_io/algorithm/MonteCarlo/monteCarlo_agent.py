import pickle
import numpy as np
import random
from examples.paper_io.algorithm.base_agent import BaseAgent

class MCAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.003, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.returns_sum = {}  # Stores the sum of returns for each state-action pair
        self.returns_count = {}  # Counts the number of returns for each state-action pair
        self.episode_history = []  # Stores state, action, reward for the current episode
        self.load_only = load_only  # Skip updates if True

    def get_state(self, observation, player_idx):
        """Extracts a compact and hashable state representation."""
        grid = observation[player_idx]
        player = self.env.players[player_idx]
        player_id = player['id']

        # Determine position and observability
        x_local, y_local = player['position'] if grid.shape == (self.env.grid_size, self.env.grid_size) else (
            grid.shape[0] // 2, grid.shape[1] // 2)

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
        return random.choice([a for a, q in enumerate(q_values) if q == max_q])

    def store_episode_step(self, state, action, reward):
        """Stores the current step in the episode history."""
        self.episode_history.append((state, action, reward))

    def update(self):
        """Performs the Monte Carlo update after the episode ends."""
        if self.load_only:
            return  # Skip updates if only loading

        # Calculate returns for each state-action pair
        G = 0
        for state, action, reward in reversed(self.episode_history):
            G = reward + self.discount_factor * G  # Discounted sum of rewards
            state_action = (state, action)

            # Only update if it's the first occurrence of the state-action pair in this episode
            if state_action not in self.episode_history[:self.episode_history.index((state, action, reward))]:
                # Update returns_sum and returns_count
                self.returns_sum[state_action] = self.returns_sum.get(state_action, 0) + G
                self.returns_count[state_action] = self.returns_count.get(state_action, 0) + 1

                # Update Q-value
                self.q_table[state_action] = self.returns_sum[state_action] / self.returns_count[state_action]

        # Clear the episode history
        self.episode_history = []

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Monte Carlo Q-table saved to {filepath}")

    def load(self, filepath):
        """Loads the Q-table from a file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Monte Carlo Q-table loaded from {filepath}")

    def _get_enemy_info(self, grid, player_id, x_local, y_local):
        """Returns distances and directions to the nearest enemies."""
        enemy_mask = (grid > 0) & (grid != player_id)
        enemy_positions = np.argwhere(enemy_mask)
        
        if not enemy_positions.size:
            return [], []  # No enemies

        differences = enemy_positions - np.array([x_local, y_local])
        distances = np.abs(differences).sum(axis=1)
        directions = ['down' if dx > 0 else 'up' if dx < 0 else 'right' if dy > 0 else 'left'
                      for dx, dy in differences]

        return distances.tolist(), directions
