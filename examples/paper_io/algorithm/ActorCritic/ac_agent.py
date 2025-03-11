import pickle
import numpy as np
import random
from examples.paper_io.algorithm.base_agent import BaseAgent  # Assuming BaseAgent is in base_agent.py

class ACAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.003, discount_factor=0.99,
                 lambda_value=0.8, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_value = lambda_value  # λ parameter for eligibility traces
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.value_table = {}  # Critic: State-Value Function
        self.policy_table = {}  # Actor: Policy
        self.e_trace_value = {}  # Eligibility traces for value function
        self.e_trace_policy = {}  # Eligibility traces for policy
        self.load_only = load_only  # To lock learning if only loading

    def get_state(self, observation, player_idx):
        """Extracts a compact and hashable state representation."""
        grid = observation[player_idx]
        player = self.env.players[player_idx]
        player_id = player['id']
        x_local, y_local = player['position']

        cell_value = grid[x_local, y_local]
        position_status = (
            'on_territory' if cell_value == player_id else
            'on_trail' if cell_value == -player_id else
            'in_neutral'
        )
        trail_length = min(len(player['trail']), 5)

        enemy_distances, enemy_directions = self._get_enemy_info(grid, player_id, x_local, y_local)
        nearest_enemy_distance = min(enemy_distances, default=10)
        nearest_enemy_direction = enemy_directions[enemy_distances.index(nearest_enemy_distance)] if enemy_distances else 'none'

        return (position_status, trail_length, nearest_enemy_direction, nearest_enemy_distance)

    def get_action(self, observation, player_idx):
        """Chooses an action using a learned policy (Actor)."""
        if not self.env.alive[player_idx]:
            return None

        state = self.get_state(observation, player_idx)
        num_actions = self.env.action_spaces[player_idx].n

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, num_actions - 1)  # Explore

        action_probs = self.policy_table.get(state, np.ones(num_actions) / num_actions)  # Default uniform probabilities
        return np.random.choice(num_actions, p=action_probs)

    def update(self, state, action, reward, next_state, done, player_idx):
        """Performs Actor-Critic TD(λ) update with eligibility traces."""
        if self.load_only:
            return  # Skip updates if only loading

        # **Critic Update (Value Function)**
        current_v = self.value_table.get(state, 0)
        next_v = 0 if done else self.value_table.get(next_state, 0)
        advantage = reward + self.discount_factor * next_v - current_v

        self.e_trace_value[state] = self.e_trace_value.get(state, 0) + 1

        for s in list(self.e_trace_value.keys()):
            self.value_table[s] = self.value_table.get(s, 0) + self.learning_rate * advantage * self.e_trace_value[s]
            self.e_trace_value[s] *= self.discount_factor * self.lambda_value

            if self.e_trace_value[s] < 1e-5:
                del self.e_trace_value[s]

        # **Actor Update (Policy Improvement)**
        num_actions = self.env.action_spaces[player_idx].n
        if state not in self.policy_table:
            self.policy_table[state] = np.ones(num_actions) / num_actions  # Initialize with uniform probabilities

        self.e_trace_policy[(state, action)] = self.e_trace_policy.get((state, action), 0) + 1

        for (s, a), trace_value in list(self.e_trace_policy.items()):
            action_probs = self.policy_table[s]
            action_probs[a] += self.learning_rate * advantage * trace_value
            action_probs = np.maximum(action_probs, 1e-5)  # Prevent zero probabilities
            action_probs /= np.sum(action_probs)  # Normalize to a probability distribution
            self.policy_table[s] = action_probs

            self.e_trace_policy[(s, a)] *= self.discount_factor * self.lambda_value
            if self.e_trace_policy[(s, a)] < 1e-5:
                del self.e_trace_policy[(s, a)]

        # Reset traces if episode ends
        if done:
            self.e_trace_value.clear()
            self.e_trace_policy.clear()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the Actor-Critic tables to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({"value_table": self.value_table, "policy_table": self.policy_table}, f)
        print(f"Actor-Critic tables saved to {filepath}")

    def load(self, filepath):
        """Loads the Actor-Critic tables from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.value_table = data["value_table"]
            self.policy_table = data["policy_table"]
        print(f"Actor-Critic tables loaded from {filepath}")

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
