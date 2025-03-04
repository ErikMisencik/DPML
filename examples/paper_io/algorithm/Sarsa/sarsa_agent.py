import pickle
import numpy as np
import random
import heapq
from collections import defaultdict
from examples.paper_io.algorithm.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1,
                 n_step=3, replay_size=5000, batch_size=64, lambda_value=0.8, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_step = n_step  
        self.batch_size = batch_size
        self.lambda_value = lambda_value  
        self.q_table = defaultdict(float)  
        self.replay_memory = []  # Use list for heapq compatibility
        self.replay_size = replay_size  # Limit replay size
        self.n_step_buffer = []  
        self.e_trace = defaultdict(float)  
        self.load_only = load_only  

    def get_state(self, observation, player_idx):
        """Extracts a hashable state representation."""
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
        nearest_enemy_distance = min(self._get_nearest_enemy_distance(grid, player_id, x_local, y_local), 10)
        nearest_enemy_direction = self._get_nearest_enemy_direction(grid, player_id, x_local, y_local)

        return (position_status, trail_length, nearest_enemy_direction, nearest_enemy_distance)

    def get_action(self, observation, player_idx):
        """Boltzmann Exploration for action selection."""
        if not self.env.alive[player_idx]:
            return None

        state = self.get_state(observation, player_idx)
        num_actions = self.env.action_spaces[player_idx].n
        q_values = np.array([self.q_table[(state, a)] for a in range(num_actions)])

        tau = max(0.1, self.epsilon)  
        q_values = q_values - np.max(q_values)  
        exp_q = np.exp(q_values / tau)
        probabilities = exp_q / np.sum(exp_q)

        return np.random.choice(num_actions, p=probabilities)

    def update(self, state, action, reward, next_state, next_action, done, player_idx):
        """N-Step SARSA(λ) update with Prioritized Experience Replay and Eligibility Traces."""
        if self.load_only:
            return  

        self.n_step_buffer.append((state, action, reward, next_state, next_action))
        self.e_trace[(state, action)] += 1  

        if len(self.n_step_buffer) == self.n_step:
            G = sum(self.discount_factor ** i * r for i, (_, _, r, _, _) in enumerate(self.n_step_buffer))
            s, a, _, ns, na = self.n_step_buffer.pop(0)  

            # Trim replay buffer if exceeding limit
            if len(self.replay_memory) >= self.replay_size:
                heapq.heappop(self.replay_memory)  

            heapq.heappush(self.replay_memory, (-abs(G), (s, a, G, ns, na)))  

        if len(self.replay_memory) >= self.batch_size:
            self._update_from_replay(player_idx)

        self._td_lambda_update(state, action, reward, next_state, next_action, done, player_idx)

    def _td_lambda_update(self, state, action, reward, next_state, next_action, done, player_idx):
        """TD(λ) update using eligibility traces."""
        next_q = self.q_table[(next_state, next_action)] if (next_state and next_action and not done) else 0
        td_error = reward + self.discount_factor * next_q - self.q_table[(state, action)]

        for (s, a), trace_value in list(self.e_trace.items()):
            self.q_table[(s, a)] += self.learning_rate * td_error * trace_value
            self.e_trace[(s, a)] *= self.discount_factor * self.lambda_value  

        for key in list(self.e_trace.keys()):
            if self.e_trace[key] < 1e-5:
                del self.e_trace[key]

    def _update_from_replay(self, player_idx):
        """Efficient batch updates using Prioritized Experience Replay."""
        batch = []
        for _ in range(min(self.batch_size, len(self.replay_memory))):
            if self.replay_memory:
                batch.append(heapq.heappop(self.replay_memory)[1])  

        for s, a, r, ns, na in batch:
            current_q = self.q_table[(s, a)]
            next_q = self.q_table[(ns, na)] if ns else 0

            td_error = r + self.discount_factor * next_q - current_q
            self.q_table[(s, a)] += self.learning_rate * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"SARSA Q-table saved to {filepath}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                self.q_table.update(pickle.load(f))  
            print(f"SARSA Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"Q-table file {filepath} not found. Starting fresh.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

    def _get_nearest_enemy_distance(self, grid, player_id, x_local, y_local):
        enemy_positions = np.argwhere((grid > 0) & (grid != player_id))
        if enemy_positions.size == 0:
            return self.env.grid_size * 2  

        return np.min(np.sum(np.abs(enemy_positions - np.array([x_local, y_local])), axis=1))

    def _get_nearest_enemy_direction(self, grid, player_id, x_local, y_local):
        enemy_positions = np.argwhere((grid > 0) & (grid != player_id))
        if enemy_positions.size == 0:
            return 'none'

        differences = enemy_positions - np.array([x_local, y_local])
        distances = np.sum(np.abs(differences), axis=1)
        dx, dy = differences[np.argmin(distances)]

        if abs(dx) > abs(dy):
            return 'down' if dx > 0 else 'up'
        return 'right' if dy > 0 else 'left'
