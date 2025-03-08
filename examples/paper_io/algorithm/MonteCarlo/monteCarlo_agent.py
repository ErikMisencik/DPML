import pickle
import numpy as np
import random
import heapq
from collections import deque, defaultdict
from examples.paper_io.algorithm.base_agent import BaseAgent

class MCAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1, 
                 replay_size=5000, batch_updates=5, batch_size=64, load_only=False):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size  
        self.q_table = defaultdict(float)  
        self.returns_sum = defaultdict(float)  
        self.returns_count = defaultdict(int)  
        self.returns_priority = defaultdict(float)  
        self.episode_history = deque()  
        self.replay_memory = deque(maxlen=replay_size)  
        self.load_only = load_only
        self.batch_updates = batch_updates  
        self.episode_count = 0  

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
        q_values -= np.max(q_values)  # Prevents overflow
        exp_q = np.exp(q_values / tau)
        probabilities = exp_q / np.sum(exp_q)

        return np.random.choice(num_actions, p=probabilities)

    def store_episode_step(self, state, action, reward):
        """Stores a step in the episode history."""
        self.episode_history.append((state, action, reward))

    def update(self):
        """Performs the Monte Carlo update after each episode."""
        if self.load_only or not self.episode_history:
            return  

        G = 0  
        visited_state_actions = set()
        self.episode_count += 1

        for state, action, reward in reversed(self.episode_history):
            G = reward + self.discount_factor * G
            state_action = (state, action)

            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)

                self.returns_sum[state_action] += G
                self.returns_count[state_action] += 1
                self.returns_priority[state_action] = abs(G)  

                W = 1.0 / (self.returns_count[state_action] + 1e-5)
                self.q_table[state_action] = (1 - W) * self.q_table[state_action] + W * G

        if len(self.replay_memory) >= self.batch_size:
            self.replay_memory.popleft()
        self.replay_memory.append(list(self.episode_history))  

        if self.episode_count % self.batch_updates == 0:
            self._apply_prioritized_updates()

        self.episode_history.clear()

    def _apply_prioritized_updates(self):
        """Safe batch updates using Prioritized Experience Replay."""
        if not self.returns_priority:
            return  

        batch = heapq.nlargest(self.batch_size, self.returns_priority.items(), key=lambda x: x[1])  

        for (state, action), _ in batch:
            if self.returns_count[(state, action)] == 0:
                continue  

            W = 1.0 / (self.returns_count[(state, action)] + 1e-5)
            self.q_table[(state, action)] = (1 - W) * self.q_table[(state, action)] + W * self.returns_sum[(state, action)]

        if len(self.returns_priority) > 500:
            self.returns_priority.pop(next(iter(self.returns_priority)))

    def decay_epsilon(self):
        """Gradually decreases ε for exploration-exploitation tradeoff."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the Q-table to a file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"Monte Carlo Q-table saved to {filepath}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load(self, filepath):
        """Loads the Q-table from a file."""
        try:
            with open(filepath, 'rb') as f:
                self.q_table.update(pickle.load(f))  
            print(f"Monte Carlo Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"Q-table file {filepath} not found. Starting fresh.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

    def _get_nearest_enemy_distance(self, grid, player_id, x_local, y_local):
        """Optimized nearest enemy distance calculation."""
        enemy_positions = np.argwhere((grid > 0) & (grid != player_id))
        if enemy_positions.size == 0:
            return self.env.grid_size * 2  
        return np.min(np.sum(np.abs(enemy_positions - np.array([x_local, y_local])), axis=1))

    def _get_nearest_enemy_direction(self, grid, player_id, x_local, y_local):
        """Optimized nearest enemy direction calculation."""
        enemy_positions = np.argwhere((grid > 0) & (grid != player_id))
        if enemy_positions.size == 0:
            return 'none'

        differences = enemy_positions - np.array([x_local, y_local])
        distances = np.sum(np.abs(differences), axis=1)
        dx, dy = differences[np.argmin(distances)]

        if abs(dx) > abs(dy):
            return 'down' if dx > 0 else 'up'
        return 'right' if dy > 0 else 'left'
