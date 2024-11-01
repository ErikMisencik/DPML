import pickle
import numpy as np
import random
from examples.paper_io.algorithm.base_agent import BaseAgent
 # Assuming BaseAgent is saved in base_agent.py

class QLAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.003, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.1):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.td_errors = []

    def get_state(self, observation, player_idx):
        grid = observation[player_idx]
        player = self.env.players[player_idx]
        x, y = player['position']
        cell_value = grid[x, y]
        position_status = 'on_territory' if cell_value == player['id'] else 'on_trail' if cell_value == -player['id'] else 'in_neutral'
        trail_length = len(player['trail'])
        nearest_enemy_distance = self._get_nearest_enemy_distance(player_idx)
        nearest_enemy_direction = self._get_nearest_enemy_direction(player_idx)
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

    def _get_nearest_enemy_distance(self, player_idx):
        player = self.env.players[player_idx]
        x1, y1 = player['position']
        min_distance = self.env.grid_size * 2
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
                    if x2 > x1:
                        direction = 'down'
                    elif x2 < x1:
                        direction = 'up'
                    elif y2 > y1:
                        direction = 'right'
                    elif y2 < y1:
                        direction = 'left'
        return direction
