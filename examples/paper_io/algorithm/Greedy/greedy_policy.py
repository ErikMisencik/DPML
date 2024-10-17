import numpy as np
import copy

class GreedyPolicy:
    def __init__(self, env):
        self.env = env

    def get_actions(self, observation):
        actions = []
        for i in range(self.env.num_players):
            if not self.env.alive[i]:
                actions.append(None)  # Player is dead, skip action
                continue

            # Evaluate all possible actions
            best_action = 0
            best_reward = -float('inf')

            for action in range(4):  # Actions: 0-Up, 1-Down, 2-Left, 3-Right
                simulated_env = self._simulate_action(i, action)
                reward = self._evaluate_simulation(simulated_env, i)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

            actions.append(best_action)
        return actions

    def _simulate_action(self, player_idx, action):
        # Create a deep copy of the environment
        simulated_env = copy.deepcopy(self.env)
        player = simulated_env.players[player_idx]
        x, y = player['position']

        # Simulate the new position based on the action
        new_x, new_y = x, y
        if action == 0 and x > 0:  # Up
            new_x -= 1
        elif action == 1 and x < self.env.grid_size - 1:  # Down
            new_x += 1
        elif action == 2 and y > 0:  # Left
            new_y -= 1
        elif action == 3 and y < self.env.grid_size - 1:  # Right
            new_y += 1

        new_position = (new_x, new_y)
        cell_value = simulated_env.grid[new_x, new_y]

        # Update the simulated environment based on the action
        player['position'] = new_position
        if cell_value == 0:
            # Leaving a trail
            simulated_env.grid[new_x, new_y] = -player['id']
            player['trail'].append(new_position)
        elif cell_value == player['id'] and player['trail']:
            # Convert trail to territory
            simulated_env.convert_trail_to_territory(player['id'])

        return simulated_env

    def _evaluate_simulation(self, simulated_env, player_idx):
        # Evaluate based on the area of the player's territory
        player_id = simulated_env.players[player_idx]['id']
        territory_area = np.sum(simulated_env.grid == player_id)
        return territory_area
