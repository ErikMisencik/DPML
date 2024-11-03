import numpy as np

class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation, index=None):
        actions = []
        for i in range(self.env.num_players):
            if not self.env.alive[i]:
                actions.append(None)  # Player is dead, skip action
                continue

            best_action = 0
            best_reward = -float('inf')

            # Iterate over possible actions: 0 - Left, 1 - Right, 2 - Straight
            for action in range(3):
                # Store original state for quick reversal
                original_position = self.env.players[i]['position']
                original_direction = self.env.directions[i]
                original_trail_length = len(self.env.players[i]['trail'])

                # Apply the action and evaluate reward
                reward = self._simulate_and_evaluate(i, action)

                # Compare with the best reward
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

                # Revert to original state
                self.env.players[i]['position'] = original_position
                self.env.directions[i] = original_direction
                self.env.players[i]['trail'] = self.env.players[i]['trail'][:original_trail_length]

            actions.append(best_action)
        return actions

    def _simulate_and_evaluate(self, player_idx, action):
        """
        Simulates the given action and returns the computed reward
        """
        # Turn the direction if needed
        if action == 0:  # Left
            self.env.directions[player_idx] = self.env._turn_left(self.env.directions[player_idx])
        elif action == 1:  # Right
            self.env.directions[player_idx] = self.env._turn_right(self.env.directions[player_idx])

        # Move the player in the current direction
        player = self.env.players[player_idx]
        dx, dy = self.env.directions[player_idx]
        new_x, new_y = player['position'][0] + dx, player['position'][1] + dy

        # Boundary check
        if not self.env._within_arena(new_x, new_y):
            return -float('inf')  # Invalid move, assign low reward

        new_position = (new_x, new_y)
        cell_value = self.env.grid[new_x, new_y]

        # Check self-elimination by moving into own trail
        if new_position in player['trail']:
            return self.env.reward_config['self_elimination_penalty']

        # Initial reward calculation
        reward = 0

        # Empty cell: extend trail
        if cell_value == 0:
            player['position'] = new_position
            self.env.grid[new_x, new_y] = -player['id']
            player['trail'].append(new_position)

        elif cell_value == player['id'] and player['trail']:
            # Close loop and convert trail to territory
            reward += self._evaluate_trail_to_territory(player_idx)

        elif cell_value < 0 and cell_value != -player['id']:
            # Opponent elimination
            reward += self.env.reward_config['opponent_elimination_reward']

        # Penalty for long trails
        if len(player['trail']) > self.env.reward_config['max_trail_length']:
            reward += self.env.reward_config['long_trail_penalty']

        return reward

    def _evaluate_trail_to_territory(self, player_idx):
        """
        Calculate reward for closing a loop and converting trail to territory
        """
        player = self.env.players[player_idx]
        reward = 0
        area_captured = len(player['trail'])  # Simplified area calculation

        # Calculate reward based on captured area
        reward += (area_captured ** 1.5) * self.env.reward_config['territory_capture_reward_per_cell']
        reward += self.env.reward_config['loop_closure_bonus']
        
        # Clear trail for this player
        for x, y in player['trail']:
            self.env.grid[x, y] = player['id']
        player['trail'] = []

        return reward
