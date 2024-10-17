import numpy as np

class GreedyPolicy:
    def __init__(self, env):
        self.env = env

    def run(self, num_steps):
        self.env.reset()

        # Track total rewards and episode lengths
        total_reward = 0.0
        episode_length = 0

        for step in range(num_steps):
            # Simulate the greedy policy for each player
            actions = []
            for i in range(self.env.num_players):
                if not self.env.alive[i]:
                    actions.append(None)  # Player is dead, skip action
                    continue

                # For each player, evaluate all possible actions (0: Up, 1: Down, 2: Left, 3: Right)
                best_action = 0
                best_reward = -float('inf')  # Start with negative infinity

                for action in range(4):  # Actions are [0, 1, 2, 3]
                    simulated_env = self._simulate_action(i, action)
                    reward = self._evaluate_simulation(simulated_env)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = action

                actions.append(best_action)

            # Execute the chosen actions
            obs, rewards, done, _ = self.env.step(actions)
            total_reward += sum(rewards)
            episode_length += 1

            if done:
                print(f"Episode finished after {episode_length} steps with total reward: {total_reward}")
                self.env.reset()
                total_reward = 0.0
                episode_length = 0

    def _simulate_action(self, player_idx, action):
        """
        Simulate what happens if the player takes a given action.
        We return a deep copy of the environment to avoid modifying the actual state.
        """
        import copy
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

        # Check the result of the move
        if cell_value == 0:  # Empty space
            simulated_env.grid[new_x, new_y] = -player['id']
        elif cell_value < 0:  # Trail
            pass  # For simplicity, assume no punishment for walking on trails

        # Update player's position
        player['position'] = new_position
        return simulated_env

    def _evaluate_simulation(self, simulated_env):
        """
        Evaluate the reward of the simulated environment.
        This can be customized to prioritize different behaviors.
        """
        reward = 0
        for player in simulated_env.players:
            # For now, let's say reward is based on how much territory the player owns
            reward += np.sum(simulated_env.grid == player['id'])
        return reward
