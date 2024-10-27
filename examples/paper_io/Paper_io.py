import random
import numpy as np
from examples.paper_io.utils.render import render_game
import pygame
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2, render=False):
        # Initialization remains the same
        #captured_area reward = len(player['trail']) + captured_area * self.reward_config['territory_capture_reward_per_cell']
        self.reward_config = {
            'self_elimination_penalty': -300,  # Increased penalty
            'trail_reward': 2,  # Reduced trail reward per 3 steps
            'max_trail_reward': 20,
            'territory_capture_reward_per_cell': 10,  
            'opponent_elimination_reward': 150,  # Increased reward
            'opponent_elimination_penalty': -50,  # Increased penalty for being eliminated
            'enemy_territory_capture_reward_per_cell': 10,  # Increased reward per cell
            'territory_loss_penalty_per_cell': -10  # Increased penalty per cell lost
        }
        self.steps_taken = 0  # Initialize steps
        self.grid_size = grid_size
        self.num_players = num_players
        self.cell_size = 15  
        self.window_size = self.grid_size * self.cell_size 
        self.render_game = render  
        self.screen = None
        if self.render_game:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Paper.io with Pygame")
            self.clock = pygame.time.Clock()

        # Initialize players' directions
        self.directions = [(0, 1)] * self.num_players  # Default to moving right

        # Initialize tracking variables
        self.eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.self_eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.agent_wins = [0 for _ in range(self.num_players)]
        self.cumulative_rewards = [0 for _ in range(self.num_players)]

        self.reset()

        # Observation space remains the same
        self.observation_spaces = [
            Box(low=-self.num_players, high=self.num_players, shape=(self.grid_size, self.grid_size), dtype=np.int8)
            for _ in range(self.num_players)
        ]

        # Action space is now 3: 0 - turn left, 1 - turn right, 2 - go straight
        self.action_spaces = [Discrete(3) for _ in range(self.num_players)]

    def reset(self):
        # Reset game state and players' positions
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.players = []
        self.alive = [True] * self.num_players
        self.directions = [self._random_direction() for _ in range(self.num_players)]  # Random starting directions
        self.steps_taken = 0

        # Reset tracking variables
        self.eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.self_eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.cumulative_rewards = [0 for _ in range(self.num_players)]

        for i in range(self.num_players):
            while True:
                x = np.random.randint(5, self.grid_size - 5)
                y = np.random.randint(5, self.grid_size - 5)
                if self.grid[x, y] == 0 and self._within_arena(x, y):
                    break
            position = (x, y)
            player_id = i + 1
            self.players.append({
                'position': position,
                'id': player_id,
                'trail': [],
                'territory': 4  # Start with initial territory
            })
            self.grid[x:x+2, y:y+2] = player_id  # Mark 4 cells as the player's territory

        # Return initial observations for each player
        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        return observations

    def step(self, actions):
        rewards = [0] * self.num_players
        done = False
        eliminations = []
        self.steps_taken += 1
        info = {}

        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue  # Skip eliminated players
            player = self.players[i]
            x, y = player['position']
            player_id = player['id']

            # Update direction based on action
            if action == 0:
                self.directions[i] = self._turn_left(self.directions[i])
            elif action == 1:
                self.directions[i] = self._turn_right(self.directions[i])

            # Move in the current direction
            dx, dy = self.directions[i]
            new_x, new_y = x + dx, y + dy

            if not self._within_arena(new_x, new_y):
                new_x, new_y = x, y  # Stay in the current position

            new_position = (new_x, new_y)
            cell_value = self.grid[new_x, new_y]

            # Self-Elimination: Agent steps on its own trail
            if new_position in player['trail']:
                rewards[i] += self.reward_config['self_elimination_penalty']
                self.self_eliminations_by_agent[i] += 1
                self.alive[i] = False  # Mark the agent as eliminated
                eliminations.append(i)
                done = True  # End the game if self-elimination occurs
                break  # Stop further processing

            # Opponent Elimination: Agent steps on an opponent's trail
            if cell_value < 0 and cell_value != -player_id:
                owner_id = -cell_value
                if self.alive[owner_id - 1]:  # Ensure the owner is still active
                    self.alive[owner_id - 1] = False
                    eliminations.append(owner_id - 1)
                    rewards[owner_id - 1] += self.reward_config['opponent_elimination_penalty']
                    rewards[i] += self.reward_config['opponent_elimination_reward']
                    self.eliminations_by_agent[i] += 1  # Count this as an elimination
                    done = True  # End the game if opponent elimination occurs
                    break  # Stop further processing
                    

            # Update position and trail
            player['position'] = new_position

            if cell_value == 0 or cell_value == -player_id:
                # Moving into empty space or own trail
                self.grid[new_x, new_y] = -player_id
                player['trail'].append(new_position)

                # Gain rewards for creating a trail
                if len(player['trail']) % 3 == 0:
                    rewards[i] += min(len(player['trail']) // 3 * self.reward_config['trail_reward'], 
                                    self.reward_config['max_trail_reward'])

            elif cell_value == player_id and player['trail']:
                # Agent returns to their own territory and closes a loop
                # Convert trail to territory
                self.convert_trail_to_territory(player_id, rewards)
                # Rewards are handled inside convert_trail_to_territory

            elif cell_value > 0 and cell_value != player_id:
                # Agent moves into another agent's territory
                owner_id = cell_value
                if self.alive[owner_id - 1]:
                    # Capture the enemy territory cell
                    self.grid[new_x, new_y] = -player_id  # Mark it as part of the agent's trail
                    player['trail'].append(new_position)
                    # Adjust territories
                    self.players[owner_id - 1]['territory'] -= 1
                    self.players[player_id - 1]['territory'] += 1
                    # Penalize the owner for losing territory
                    rewards[owner_id - 1] += self.reward_config['territory_loss_penalty_per_cell']
                    # Reward the capturing agent
                    rewards[i] += self.reward_config['enemy_territory_capture_reward_per_cell']
                else:
                    # Owner is already eliminated; capture territory directly
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].append(new_position)
                    self.players[player_id - 1]['territory'] += 1

            else:
                # Unhandled cases (should not occur)
                pass

        # Update cumulative rewards
        for i in range(self.num_players):
            self.cumulative_rewards[i] += rewards[i]

        # Process eliminations
        for idx in eliminations:
            self._process_elimination(idx)

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        if sum(self.alive) <= 1:
            done = True

        # Determine the winner based on criteria
        if done:
            winners = []
            # Check if any agent is still alive and eliminated the opponent
            for i in range(self.num_players):
                if self.alive[i] and self.eliminations_by_agent[i] > 0:
                    winners.append(i)

            # If no agent eliminated the other, check for the highest cumulative reward
            if not winners:
                max_reward = max(self.cumulative_rewards)
                if self.cumulative_rewards.count(max_reward) == 1:
                    winners = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]

            # Update win counts for the winning agents if there are any winners
            if winners:
                for i in winners:
                    self.agent_wins[i] += 1

            # Include counts in the info dictionary
            info = {
                'eliminations_by_agent': self.eliminations_by_agent.copy(),
                'self_eliminations_by_agent': self.self_eliminations_by_agent.copy(),
                'winners': winners
            }
        else:
            info = {}

        return observations, rewards, done, info

    def render(self, player_colors=None):
        if self.render_game and self.screen:
            # Use external utility function to render the game
            render_game(self.screen, self.grid, self.players, self.alive, self.cell_size, self.window_size, self.num_players, self.steps_taken, player_colors)
            pygame.display.flip()  # Update the pygame display
            # Limit the frame rate to 30 FPS
            self.clock.tick(30)

    def _turn_left(self, direction):
        # Rotate direction 90 degrees counterclockwise
        dx, dy = direction
        return (-dy, dx)

    def _turn_right(self, direction):
        # Rotate direction 90 degrees clockwise
        dx, dy = direction
        return (dy, -dx)
    
    def _random_direction(self):
        # Randomly choose one of four directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return random.choice(directions)

    def _process_elimination(self, idx):
        # Handle player elimination by removing their trail and territory
        player = self.players[idx]
        for trail_cell in player['trail']:
            x, y = trail_cell
            self.grid[x, y] = 0
        self.grid[self.grid == player['id']] = 0
        player['trail'] = []

    def get_observation_for_player(self, player_idx):
        # Return the current observation (grid state) for a player
        return self.grid.copy()

    def convert_trail_to_territory(self, player_id, rewards):
        player = self.players[player_id - 1]
        # Convert trail cells to territory
        for cell in player['trail']:
            x, y = cell
            self.grid[x, y] = player_id
            self.players[player_id - 1]['territory'] += 1  # Update territory count

        # Capture enclosed areas including enemy territories
        captured_area = self.capture_area(player_id, rewards)

        # Total area captured includes trail cells and enclosed area
        total_captured_area = len(player['trail']) + captured_area

        # Clear the player's trail
        player['trail'] = []

        # Reward based on total area captured
        reward = total_captured_area * self.reward_config['territory_capture_reward_per_cell']
        rewards[player_id - 1] += reward
        return reward

    def capture_area(self, player_id, rewards):
        # Identify player's cells (both territory and trail)
        player_cells = (self.grid == player_id) | (self.grid == -player_id)
        mask = ~player_cells
        filled = np.zeros_like(self.grid, dtype=bool)

        # Flood fill from the borders to find non-enclosed areas
        def flood_fill(start_x, start_y):
            stack = [(start_x, start_y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                    continue
                if filled[x, y] or not mask[x, y]:
                    continue
                filled[x, y] = True
                stack.extend([
                    (x - 1, y), (x + 1, y),
                    (x, y - 1), (x, y + 1)
                ])

        # Perform flood fill from the borders
        for x in range(self.grid_size):
            flood_fill(x, 0)
            flood_fill(x, self.grid_size - 1)
        for y in range(self.grid_size):
            flood_fill(0, y)
            flood_fill(self.grid_size - 1, y)

        # Enclosed area is where filled is False and mask is True
        enclosed_area = ~filled & mask

        # Assign territory to the player and adjust rewards/penalties
        for x, y in zip(*np.where(enclosed_area)):
            old_player_id = self.grid[x, y]
            if old_player_id > 0 and old_player_id != player_id:
                # Enemy territory captured
                self.players[old_player_id - 1]['territory'] -= 1
                rewards[old_player_id - 1] += self.reward_config['territory_loss_penalty_per_cell']
                rewards[player_id - 1] += self.reward_config['enemy_territory_capture_reward_per_cell']
            elif old_player_id == 0:
                # Neutral territory captured
                rewards[player_id - 1] += self.reward_config['territory_capture_reward_per_cell']
            # Update the grid and player's territory count
            self.grid[x, y] = player_id
            self.players[player_id - 1]['territory'] += 1

        # Return the number of cells captured
        return np.sum(enclosed_area)

    def _within_arena(self, x, y):
        """
        Checks if a given position (x, y) is within the circular arena.
        """
        cell_size = self.cell_size
        center = (self.grid_size * cell_size // 2, self.grid_size * cell_size // 2)
        radius = self.grid_size * cell_size // 2 - 20

        # Calculate the center of the current cell
        cell_center_x = (y * cell_size) + (cell_size // 2)
        cell_center_y = (x * cell_size) + (cell_size // 2)

        # Check if the distance from the center is within the radius of the circle
        distance = np.sqrt((cell_center_x - center[0]) ** 2 + (cell_center_y - center[1]) ** 2)
        return distance <= radius

    def close(self):
        if self.render_game:
            pygame.quit()
