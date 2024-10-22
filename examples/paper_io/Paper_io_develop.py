import random
import numpy as np
from examples.paper_io.utils.render import render_game
import pygame
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2, render=False):
        # Initialization stays the same
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
                'territory': 1  # Start with 1 territory (initial position)
            })
            self.grid[x, y] = player_id  # Mark initial player territory

        # Return initial observations for each player
        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        return observations

    def step(self, actions):
        rewards = [0] * self.num_players
        done = False
        eliminations = []

        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue  # Skip eliminated players
            player = self.players[i]
            x, y = player['position']
            player_id = player['id']

            # Update direction based on action (turn left, turn right, or go straight)
            if action == 0:  # Turn left
                self.directions[i] = self._turn_left(self.directions[i])
            elif action == 1:  # Turn right
                self.directions[i] = self._turn_right(self.directions[i])

            # Move forward in the current direction
            dx, dy = self.directions[i]
            new_x, new_y = x + dx, y + dy

            if not self._within_arena(new_x, new_y):
                new_x, new_y = x, y  # Stay in the current position

            new_position = (new_x, new_y)
            cell_value = self.grid[new_x, new_y]

            # Check for self-collision (stepping on own trail)
            if new_position in player['trail']:
                # Self-elimination: mark the player as eliminated and apply a negative reward
                self.alive[i] = False
                eliminations.append(i)
                rewards[i] -= 15  # Penalty for self-elimination
                continue  # Skip further processing for this player

            # Handle collisions and territory control (same logic as before)
            if cell_value == 0 or cell_value == player_id or cell_value == -player_id:
                player['position'] = new_position
                if cell_value == 0 or cell_value == -player_id:
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].append(new_position)

                    if len(player['trail']) % 3 == 0:
                        rewards[i] += 3

                elif cell_value == player_id and player['trail']:
                    rewards[i] += self.convert_trail_to_territory(player_id, rewards)
                    rewards[i] += self.players[i]['territory']
            else:
                if cell_value < 0:
                    owner_id = -cell_value
                    if self.alive[owner_id - 1]:
                        self.alive[owner_id - 1] = False
                        eliminations.append(owner_id - 1)
                        rewards[owner_id - 1] -= 10
                        rewards[i] += 10 
                player['position'] = new_position
                self.grid[new_x, new_y] = -player_id
                player['trail'].append(new_position)

        # Process eliminations
        for idx in eliminations:
            self._process_elimination(idx)

        # # Reward for surviving a step
        # for i in range(self.num_players):
        #     if self.alive[i]:
        #         rewards[i] += 1  # Reward for survival

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        if sum(self.alive) <= 1:
            done = True

        return observations, rewards, done, {}


    def render(self):
        if self.render_game and self.screen:
            # Use external utility function to render the game
            render_game(self.screen, self.grid, self.players, self.alive, self.cell_size, self.window_size, self.num_players)
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
        # Randomly choose one of four directions: (Up, Down, Left, Right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        return random.choice(directions)

    def _get_new_position(self, x, y, action):
        # This method is now obsolete but kept for legacy reference
        return x, y
    
    def close(self):
        if self.render_game:
            pygame.quit()

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
        # Convert player's trail into permanent territory and return reward for captured area
        player = self.players[player_id - 1]
        captured_area = 0  # Initialize captured area size
        for cell in player['trail']:
            x, y = cell
            self.grid[x, y] = player_id
            captured_area += 1  # Increment captured territory count
        captured_area += self.capture_area(player_id, rewards)  # Pass rewards to capture_area
        player['trail'] = []

        # Reward based on how much area was captured
        return captured_area * 3.5 

    def capture_area(self, player_id, rewards):
        # Implement area capture logic and track territories lost by other players
        player_cells = (self.grid == player_id) | (self.grid == -player_id)
        mask = ~player_cells
        filled = np.zeros_like(self.grid, dtype=bool)
        territory_lost = [0] * self.num_players  # Track territory lost for each player

        def flood_fill(start_x, start_y):
            stack = [(start_x, start_y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                    continue
                if filled[x, y] or not mask[x, y]:
                    continue
                filled[x, y] = True
                stack.append((x - 1, y))
                stack.append((x + 1, y))
                stack.append((x, y - 1))
                stack.append((x, y + 1))

        for x in range(self.grid_size):
            flood_fill(x, 0)
            flood_fill(x, self.grid_size - 1)
        for y in range(self.grid_size):
            flood_fill(0, y)
            flood_fill(self.grid_size - 1, y)

        enclosed_area = ~filled & mask

        # Assign territory to the player and subtract it from others
        for x, y in zip(*np.where(enclosed_area)):
            old_player_id = self.grid[x, y]
            if old_player_id > 0 and old_player_id != player_id:
                # Subtract territory from the original owner
                territory_lost[old_player_id - 1] += 1
        self.grid[enclosed_area] = player_id

        # Penalize players who lost territory
        for i in range(self.num_players):
            if i != player_id - 1 and self.alive[i]:
                self.players[i]['territory'] -= territory_lost[i]
                rewards[i] -= territory_lost[i]  # Penalty for losing territory

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
