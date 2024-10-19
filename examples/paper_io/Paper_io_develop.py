import numpy as np
import pygame
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2):
       # Initialize grid size and number of players
        self.grid_size = grid_size
        self.num_players = num_players
        self.cell_size = 15  # Each grid cell size in pixels

        # Initialize Pygame display
        self.window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Paper.io with Pygame")

        # Initialize Pygame clock to control frame rate
        self.clock = pygame.time.Clock()  # Add this line

        # Other initializations
        self.reset()

        # Define observation and action spaces for each player
        self.observation_spaces = [
            Box(
                low=-self.num_players,
                high=self.num_players,
                shape=(self.grid_size, self.grid_size),
                dtype=np.int8
            )
            for _ in range(self.num_players)
        ]
        self.action_spaces = [Discrete(4) for _ in range(self.num_players)]  # Up, Down, Left, Right

        # self.clock = pygame.time.Clock()

    def reset(self):
        # Reset game state and players' positions
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.players = []
        self.alive = [True] * self.num_players

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
        # Process actions for each player
        rewards = [0] * self.num_players
        done = False
        eliminations = []

        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue  # Skip eliminated players
            player = self.players[i]
            x, y = player['position']
            player_id = player['id']

            # Determine the new position based on action
            new_x, new_y = self._get_new_position(x, y, action)
            if not self._within_arena(new_x, new_y):
                # Prevent the player from moving outside the circular arena
                new_x, new_y = x, y  # Stay in the current position
            new_position = (new_x, new_y)
            cell_value = self.grid[new_x, new_y]

            # Handle collisions and territory control
            if cell_value == 0 or cell_value == player_id or cell_value == -player_id:
                player['position'] = new_position
                if cell_value == 0 or cell_value == -player_id:
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].append(new_position)
                elif cell_value == player_id and player['trail']:
                    rewards[i] += self.convert_trail_to_territory(player_id)
            else:
                if cell_value < 0:
                    owner_id = -cell_value
                    if self.alive[owner_id - 1]:
                        self.alive[owner_id - 1] = False
                        eliminations.append(owner_id - 1)
                        rewards[owner_id - 1] -= 1
                        rewards[i] += 1
                player['position'] = new_position
                self.grid[new_x, new_y] = -player_id
                player['trail'].append(new_position)

        for idx in eliminations:
            self._process_elimination(idx)

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        if sum(self.alive) <= 1:
            done = True

        return observations, rewards, done, {}

    def render(self):
        # Fill background with white
        self.screen.fill((255, 255, 255))

        # Define player colors
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        # Draw the circular arena
        center = (self.window_size // 2, self.window_size // 2)
        radius = self.window_size // 2 - 10

        # Create a transparent surface for drawing elements with alpha (transparency)
        arena_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)

        # Draw a white circular arena
        pygame.draw.circle(arena_surface, (255, 255, 255), center, radius)

        # Draw trails and territories on the arena surface
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                top_left = (y * self.cell_size, x * self.cell_size)
                rect = pygame.Rect(top_left[0], top_left[1], self.cell_size, self.cell_size)

                if cell_value > 0:
                    # Territory
                    player_id = cell_value - 1
                    pygame.draw.rect(arena_surface, colors[player_id], rect)
                elif cell_value < 0:
                    # Trail
                    player_id = -cell_value - 1
                    faded_color = [int(0.5 * 255 + 0.5 * c) for c in colors[player_id]]
                    pygame.draw.rect(arena_surface, faded_color, rect)

        # Draw the arena with transparency
        self.screen.blit(arena_surface, (0, 0))

       # Highlight players with a stronger 3D effect
        for i, player in enumerate(self.players):
            if not self.alive[i]:  # Check if the player is alive using self.alive list
                continue
            x, y = player['position']
            top_left = (y * self.cell_size, x * self.cell_size)
            bottom_right = ((y + 1) * self.cell_size, (x + 1) * self.cell_size)
            color = [min(255, c + 100) for c in colors[i % len(colors)]]

            # Draw player with a stronger 3D effect
            pygame.draw.rect(self.screen, color, pygame.Rect(top_left[0], top_left[1], self.cell_size, self.cell_size))

            # Stronger highlight on the top-left to simulate light source for player
            light_color = [min(255, int(c * 1.3)) for c in color]  # Stronger lighter color
            pygame.draw.line(self.screen, light_color, top_left, (bottom_right[0], top_left[1]), 2)  # Top border
            pygame.draw.line(self.screen, light_color, top_left, (top_left[0], bottom_right[1]), 2)  # Left border

            # Stronger shadow on the bottom-right to simulate depth for player
            shadow_color = [max(0, int(c * 0.6)) for c in color]  # Stronger darker color
            pygame.draw.line(self.screen, shadow_color, bottom_right, (bottom_right[0], top_left[1]), 2)  # Bottom border
            pygame.draw.line(self.screen, shadow_color, bottom_right, (top_left[0], bottom_right[1]), 2)  # Right border

        # Update display
        pygame.display.flip()

        # Limit the frame rate to 30 FPS
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def _get_new_position(self, x, y, action):
        # Helper function to calculate new position based on action
        if action == 0 and x > 0:
            return x - 1, y  # Move Up
        if action == 1 and x < self.grid_size - 1:
            return x + 1, y  # Move Down
        if action == 2 and y > 0:
            return x, y - 1  # Move Left
        if action == 3 and y < self.grid_size - 1:
            return x, y + 1  # Move Right
        return x, y  # No movement

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

    def convert_trail_to_territory(self, player_id):
        # Convert player's trail into permanent territory and return reward for captured area
        player = self.players[player_id - 1]
        captured_area = 0  # Initialize captured area size
        for cell in player['trail']:
            x, y = cell
            self.grid[x, y] = player_id
            captured_area += 1  # Increment captured territory count
        captured_area += self.capture_area(player_id)
        player['trail'] = []

        # Reward based on how much area was captured
        return captured_area

    def capture_area(self, player_id):
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

        return np.sum(enclosed_area)

    def _within_arena(self, x, y):
        """
        Checks if a given position (x, y) is within the circular arena.
        """
        cell_size = self.cell_size
        center = (self.grid_size * cell_size // 2, self.grid_size * cell_size // 2)
        radius = self.grid_size * cell_size // 2 - 10

        # Calculate the center of the current cell
        cell_center_x = (y * cell_size) + (cell_size // 2)
        cell_center_y = (x * cell_size) + (cell_size // 2)

        # Check if the distance from the center is within the radius of the circle
        distance = np.sqrt((cell_center_x - center[0]) ** 2 + (cell_center_y - center[1]) ** 2)
        return distance <= radius
