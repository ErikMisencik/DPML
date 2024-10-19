import numpy as np
import cv2
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2):
        # Initialize grid size and number of players
        self.grid_size = grid_size
        self.num_players = num_players
        self.window_name = "Paper.io"  # Game window name for OpenCV
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
        # Render the game grid visually using OpenCV
        cell_size = 15  # Each grid cell size in pixels
        img_size = self.grid_size * cell_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Define player colors
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        # Define the center and radius of the circular arena
        center = (img_size // 2, img_size // 2)
        radius = img_size // 2 - 10  # Decrease the margin to make the arena larger

        # Fill the circle with white color to represent the arena
        cv2.circle(img, center, radius, (255, 255, 255), -1)  # White circular arena

        # Add a 3D-like border around the circular arena
        border_thickness = 10  # Thickness of the border for the 3D effect
        cv2.circle(img, center, radius, (200, 200, 200), border_thickness)  # Light border (top-left)
        cv2.circle(img, center, radius - border_thickness // 2, (100, 100, 100), border_thickness // 2)  # Darker border (bottom-right)

        # Draw the grid based on the player's territories and trails
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                top_left = (y * cell_size, x * cell_size)
                bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)

                # Get the center of the current cell
                cell_center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

                # Only draw cells inside the circular arena
                if np.sqrt((cell_center[0] - center[0]) ** 2 + (cell_center[1] - center[1]) ** 2) < radius:
                    if cell_value > 0:
                        # Territory: make less bright by blending more with white (75% original color, 25% white)
                        player_id = cell_value - 1
                        color = colors[player_id % len(colors)]
                        faded_territory_color = [int(0.25 * 255 + 0.75 * c) for c in color]  # Less bright territory

                        # Draw the main block for territory
                        cv2.rectangle(img, top_left, bottom_right, faded_territory_color, -1)

                        # Subtle highlight on the top-left for territory
                        light_color = [min(255, int(c * 1.1)) for c in faded_territory_color]  # Slightly lighter
                        cv2.line(img, top_left, (bottom_right[0], top_left[1]), light_color, 1)  # Top border
                        cv2.line(img, top_left, (top_left[0], bottom_right[1]), light_color, 1)  # Left border

                        # Subtle shadow on the bottom-right for territory
                        shadow_color = [max(0, int(c * 0.9)) for c in faded_territory_color]  # Slightly darker
                        cv2.line(img, bottom_right, (bottom_right[0], top_left[1]), shadow_color, 1)  # Bottom border
                        cv2.line(img, bottom_right, (top_left[0], bottom_right[1]), shadow_color, 1)  # Right border

                    elif cell_value < 0:
                        # Trail: subtle 3D effect (slightly faded, with lighter shadows/highlights)
                        player_id = -cell_value - 1
                        color = colors[player_id % len(colors)]
                        faded_color = [int(0.75 * 255 + 0.25 * c) for c in color]

                        # Draw the main block for trail
                        cv2.rectangle(img, top_left, bottom_right, faded_color, -1)

                        # Subtle highlight on the top-left for trail
                        light_color = [min(255, int(c * 1.05)) for c in faded_color]  # Very subtle highlight
                        cv2.line(img, top_left, (bottom_right[0], top_left[1]), light_color, 1)  # Top border
                        cv2.line(img, top_left, (top_left[0], bottom_right[1]), light_color, 1)  # Left border

                        # Subtle shadow on the bottom-right for trail
                        shadow_color = [max(0, int(c * 0.95)) for c in faded_color]  # Very subtle shadow
                        cv2.line(img, bottom_right, (bottom_right[0], top_left[1]), shadow_color, 1)  # Bottom border
                        cv2.line(img, bottom_right, (top_left[0], bottom_right[1]), shadow_color, 1)  # Right border

        # Highlight players with a stronger 3D effect
        for i, player in enumerate(self.players):
            if not self.alive[i]:
                continue
            x, y = player['position']
            top_left = (y * cell_size, x * cell_size)
            bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)
            color = [min(255, c + 100) for c in colors[i % len(colors)]]

            # Draw player with a stronger 3D effect
            cv2.rectangle(img, top_left, bottom_right, color, -1)

            # Stronger highlight on the top-left to simulate light source for player
            light_color = [min(255, int(c * 1.3)) for c in color]  # Stronger lighter color
            cv2.line(img, top_left, (bottom_right[0], top_left[1]), light_color, 2)  # Top border
            cv2.line(img, top_left, (top_left[0], bottom_right[1]), light_color, 2)  # Left border

            # Stronger shadow on the bottom-right to simulate depth for player
            shadow_color = [max(0, int(c * 0.6)) for c in color]  # Stronger darker color
            cv2.line(img, bottom_right, (bottom_right[0], top_left[1]), shadow_color, 2)  # Bottom border
            cv2.line(img, bottom_right, (top_left[0], bottom_right[1]), shadow_color, 2)  # Right border

        # Display the grid
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)


    def close(self):
        # Close the game window
        cv2.destroyAllWindows()

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
        cell_size = 15  # Each grid cell size in pixels
        img_size = self.grid_size * cell_size
        center = (img_size // 2, img_size // 2)
        radius = img_size // 2 - 10  # Decrease the margin to make the arena larger

        # Calculate the center of the current cell
        cell_center_x = (y * cell_size) + (cell_size // 2)
        cell_center_y = (x * cell_size) + (cell_size // 2)

        # Check if the distance from the center is within the radius of the circle
        distance = np.sqrt((cell_center_x - center[0]) ** 2 + (cell_center_y - center[1]) ** 2)
        return distance <= radius