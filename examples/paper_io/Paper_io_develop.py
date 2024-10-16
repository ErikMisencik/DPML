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
                if self.grid[x, y] == 0:
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

        # Print rewards collected by each player at the end of this step
        # print(f"Rewards collected: {rewards}")

        return observations, rewards, done, {}

    def render(self):
        # Render the game grid visually using OpenCV
        cell_size = 15  # Each grid cell size in pixels
        img_size = self.grid_size * cell_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Define player colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        # Draw the grid based on the player's territories and trails
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                top_left = (y * cell_size, x * cell_size)
                bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)
                if cell_value > 0:
                    player_id = cell_value - 1
                    color = colors[player_id % len(colors)]
                    cv2.rectangle(img, top_left, bottom_right, color, -1)
                elif cell_value < 0:
                    player_id = -cell_value - 1
                    color = [c // 2 for c in colors[player_id % len(colors)]]
                    cv2.rectangle(img, top_left, bottom_right, color, -1)

        # Draw players in their current position
        for i, player in enumerate(self.players):
            if not self.alive[i]:
                continue
            x, y = player['position']
            top_left = (y * cell_size, x * cell_size)
            bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)
            color = [min(255, c + 100) for c in colors[i % len(colors)]]
            cv2.rectangle(img, top_left, bottom_right, color, -1)

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
