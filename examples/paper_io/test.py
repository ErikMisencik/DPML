import numpy as np
import cv2
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=70, num_players=2):
        self.grid_size = grid_size  # Reduced size of the game grid
        self.num_players = num_players
        self.reset()

        # Define observation and action spaces
        self.observation_space = Box(
            low=-self.num_players, high=self.num_players, shape=(grid_size, grid_size), dtype=np.int8
        )
        self.action_space = Discrete(4)  # Up, Down, Left, Right

    def reset(self):
        # 0: empty, positive integers: player territory, negative integers: player trail
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.players = []
        self.alive = [True] * self.num_players

        # Initialize players at different positions
        for i in range(self.num_players):
            while True:
                x = np.random.randint(5, self.grid_size - 5)
                y = np.random.randint(5, self.grid_size - 5)
                if self.grid[x, y] == 0:
                    break
            position = (x, y)
            player_id = i + 1  # Assign player IDs starting from 1
            self.players.append({
                'position': position,
                'id': player_id,
                'trail': []  # Cells that form the current trail
            })
            self.grid[x, y] = player_id  # Mark the territory

        return self.get_observation()

    def step(self, actions):
        # Actions: List of actions for each player
        rewards = [0] * self.num_players
        done = False
        info = {}

        # Keep track of eliminations this step to process after all movements
        eliminations = []

        # Move players
        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue  # Skip if the player is eliminated

            player = self.players[i]
            x, y = player['position']
            player_id = player['id']

            # Determine new position based on action
            new_x, new_y = x, y
            if action == 0 and x > 0:  # Up
                new_x -= 1
            elif action == 1 and x < self.grid_size - 1:  # Down
                new_x += 1
            elif action == 2 and y > 0:  # Left
                new_y -= 1
            elif action == 3 and y < self.grid_size - 1:  # Right
                new_y += 1

            new_position = (new_x, new_y)
            cell_value = self.grid[new_x, new_y]

            # Check for collisions
            if cell_value == 0 or cell_value == player_id or cell_value == -player_id:
                # Empty cell, own territory, or own trail
                player['position'] = new_position
                if cell_value == 0:
                    # Leaving a trail
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].append(new_position)
                elif cell_value == -player_id:
                    # Crossing own trail; allowed
                    pass
                elif cell_value == player_id and player['trail']:
                    # Returning to own territory with a trail; capture area
                    self.convert_trail_to_territory(player_id)
            else:
                # Collision with another player's trail or territory
                if cell_value < 0:
                    # Collided with another player's trail; eliminate the owner
                    owner_id = -cell_value
                    if self.alive[owner_id - 1]:
                        self.alive[owner_id - 1] = False
                        eliminations.append(owner_id - 1)
                        rewards[owner_id - 1] -= 1  # Penalize eliminated player
                        rewards[i] += 1  # Reward the player causing elimination
                # Move into the cell (even if it's another player's territory)
                player['position'] = new_position
                # Leaving a trail
                self.grid[new_x, new_y] = -player_id
                player['trail'].append(new_position)

        # After all movements, check if any players have been eliminated and remove their trails
        for idx in eliminations:
            player = self.players[idx]
            # Remove player's trails from the grid
            for trail_cell in player['trail']:
                x, y = trail_cell
                self.grid[x, y] = 0
            player['trail'] = []
            # Remove player's territory from the grid
            self.grid[self.grid == player['id']] = 0

        # Prepare the observation
        observation = self.get_observation()

        # Check if the game is over
        if sum(self.alive) <= 1:
            done = True

        return observation, rewards, done, info

    def convert_trail_to_territory(self, player_id):
        player = self.players[player_id - 1]

        # Convert the trail into territory
        for cell in player['trail']:
            x, y = cell
            self.grid[x, y] = player_id

        # Perform area filling to capture enclosed area
        self.capture_area(player_id)

        # Clear the player's trail
        player['trail'] = []

    def capture_area(self, player_id):
        # Create a mask of the player's territory and trail
        player_cells = (self.grid == player_id) | (self.grid == -player_id)

        # Invert the mask to get the empty and other players' territories
        mask = ~player_cells

        # Flood fill from the boundaries on the inverted mask
        filled = np.zeros_like(self.grid, dtype=bool)

        # Iterative flood fill using a stack
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

        # Flood fill from all edges
        for x in range(self.grid_size):
            if not filled[x, 0]:
                flood_fill(x, 0)
            if not filled[x, self.grid_size - 1]:
                flood_fill(x, self.grid_size - 1)
        for y in range(self.grid_size):
            if not filled[0, y]:
                flood_fill(0, y)
            if not filled[self.grid_size - 1, y]:
                flood_fill(self.grid_size - 1, y)

        # Cells not filled are enclosed by the player's trail
        enclosed_area = ~filled & mask

        # Update the grid: Enclosed area becomes the player's territory
        self.grid[enclosed_area] = player_id

        # Remove any eliminated players whose entire territory has been captured
        for idx, other_player in enumerate(self.players):
            if idx == player_id - 1 or not self.alive[idx]:
                continue
            other_player_id = other_player['id']
            if not np.any(self.grid == other_player_id):
                self.alive[idx] = False

    def get_observation(self):
        # Return a copy of the grid as the observation
        return self.grid.copy()

    def render(self):
        # Visualize the grid
        cell_size = 10  # Size of each cell in pixels
        img_size = self.grid_size * cell_size
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Color map for players
        colors = [
            (0, 255, 0),    # Player 1: Green
            (255, 0, 0),    # Player 2: Blue
            (0, 0, 255),    # Player 3: Red
            (255, 255, 0),  # Player 4: Cyan
            (255, 0, 255),  # Player 5: Magenta
            (0, 255, 255),  # Player 6: Yellow
        ]

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = self.grid[x, y]
                top_left = (y * cell_size, x * cell_size)
                bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)
                if cell_value > 0:
                    # Territory
                    player_id = cell_value - 1
                    color = colors[player_id % len(colors)]
                    cv2.rectangle(img, top_left, bottom_right, color, -1)
                elif cell_value < 0:
                    # Trail
                    player_id = -cell_value - 1
                    color = colors[player_id % len(colors)]
                    color = [c // 2 for c in color]  # Darker color for trail
                    cv2.rectangle(img, top_left, bottom_right, color, -1)

        # Draw players
        for i, player in enumerate(self.players):
            if not self.alive[i]:
                continue
            x, y = player['position']
            top_left = (y * cell_size, x * cell_size)
            bottom_right = ((y + 1) * cell_size, (x + 1) * cell_size)
            color = [min(255, c + 100) for c in colors[i % len(colors)]]  # Brighter color for player
            cv2.rectangle(img, top_left, bottom_right, color, -1)

        # Display the image
        cv2.imshow('Paper.io', img)
        cv2.waitKey(1)
