import numpy as np
import cv2
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=100, num_players=2):
        self.grid_size = grid_size  # Size of the game grid
        self.num_players = num_players
        self.reset()

        # Define observation and action spaces
        self.observation_space = Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.uint8)
        self.action_space = Discrete(4)  # Up, Down, Left, Right

    def reset(self):
        # 0: empty, 1: player territory, 2: player trail
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.players = []
        self.alive = [True] * self.num_players

        # Initialize players at different positions
        for i in range(self.num_players):
            x = np.random.randint(10, self.grid_size - 10)
            y = np.random.randint(10, self.grid_size - 10)
            position = (x, y)
            self.players.append({
                'position': position,
                'territory': [position],  # List of cells belonging to the player's territory
                'trail': []  # Cells that form the current trail
            })
            self.grid[x, y] = 1  # Mark the territory

        return self.grid.copy()

    def step(self, actions):
        # Actions: List of actions for each player
        rewards = [0] * self.num_players
        done = False
        info = {}

        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue  # Skip if the player is eliminated

            player = self.players[i]
            x, y = player['position']

            # Determine new position based on action
            if action == 0 and x > 0:  # Up
                x -= 1
            elif action == 1 and x < self.grid_size - 1:  # Down
                x += 1
            elif action == 2 and y > 0:  # Left
                y -= 1
            elif action == 3 and y < self.grid_size - 1:  # Right
                y += 1

            new_position = (x, y)
            player['position'] = new_position

            # Check for collision with own territory
            if new_position in player['territory']:
                # Convert trail to territory
                for cell in player['trail']:
                    self.grid[cell] = 1
                    player['territory'].append(cell)
                player['trail'] = []
            else:
                # Leave a trail
                self.grid[new_position] = 2
                player['trail'].append(new_position)

            # Check for collision with other players' trails or territories
            for j, other_player in enumerate(self.players):
                if i == j or not self.alive[j]:
                    continue
                if new_position == other_player['position'] or new_position in other_player['trail']:
                    # Player i is eliminated
                    self.alive[i] = False
                    rewards[i] -= 1
                    rewards[j] += 1  # Optional: reward the player who caused the elimination

        # Prepare the observation (could be customized per player)
        observation = self.grid.copy()

        # Check if the game is over
        if sum(self.alive) <= 1:
            done = True

        return observation, rewards, done, info

    def render(self):
        # Visualize the grid
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    # Territory
                    img[i, j] = [0, 255, 0]
                elif self.grid[i, j] == 2:
                    # Trail
                    img[i, j] = [0, 0, 255]

        # Draw players
        for i, player in enumerate(self.players):
            if not self.alive[i]:
                continue
            x, y = player['position']
            img[x, y] = [255, 0, 0]

        # Resize for better visibility
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Paper.io', img)
        cv2.waitKey(1)
