import random
import numpy as np
from collections import deque  # For efficient BFS
import pygame  # type: ignore
from gym.spaces import Box, Discrete

from examples.paper_io.utils.render import render_game

class PaperIoEnv:
    BORDER_VALUE = 99

    def __init__(self, grid_size=50, num_players=2, render=False, max_steps=1000, partial_observability=False):
        """
        Initialize the Paper.io environment.
        """
        self.reward_config = {
            'self_elimination_penalty': -300,
            'long_camping_penalty': -300,
            'trail_reward': 40,
            'max_trail_reward': 200,
            'territory_capture_reward_per_cell': 30,
            'max_trail_length': 15,
            'long_trail_penalty': -20,
            'opponent_elimination_reward': 200,
            'opponent_elimination_penalty': -50,
            'enemy_territory_capture_reward_per_cell': 30,
            'territory_loss_penalty_per_cell': -50,
            'elimination_reward_modifier': 0.70,
        }
        self.grid_size = grid_size
        self.num_players = num_players
        self.cell_size = 15
        self.window_size = self.grid_size * self.cell_size
        self.render_game = render
        self.screen = None
        self.partial_observability = partial_observability

        if self.render_game:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Paper.io with Pygame")
            self.clock = pygame.time.Clock()

        self.max_steps = max_steps
        self.steps_taken = 0

        # Trackers
        self.directions = [(0, 1)] * self.num_players
        self.eliminations_by_agent = [0] * self.num_players
        self.self_eliminations_by_agent = [0] * self.num_players
        self.agent_wins = [0] * self.num_players
        self.cumulative_rewards = [0] * self.num_players

        self.reset()

        # Observation space
        self.observation_spaces = [
            Box(low=-self.num_players, high=self.num_players,
                shape=(self.grid_size, self.grid_size), dtype=np.int8)
            for _ in range(self.num_players)
        ]
        # Action space: 0 - turn left, 1 - turn right, 2 - go straight
        self.action_spaces = [Discrete(3) for _ in range(self.num_players)]

    def reset(self):
        """
        Reset the game state and players' positions.
        """
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._add_arena_border()

        self.players = []
        self.alive = [True] * self.num_players
        self.directions = [self._random_direction() for _ in range(self.num_players)]
        self.steps_taken = 0

        # Reset trackers
        self.eliminations_by_agent = [0] * self.num_players
        self.self_eliminations_by_agent = [0] * self.num_players
        self.cumulative_rewards = [0] * self.num_players

        # Place players away from the border
        for i in range(self.num_players):
            while True:
                x = np.random.randint(5, self.grid_size - 5)
                y = np.random.randint(5, self.grid_size - 5)
                if self.grid[x, y] == 0:
                    break
            player_id = i + 1
            self.players.append({
                'position': (x, y),
                'id': player_id,
                # MAIN OPTIMIZATION: store trail in a set for O(1) membership
                'trail': set(),
                'territory': 9,
                'steps_in_own_territory': 0
            })
            self.grid[x : x+3, y : y+3] = player_id

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        return observations

    def _add_arena_border(self):
        """
        Mark the outermost rows/columns with BORDER_VALUE.
        """
        self.grid[0, :] = self.BORDER_VALUE
        self.grid[self.grid_size - 1, :] = self.BORDER_VALUE
        self.grid[:, 0] = self.BORDER_VALUE
        self.grid[:, self.grid_size - 1] = self.BORDER_VALUE

    def step(self, actions):
        rewards = [0] * self.num_players
        done = False
        self.steps_taken += 1
        info = {}

        for i, action in enumerate(actions):
            player = self.players[i]
            (x, y) = player['position']
            player_id = player['id']

            # Update direction
            if action == 0:
                self.directions[i] = self._turn_left(self.directions[i])
            elif action == 1:
                self.directions[i] = self._turn_right(self.directions[i])

            dx, dy = self.directions[i]
            new_x, new_y = x + dx, y + dy

            # Check grid boundaries
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                # Out of grid => elimination
                rewards[i] += self.reward_config['self_elimination_penalty']
                self.self_eliminations_by_agent[i] += 1
                self._process_elimination(i)
                continue

            cell_value = self.grid[new_x, new_y]

            # Border => immediate penalty/elimination
            if cell_value == self.BORDER_VALUE:
                rewards[i] += self.reward_config['self_elimination_penalty']
                self.self_eliminations_by_agent[i] += 1
                self._process_elimination(i)
                continue

            # Move
            if (new_x, new_y) != (x, y):
                # Stepping on own trail => self-elimination
                if (new_x, new_y) in player['trail']:
                    rewards[i] += self.reward_config['self_elimination_penalty']
                    self.self_eliminations_by_agent[i] += 1
                    self._process_elimination(i)
                    continue

                # Stepping on opponent's trail => eliminate them
                if cell_value < 0 and cell_value != -player_id:
                    owner_id = -cell_value
                    rewards[owner_id - 1] += self.reward_config['opponent_elimination_penalty']
                    rewards[i] += self.reward_config['opponent_elimination_reward']
                    self.eliminations_by_agent[i] += 1
                    self._process_elimination(owner_id - 1)

                # Update player position
                player['position'] = (new_x, new_y)

                # Stepping into empty or own trail
                if cell_value == 0 or cell_value == -player_id:
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].add((new_x, new_y))  

                    # Simple trail reward
                    trail_len = len(player['trail'])
                    if trail_len % 2 == 0:
                        rewards[i] += min((trail_len // 2) * self.reward_config['trail_reward'],
                                          self.reward_config['max_trail_reward'])

                # Returning to own territory => close loop
                elif cell_value == player_id and player['trail']:
                    self.convert_trail_to_territory(player_id, rewards)

                # Entering enemy territory
                elif cell_value > 0 and cell_value != player_id:
                    owner_id = cell_value
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].add((new_x, new_y))
                    self.players[owner_id - 1]['territory'] -= 1
                    self.players[player_id - 1]['territory'] += 1

                # Check long trail penalty
                if len(player['trail']) > self.reward_config['max_trail_length']:
                    rewards[i] += self.reward_config['long_trail_penalty']
                
                CAMPING_THRESHOLD = 10
                if self.grid[new_x, new_y] == player_id:
                    player['steps_in_own_territory'] += 1
                else:
                    player['steps_in_own_territory'] = 0

                if player['steps_in_own_territory'] >= CAMPING_THRESHOLD:
                    rewards[i] += self.reward_config['long_camping_penalty']
                    player['steps_in_own_territory'] = 0  # reset after penalizing

        # Update cumulative rewards
        for i, rew in enumerate(rewards):
            self.cumulative_rewards[i] += rew

        # Check if done
        if self.steps_taken >= self.max_steps:
            done = True

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]

        # Determine winner
        if done:
            winners = []
            max_reward = max(self.cumulative_rewards)
            # Single highest
            if self.cumulative_rewards.count(max_reward) == 1:
                winners = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]
            if not winners:
                # Use eliminations as tie-break
                max_elims = max(self.eliminations_by_agent)
                candidates = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]
                winners = [i for i in candidates if self.eliminations_by_agent[i] == max_elims]
                if len(winners) > 1:
                    # Use self-eliminations as final tie-break
                    min_self_elims = min(self.self_eliminations_by_agent[idx] for idx in winners)
                    winners = [idx for idx in winners if self.self_eliminations_by_agent[idx] == min_self_elims]

            if winners:
                for w in winners:
                    self.agent_wins[w] += 1

            info = {
                'eliminations_by_agent': self.eliminations_by_agent[:],
                'self_eliminations_by_agent': self.self_eliminations_by_agent[:],
                'winners': winners,
                'cumulative_rewards': self.cumulative_rewards[:],
                'territory_by_agent': [p['territory'] for p in self.players],
            }
        else:
            info = {}

        return observations, rewards, done, info

    def render(self, player_colors=None):
        if self.render_game and self.screen:
            # Assumes an external render_game function
            render_game(self.screen, self.grid, self.players, self.alive,
                        self.cell_size, self.window_size, self.num_players,
                        self.steps_taken, player_colors)
            pygame.display.flip()
            self.clock.tick(30)

    def _turn_left(self, direction):
        dx, dy = direction
        return -dy, dx

    def _turn_right(self, direction):
        dx, dy = direction
        return dy, -dx

    def _random_direction(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return random.choice(directions)

    def _process_elimination(self, idx):
        player = self.players[idx]
        player_id = player['id']

        # Clear trail
        for (cx, cy) in player['trail']:
            self.grid[cx, cy] = 0
        player['trail'].clear()

        # Clear territory
        self.grid[self.grid == player_id] = 0
        player['territory'] = 0

        # Survival penalty
        self.cumulative_rewards[idx] *= self.reward_config['elimination_reward_modifier']

        # Respawn
        while True:
            x = np.random.randint(5, self.grid_size - 5)
            y = np.random.randint(5, self.grid_size - 5)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Check 3x3 area
                subgrid = self.grid[x : x+3, y : y+3]
                if subgrid.max() == 0:  # area is free
                    break
        player['position'] = (x, y)
        self.grid[x : x+3, y : y+3] = player_id
        player['territory'] = 9
        self.directions[idx] = self._random_direction()

    def get_observation_for_player(self, player_idx):
        if self.partial_observability:
            player = self.players[player_idx]
            x, y = player['position']
            obs_radius = 5
            x_min = max(0, x - obs_radius)
            x_max = min(self.grid_size, x + obs_radius + 1)
            y_min = max(0, y - obs_radius)
            y_max = min(self.grid_size, y + obs_radius + 1)

            local_grid = self.grid[x_min:x_max, y_min:y_max]
            padded_grid = np.full((2*obs_radius + 1, 2*obs_radius + 1), -127, dtype=np.int8)

            x_offset = x_min - (x - obs_radius)
            y_offset = y_min - (y - obs_radius)

            padded_grid[x_offset : x_offset + local_grid.shape[0],
                        y_offset : y_offset + local_grid.shape[1]] = local_grid
            return padded_grid
        else:
            return self.grid.copy()

    def convert_trail_to_territory(self, player_id, rewards):
        player = self.players[player_id - 1]
        for (cx, cy) in player['trail']:
            self.grid[cx, cy] = player_id
            player['territory'] += 1

        captured_area = self.capture_area(player_id, rewards)
        total_area = len(player['trail']) + captured_area
        player['trail'].clear()

        # Non-linear bonus for area
        bonus = (total_area ** 1.5) * self.reward_config['territory_capture_reward_per_cell']
        rewards[player_id - 1] += bonus
        return bonus

    def capture_area(self, player_id, rewards):
        # BFS flood fill to find which cells are not enclosed
        player_cells = (self.grid == player_id) | (self.grid == -player_id)
        mask = ~player_cells
        filled = np.zeros_like(self.grid, dtype=bool)

        def flood_fill(start_x, start_y):
            queue = deque()
            queue.append((start_x, start_y))
            while queue:
                sx, sy = queue.popleft()
                if (0 <= sx < self.grid_size) and (0 <= sy < self.grid_size):
                    if not filled[sx, sy] and mask[sx, sy]:
                        filled[sx, sy] = True
                        queue.append((sx - 1, sy))
                        queue.append((sx + 1, sy))
                        queue.append((sx, sy - 1))
                        queue.append((sx, sy + 1))

        # Flood fill from the outer boundary
        for row in range(self.grid_size):
            flood_fill(row, 0)
            flood_fill(row, self.grid_size - 1)
        for col in range(self.grid_size):
            flood_fill(0, col)
            flood_fill(self.grid_size - 1, col)

        enclosed_area = ~filled & mask
        # Turn enclosed area into territory
        coords = np.where(enclosed_area)
        for (rx, ry) in zip(coords[0], coords[1]):
            old_id = self.grid[rx, ry]
            if old_id > 0 and old_id != player_id:
                self.players[old_id - 1]['territory'] -= 1
            elif old_id == 0:
                rewards[player_id - 1] += self.reward_config['territory_capture_reward_per_cell']

            self.grid[rx, ry] = player_id
            self.players[player_id - 1]['territory'] += 1

        return len(coords[0])  # number of enclosed cells
