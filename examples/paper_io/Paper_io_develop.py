import random
import numpy as np
from collections import deque  # For efficient BFS
import pygame  # type: ignore
from gym.spaces import Box, Discrete

from examples.paper_io.utils.render import render_game

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2, render=False, max_steps=1000,
                 partial_observability=False, 
                 trail_obs_representation="negative",    # "negative" or "border"
                 territory_obs_representation="raw"):      # "raw" or "transformed"
        """
        Initialize the Paper.io environment.

        Parameters:
          - grid_size: size of the grid.
          - num_players: number of players.
          - render: whether to initialize rendering.
          - max_steps: maximum steps per episode.
          - partial_observability: whether to use a local observation window.
          - trail_obs_representation: 
                "negative" => trails remain as negative player indices.
                "border"   => own trail is shown as BORDER_VALUE (99) and enemy trails as -77.
          - territory_obs_representation:
                "raw"         => territory cells are returned as player IDs.
                "transformed" => own territory appears as OWN_TERRITORY_VALUE (111)
                                 and enemy territory as ENEMY_TERRITORY_VALUE (-88).
        """
        self.CAMPING_PENALTY = False

        self.reward_config = {
            'self_elimination_penalty': -150,
            'camping_penalty': self.CAMPING_PENALTY,
            'max_camping_penalty_per_episode': 30,
            'trail_reward': 15,
            'max_trail_reward_count': 5,
            'max_trail_length': 10,
            'long_trail_penalty': -15,
            'distance_penalty_factor': 0.5,
            'opponent_elimination_reward': 300,
            'opponent_elimination_penalty': -100,
            'enemy_territory_capture_reward_per_cell': 30,
            'territory_loss_penalty_per_cell': -20,
            'elimination_reward_modifier': 0.50,
            'elimination_static_penalty': -800,
            'territory_capture_reward_per_cell': 40,
            'shaping_return_bonus': 30,
            'shaping_distance_factor': 3,
            'expansion_bonus': 50,
            'expansion_interval': 25,
            'expansion_growth_threshold': 1,
            'exploration_reward': 1,
        }

        self.INCREMENTAL_CAMPING_PENALTY = -2
        self.BORDER_VALUE = 99
        self.grid_size = grid_size
        self.num_players = num_players
        self.cell_size = 15
        self.window_size = self.grid_size * self.cell_size
        self.render_game = render
        self.screen = None
        self.partial_observability = partial_observability

        # Set default overall transformation settings.
        # "negative" leaves trails unchanged; "border" converts:
        #   own trail -> BORDER_VALUE (99)
        #   enemy trails -> -77.
        self.trail_obs_representation = trail_obs_representation

        # "raw" leaves territory as player IDs; "transformed" converts:
        #   own territory -> OWN_TERRITORY_VALUE (111)
        #   enemy territory -> ENEMY_TERRITORY_VALUE (-88).
        self.territory_obs_representation = territory_obs_representation
        self.OWN_TERRITORY_VALUE = 111
        self.ENEMY_TERRITORY_VALUE = -88

        # Dictionary to store per-agent observation configuration.
        # Example: {0: {"trail": "border", "territory": "transformed"}, ...}
        self.agent_observation_config = {}

        if self.render_game:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Paper.io with Pygame")
            self.clock = pygame.time.Clock()

        self.max_steps = max_steps
        self.steps_taken = 0

        # Trackers for statistics on trail lengths
        self.trail_length_sums = [0] * self.num_players
        self.trail_length_counts = [0] * self.num_players

        # Other trackers
        self.directions = [(0, 1)] * self.num_players
        self.eliminations_by_agent = [0] * self.num_players
        self.self_eliminations_by_agent = [0] * self.num_players
        self.agent_wins = [0] * self.num_players
        self.cumulative_rewards = [0] * self.num_players
        self.initial_territories = [0] * self.num_players
        self.enemy_territory_captured = [0] * self.num_players

        self.reset()

        # Observation and action spaces
        self.observation_spaces = [
            Box(low=-self.num_players, high=self.num_players,
                shape=(self.grid_size, self.grid_size), dtype=np.int8)
            for _ in range(self.num_players)
        ]
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

        self.eliminations_by_agent = [0] * self.num_players
        self.self_eliminations_by_agent = [0] * self.num_players
        self.cumulative_rewards = [0] * self.num_players

        # Reset trail length trackers
        self.trail_length_sums = [0] * self.num_players
        self.trail_length_counts = [0] * self.num_players

        self.initial_territories = [0] * self.num_players
        self.enemy_territory_captured = [0] * self.num_players

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
                'trail': set(),
                'territory': 9,
                'steps_in_own_territory': 0,
                'trail_reward_count': 0,
                'camping_penalty_multiplier': 1,
                'last_territory': 9,
                'last_expansion_step': 0,
                'camping_penalty_accumulated': 0,
            })
            self.grid[x:x+3, y:y+3] = player_id

        for i in range(self.num_players):
            self.initial_territories[i] = self.players[i]['territory']

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
            old_distance = self._distance_from_territory(player_id, x, y)

            # Update direction
            if action == 0:
                self.directions[i] = self._turn_left(self.directions[i])
            elif action == 1:
                self.directions[i] = self._turn_right(self.directions[i])

            dx, dy = self.directions[i]
            new_x, new_y = x + dx, y + dy

            # Check grid boundaries
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
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

                # Stepping on opponent's trail => eliminate opponent
                if cell_value < 0 and cell_value != -player_id:
                    owner_id = -cell_value
                    rewards[owner_id - 1] += self.reward_config['opponent_elimination_penalty']
                    rewards[i] += self.reward_config['opponent_elimination_reward']
                    self.eliminations_by_agent[i] += 1
                    self._process_elimination(owner_id - 1)

                # Update player position
                player['position'] = (new_x, new_y)

                # Compute new Manhattan distance to nearest territory cell
                new_distance = self._distance_from_territory(player_id, new_x, new_y)
                if new_distance < old_distance:
                    improvement = old_distance - new_distance
                    rewards[i] += improvement * self.reward_config['shaping_distance_factor']
                if new_distance == 0 and old_distance > 0:
                    rewards[i] += self.reward_config['shaping_return_bonus']

                # Stepping into empty or own trail cell
                if cell_value == 0 or cell_value == -player_id:
                    self.grid[new_x, new_y] = -player_id  # keep internal trail as negative
                    player['trail'].add((new_x, new_y))
                    if player['trail_reward_count'] < self.reward_config['max_trail_reward_count']:
                        rewards[i] += self.reward_config['trail_reward']
                        player['trail_reward_count'] += 1

                # Returning to own territory => convert trail to territory
                elif cell_value == player_id and player['trail']:
                    # Record the trail length before converting
                    self.trail_length_sums[player_id - 1] += len(player['trail'])
                    self.trail_length_counts[player_id - 1] += 1
                    self.convert_trail_to_territory(player_id, rewards)
                    player['trail_reward_count'] = 0

                # Entering enemy territory
                elif cell_value > 0 and cell_value != player_id:
                    owner_id = cell_value
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].add((new_x, new_y))

                # Check if trail length exceeds maximum and apply penalties
                if len(player['trail']) > self.reward_config['max_trail_length']:
                    new_distance_combined = self._distance_from_territory(player_id, new_x, new_y)
                    base_penalty = self.reward_config['long_trail_penalty']
                    extra_distance = max(0, new_distance_combined - 3)
                    distance_penalty = extra_distance * self.reward_config['distance_penalty_factor']
                    improvement = max(0, old_distance - new_distance_combined)
                    shaping_reward = improvement * self.reward_config['shaping_distance_factor']
                    net_effect = base_penalty - distance_penalty + shaping_reward
                    rewards[i] += net_effect
                    if new_distance_combined == 0 and old_distance > 0:
                        rewards[i] += self.reward_config['shaping_return_bonus']
                elif len(player['trail']) < self.reward_config['max_trail_length'] and cell_value != player_id:
                    rewards[i] += self.reward_config['exploration_reward']

                # Check camping: count steps in own territory
                if self.grid[new_x, new_y] == player_id:
                    player['steps_in_own_territory'] += 1
                else:
                    player['steps_in_own_territory'] = 0
                    player['camping_penalty_multiplier'] = 1

                if player['steps_in_own_territory'] > 0 and player['steps_in_own_territory'] % 5 == 0 and self.CAMPING_PENALTY:
                    penalty = self.INCREMENTAL_CAMPING_PENALTY * player['camping_penalty_multiplier']
                    self._apply_camping_penalty(i, penalty)
                    player['camping_penalty_multiplier'] *= 1.5

        if self.steps_taken % self.reward_config['expansion_interval'] == 0:
            for i in range(self.num_players):
                if self.alive[i]:
                    player = self.players[i]
                    territory_growth = player['territory'] - player.get('last_territory', player['territory'])
                    if territory_growth >= self.reward_config['expansion_growth_threshold']:
                        rewards[i] += self.reward_config['expansion_bonus']
                    player['last_territory'] = player['territory']
                    player['last_expansion_step'] = self.steps_taken

        for i, rew in enumerate(rewards):
            self.cumulative_rewards[i] += rew

        if self.steps_taken >= self.max_steps:
            done = True

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]

        if done:
            winners = []
            max_reward = max(self.cumulative_rewards)
            # Find all agents with the maximum cumulative reward.
            candidates = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]
            if len(candidates) == 1:
                winners = candidates
            else:
                # Among candidates, compute territory gain for each.
                territory_gains = [self.players[i]['territory'] - self.initial_territories[i] for i in candidates]
                max_territory = max(territory_gains)
                winners = [candidates[i] for i, gain in enumerate(territory_gains) if gain == max_territory]

            if winners:
                for w in winners:
                    self.agent_wins[w] += 1

            average_trail_by_agent = []
            for i in range(self.num_players):
                if self.trail_length_counts[i] > 0:
                    avg_trail = self.trail_length_sums[i] / self.trail_length_counts[i]
                else:
                    avg_trail = 0.0
                average_trail_by_agent.append(avg_trail)

            territory_increase_by_agent = []
            for i in range(self.num_players):
                start_territory = self.initial_territories[i]
                end_territory = self.players[i]['territory']
                territory_increase_by_agent.append(end_territory - start_territory)

            info = {
                'eliminations_by_agent': self.eliminations_by_agent[:],
                'self_eliminations_by_agent': self.self_eliminations_by_agent[:],
                'winners': winners,
                'cumulative_rewards': self.cumulative_rewards[:],
                'territory_by_agent': [p['territory'] for p in self.players],
                'average_trail_by_agent': average_trail_by_agent,
                'average_territory_increase_by_agent': territory_increase_by_agent,
                'enemy_territory_captured': self.enemy_territory_captured[:],
            }
        else:
            info = {}

        return observations, rewards, done, info

    def render(self, player_colors=None):
        if self.render_game and self.screen:
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
        # If the agent has an active trail, record its length before clearing.
        if player['trail']:
            self.trail_length_sums[player_id - 1] += len(player['trail'])
            self.trail_length_counts[player_id - 1] += 1

        for (cx, cy) in player['trail']:
            self.grid[cx, cy] = 0
        player['trail'].clear()

        self.grid[self.grid == player_id] = 0
        player['territory'] = 0

        self.cumulative_rewards[idx] += self.reward_config['elimination_static_penalty']

        while True:
            x = np.random.randint(5, self.grid_size - 5)
            y = np.random.randint(5, self.grid_size - 5)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                subgrid = self.grid[x:x+3, y:y+3]
                if subgrid.max() == 0:
                    break
        player['position'] = (x, y)
        self.grid[x:x+3, y:y+3] = player_id
        player['territory'] = 9
        self.directions[idx] = self._random_direction()
        player['steps_in_own_territory'] = 0
        player['trail_reward_count'] = 0
        player['camping_penalty_accumulated'] = 0

    def get_observation_for_player(self, player_idx):
        # Get raw observation: full grid or local window.
        if self.partial_observability:
            player = self.players[player_idx]
            x, y = player['position']
            obs_radius = 5
            x_min = max(0, x - obs_radius)
            x_max = min(self.grid_size, x + obs_radius + 1)
            y_min = max(0, y - obs_radius)
            y_max = min(self.grid_size, y + obs_radius + 1)
            local_grid = self.grid[x_min:x_max, y_min:y_max]
            padded_grid = np.full((2 * obs_radius + 1, 2 * obs_radius + 1), -127, dtype=np.int8)
            x_offset = x_min - (x - obs_radius)
            y_offset = y_min - (y - obs_radius)
            padded_grid[x_offset:x_offset + local_grid.shape[0],
                        y_offset:y_offset + local_grid.shape[1]] = local_grid
            obs = padded_grid
        else:
            obs = self.grid.copy()

        # Get per-agent configuration (if any); otherwise, use default.
        config = self.agent_observation_config.get(player_idx, {})
        trail_rep = config.get("trail", self.trail_obs_representation)
        territory_rep = config.get("territory", self.territory_obs_representation)

        # Transform trail representation.
        if trail_rep == "border":
            own_id = player_idx + 1
            own_trail_mask = (obs < 0) & (np.abs(obs) == own_id)
            enemy_trail_mask = (obs < 0) & (np.abs(obs) != own_id)
            obs[own_trail_mask] = self.BORDER_VALUE
            obs[enemy_trail_mask] = -77

        # Transform territory representation.
        if territory_rep == "transformed":
            own_id = player_idx + 1
            own_territory_mask = (obs > 0) & (obs == own_id)
            enemy_territory_mask = (obs > 0) & (obs != own_id) & (obs != self.BORDER_VALUE)
            obs[own_territory_mask] = self.OWN_TERRITORY_VALUE
            obs[enemy_territory_mask] = self.ENEMY_TERRITORY_VALUE

        return obs

    def convert_trail_to_territory(self, player_id, rewards):
        player = self.players[player_id - 1]
        # Record the trail length before conversion
        self.trail_length_sums[player_id - 1] += len(player['trail'])
        self.trail_length_counts[player_id - 1] += 1
        for (cx, cy) in player['trail']:
            self.grid[cx, cy] = player_id
            player['territory'] += 1

        captured_area = self.capture_area(player_id, rewards)
        total_area = len(player['trail']) + captured_area
        player['trail'].clear()

        bonus = (total_area ** 1.2) * self.reward_config['territory_capture_reward_per_cell']
        rewards[player_id - 1] += bonus
        return bonus

    def capture_area(self, player_id, rewards):
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

        for row in range(self.grid_size):
            flood_fill(row, 0)
            flood_fill(row, self.grid_size - 1)
        for col in range(self.grid_size):
            flood_fill(0, col)
            flood_fill(self.grid_size - 1, col)

        enclosed_area = ~filled & mask
        coords = np.where(enclosed_area)
        for (rx, ry) in zip(coords[0], coords[1]):
            old_id = self.grid[rx, ry]
            if old_id > 0 and old_id != player_id:
                self.players[old_id - 1]['territory'] -= 1
                self.enemy_territory_captured[player_id - 1] += 1
            elif old_id == 0:
                rewards[player_id - 1] += self.reward_config['territory_capture_reward_per_cell']
            self.grid[rx, ry] = player_id
            self.players[player_id - 1]['territory'] += 1

        return len(coords[0])

    def _distance_from_territory(self, player_id, x, y):
        territory_indices = np.argwhere(self.grid == player_id)
        if territory_indices.size == 0:
            return 9999
        distances = np.abs(territory_indices[:, 0] - x) + np.abs(territory_indices[:, 1] - y)
        return int(distances.min())

    def _apply_camping_penalty(self, i, penalty):
        if penalty >= 0:
            return
        player = self.players[i]
        limit = self.reward_config['max_camping_penalty_per_episode']
        if player['camping_penalty_accumulated'] >= limit:
            return
        needed = abs(penalty)
        room = limit - player['camping_penalty_accumulated']
        actual_penalty = -min(needed, room)
        player['camping_penalty_accumulated'] += min(needed, room)
        self.cumulative_rewards[i] += actual_penalty
