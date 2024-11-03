import random
import numpy as np
from examples.paper_io.utils.render import render_game
import pygame
from gym.spaces import Box, Discrete

class PaperIoEnv:
    def __init__(self, grid_size=50, num_players=2, render=False, max_steps=1000,  partial_observability=False):
        # Initialization remains the same
        #captured_area reward = len(player['trail']) + captured_area * self.reward_config['territory_capture_reward_per_cell']
        self.reward_config = {
            'self_elimination_penalty': -4000,  # Increased penalty
            'trail_reward': 0,                             # Reduced trail reward per 3 steps          NOT USING RIGHT NOW
            'max_trail_reward': 0,                        # Maximum trail reward per step             NOT USING RIGHT NOW   
            'territory_capture_reward_per_cell': 15,      # Increased reward per cell captured
            'loop_closure_bonus': 1000,                      # New bonus for closing loops of area
            'max_trail_length': 10,                         # Maximum trail length before penalty
            'long_trail_penalty': -50,                      # Penalty for long trails                  # New bonus for closing loops of area
            'opponent_elimination_reward': 200,              # Increased reward for eliminating an opponent   NOT USING RIGHT NOW
            'opponent_elimination_penalty': -100,             # Increased penalty for being eliminated        NOT USING RIGHT NOW
            'enemy_territory_capture_reward_per_cell': 15,  # Increased reward per cell
            'territory_loss_penalty_per_cell': -15,           # Increased penalty per cell lost
        }
        self.steps_taken = 0  # Initialize steps
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

        # Initialize players' directions
        self.directions = [(0, 1)] * self.num_players  # Default to moving right

        # Initialize tracking variables
        self.eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.self_eliminations_by_agent = [0 for _ in range(self.num_players)]
        self.agent_wins = [0 for _ in range(self.num_players)]
        self.cumulative_rewards = [0 for _ in range(self.num_players)]

        self.max_steps = max_steps  # Added max_steps parameter

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

        arena_margin = 0  # Adjust if you have a margin
        for i in range(self.num_players):
            while True:
                x = np.random.randint(arena_margin + 5, self.grid_size - arena_margin - 5)
                y = np.random.randint(arena_margin + 5, self.grid_size - arena_margin - 5)
                if self.grid[x, y] == 0:
                    break
            position = (x, y)
            player_id = i + 1
            self.players.append({
                'position': position,
                'id': player_id,
                'trail': [],
                'territory': 9  # Start with initial territory
            })
            self.grid[x:x+3, y:y+3] = player_id  # Mark 4 cells as the player's territory

        # Return initial observations for each player
        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]
        return observations

    def step(self, actions):
        rewards = [0] * self.num_players
        done = False
        self.steps_taken += 1
        info = {}

        for i, action in enumerate(actions):
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
                # Agent cannot move forward; stays in place
                # Optionally, penalize for hitting the edge
                rewards[i] += -1  # Small penalty
                new_x, new_y = x, y
            else:
                # Agent can move
                pass  # Proceed normally

             # Check if the agent has moved
            moved = (new_x, new_y) != (x, y)
            new_position = (new_x, new_y)

            if moved:
                cell_value = self.grid[new_x, new_y]
                # Self-Elimination: Agent steps on its own trail
                if new_position in player['trail']:
                    rewards[i] += self.reward_config['self_elimination_penalty']
                    self.self_eliminations_by_agent[i] += 1
                    self._process_elimination(i)
                    continue  # Skip to next agent

                # Opponent Elimination: Agent steps on an opponent's trail
                if cell_value < 0 and cell_value != -player_id:
                    owner_id = -cell_value
                    # Process respawn of the opponent
                    rewards[owner_id - 1] += self.reward_config['opponent_elimination_penalty']
                    rewards[i] += self.reward_config['opponent_elimination_reward']
                    self.eliminations_by_agent[i] += 1
                    self._process_elimination(owner_id - 1)
                    # Continue processing current agent's move     

                # Update position and trail
                player['position'] = new_position

                if cell_value == 0 or cell_value == -player_id:
                    # Moving into empty space or own trail
                    self.grid[new_x, new_y] = -player_id
                    player['trail'].append(new_position)

                    # Gain rewards for creating a trail    REWARD? HERE
                    # if len(player['trail']) % 3 == 0:
                    #     rewards[i] += min(len(player['trail']) // 3 * self.reward_config['trail_reward'], 
                    #                     self.reward_config['max_trail_reward'])

                elif cell_value == player_id and player['trail']:
                    # Agent returns to their own territory and closes a loop
                    # Convert trail to territory
                    self.convert_trail_to_territory(player_id, rewards)
                    # Rewards are handled inside convert_trail_to_territory

                elif cell_value > 0 and cell_value != player_id:
                    # Agent moves into another agent's territory
                    owner_id = cell_value
                    # Capture the enemy territory cell
                    self.grid[new_x, new_y] = -player_id  # Mark it as part of the agent's trail
                    player['trail'].append(new_position)
                    # Adjust territories
                    self.players[owner_id - 1]['territory'] -= 1
                    self.players[player_id - 1]['territory'] += 1

                else:
                    # Unhandled cases (should not occur)
                    pass

             # Add a penalty for long trails without closure
            max_trail_length = self.reward_config['max_trail_length']
            long_trail_penalty = self.reward_config['long_trail_penalty']
            if len(player['trail']) > max_trail_length:
                rewards[i] += long_trail_penalty  # Penalty for excessive trail length

        # Update cumulative rewards
        for i in range(self.num_players):
            self.cumulative_rewards[i] += rewards[i]

        # Determine if the episode is done
        if self.steps_taken >= self.max_steps:
            done = True

        observations = [self.get_observation_for_player(i) for i in range(self.num_players)]          

       # Determine the winner based on criteria
        if done:
            winners = []
            # Find the agent with the highest cumulative reward
            max_reward = max(self.cumulative_rewards)
            # Check if there is a unique agent with the highest reward
            if self.cumulative_rewards.count(max_reward) == 1:
                winners = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]

            # If multiple agents share the same highest reward, consider additional criteria (optional)
            if not winners:
                # For example, prioritize agents with more eliminations or fewer self-eliminations
                max_eliminations = max(self.eliminations_by_agent)
                candidates = [i for i, r in enumerate(self.cumulative_rewards) if r == max_reward]
                winners = [i for i in candidates if self.eliminations_by_agent[i] == max_eliminations]

                # If still tied, further criteria can be used (e.g., fewer self-eliminations)
                if len(winners) > 1:
                    min_self_eliminations = min(self.self_eliminations_by_agent[i] for i in winners)
                    winners = [i for i in winners if self.self_eliminations_by_agent[i] == min_self_eliminations]

            # Update win counts for the winning agents if there are any winners
            if winners:
                for i in winners:
                    self.agent_wins[i] += 1

            # Include counts in the info dictionary
            info = {
                'eliminations_by_agent': self.eliminations_by_agent.copy(),
                'self_eliminations_by_agent': self.self_eliminations_by_agent.copy(),
                'winners': winners,
                'cumulative_rewards': self.cumulative_rewards.copy(),
                'territory_by_agent': [player['territory'] for player in self.players]  # Add territory info
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
        # Handle player respawn by removing their trail and territory
        player = self.players[idx]
        player_id = player['id']

        # Remove trail from the grid
        for trail_cell in player['trail']:
            x, y = trail_cell
            self.grid[x, y] = 0
        player['trail'] = []

        # Remove territory from the grid
        self.grid[self.grid == player_id] = 0
        player['territory'] = 0  # Reset territory count

        # Reduce cumulative reward by 50%
        self.cumulative_rewards[idx] *= 0.5

        # Respawn the player at a new position with initial territory
        while True:
            x = np.random.randint(5, self.grid_size - 5)
            y = np.random.randint(5, self.grid_size - 5)
            # Check if the area x:x+2, y:y+2 is free and within arena
            if self._within_arena(x, y) and \
            np.all(self.grid[x:x+3, y:y+3] == 0):
                break
        position = (x, y)
        player['position'] = position
        # Set initial territory
        initial_territory_size = 9  # As in reset method
        player['territory'] = initial_territory_size
        self.grid[x:x+3, y:y+3] = player_id  # Mark initial territory on the grid

        # Assign a new random direction
        self.directions[idx] = self._random_direction()

    def get_observation_for_player(self, player_idx):
        if self.partial_observability:
            # Partial Observability: Agent sees a local grid around its position

            # Get the player's current position
            player = self.players[player_idx]
            x, y = player['position']

            # Define the size of the local observation grid (e.g., radius of 3 for a 7x7 grid)
            obs_radius = 3  # You can adjust this value as needed

            # Calculate the bounds of the local grid, ensuring they stay within grid limits
            x_min = max(0, x - obs_radius)
            x_max = min(self.grid_size, x + obs_radius + 1)
            y_min = max(0, y - obs_radius)
            y_max = min(self.grid_size, y + obs_radius + 1)

            # Extract the local grid around the player
            local_grid = self.grid[x_min:x_max, y_min:y_max]

            # Create a padded grid to handle cases where the local grid is at the edge
            padded_grid = np.full((2 * obs_radius + 1, 2 * obs_radius + 1), -127, dtype=np.int8)
            x_offset = x_min - (x - obs_radius)
            y_offset = y_min - (y - obs_radius)
            padded_grid[
                x_offset:x_offset + (x_max - x_min),
                y_offset:y_offset + (y_max - y_min)
            ] = local_grid

            # Return only the local grid as the observation
            return padded_grid
        else:
            # Full Observability: Agent sees the entire grid

            # Return a copy of the full grid
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

        # Non-linear reward calculation
        area_captured = total_captured_area
        loop_closure_bonus = self.reward_config['loop_closure_bonus']

        # Implement non-linear reward function (e.g., area raised to 1.5 power)
        reward = (area_captured ** 1.5) * self.reward_config['territory_capture_reward_per_cell'] + loop_closure_bonus
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
                # rewards[old_player_id - 1] += self.reward_config['territory_loss_penalty_per_cell']
                # rewards[player_id - 1] += self.reward_config['enemy_territory_capture_reward_per_cell']
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
        Checks if a given position (x, y) is within the square arena.
        """
        arena_margin = 0  # Adjust if you want a margin around the edges
        return arena_margin <= x < self.grid_size - arena_margin and \
            arena_margin <= y < self.grid_size - arena_margin

    def close(self):
        if self.render_game:
            pygame.quit()
