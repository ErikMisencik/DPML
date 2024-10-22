# PaperIoEnv Documentation

## Environment: PaperIoEnv

### Initialization (`__init__` method)

#### Grid Setup:
- The game grid is a 2D numpy array of size `grid_size x grid_size` (default: 70x70).
- Each cell in the grid can have the following values:
  - `0`: Empty cell.
  - Positive integers (`1, 2, ..., num_players`): Cells that are part of a player's territory.
  - Negative integers (`-1, -2, ..., -num_players`): Cells that are part of a player's trail.

#### Players:
- The game supports multiple players (default: 2).
- Each player is represented by:
  - `position`: Current coordinates on the grid.
  - `id`: Unique identifier (starting from 1).
  - `trail`: List of cells that the player has moved through but hasn't yet converted to territory.

### Observation and Action Spaces

#### Observation Space:
- A Box space representing the grid with values ranging from `-num_players` to `num_players`.

#### Action Space:
- A Discrete(4) space representing the possible moves:
  - `0`: Up
  - `1`: Down
  - `2`: Left
  - `3`: Right

### Reset Method (`reset`)
- Initializes or resets the game state.
- Places each player at a random position on the grid that is at least 5 cells away from the edges and other players.
- Marks the initial positions as the player's territory on the grid.

### Step Method (`step`)

#### Inputs:
- A list of actions, one for each player.

#### Process:
- Moves each player according to their action, updating their position and trail.
- Handles collisions:
  - **With Empty Cells or Own Territory**: Player moves freely.
  - **With Own Trail**: Allowed; no special action.
  - **With Another Player's Trail**: Eliminates the owner of the trail.
  - **With Another Player's Territory**: Player moves into the cell, leaving a trail.

#### Capturing Territory:
- If a player returns to their own territory while having a trail, they capture the enclosed area.
- The method `convert_trail_to_territory` is called to handle this.

#### Eliminations:
- Players can be eliminated if another player collides with their trail.
- Eliminated players have their trails and territories removed from the grid.

#### Outputs:
- `Observation`: Updated grid state.
- `Rewards`: List of rewards for each player (rewards are incremented or decremented upon elimination).
- `Done`: Boolean indicating if the game is over (only one player left).
- `Info`: Additional information (currently empty).

### Trail to Territory Conversion (`convert_trail_to_territory`)
- Converts a player's trail into territory.
- Calls `capture_area` to fill in any enclosed areas.

### Area Capture (`capture_area`)
- Performs a flood-fill algorithm to determine enclosed areas.
- Any empty cells not connected to the boundary are converted into the player's territory.
- Eliminates other players if their entire territory is captured.

### Observation Retrieval (`get_observation`)
- Returns a copy of the current grid state.

### Rendering (`render`)
- Uses OpenCV (`cv2`) to visualize the grid.
- Colors are assigned to each player for their territory, trail, and current position.
- Displays the game grid in a window titled 'Paper.io'.

# Training Script Documentation

## Imports and Environment Setup:
- Imports the `PaperIoEnv` environment.
- Imports `RandomPolicy` and `GreedyPolicy` (note: their implementations are not provided here).
- Creates an instance of the environment.

## Policy Selection:
- **GreedyPolicy** is currently selected for training.
- **RandomPolicy** is also mentioned but commented out.

## Training Parameters:
- **Number of Episodes**: EDIT LATER
- **Steps per Episode**: EDIT LATER
- **Rewards Tracking**: Keeps track of total rewards per episode and calculates a moving average.

## Directories Setup:
- Creates directories for saving models and plots based on the policy name.

## Training Loop:
- Iterates over the number of episodes.

### For each episode:
1. **Environment Reset**: Resets the environment at the beginning of each episode.
2. **Episode Reward Initialization**: Initializes the episode reward.
3. **Step Iteration**: Iterates over the steps per episode.
   - **Action Selection**: Currently, actions are sampled randomly using `env.action_spaces[i].sample()` for each player.
   - **Environment Step**: Applies the actions and receives the new observation, rewards, and `done` signal.
   - **Reward Accumulation**: Sums up the rewards.
   - **Progress Display**: Updates a loading bar to show progress within the episode.
4. **Episode Termination**: Breaks the loop if `done` is `True`.

### After each episode:
- Updates the rewards list and calculates the moving average.
- Displays the progress and saves the training graph.

# Reward System for Paper.io Reinforcement Learning Environment

- **Trail Extension Reward**:  
  +1 for every 3 cells added to the trail. Encourages agents to extend their trail before converting it into territory.

- **Territory Conversion Reward**:  
  +3 for each cell converted into permanent territory. Incentivizes agents to secure new territory.

- **Opponent Elimination Reward**:  
  +10 for eliminating an opponent. Rewards aggressive strategies and successful eliminations.

- **Elimination Penalty**:  
  -10 for self-elimination (stepping on own trail). Penalizes risky behavior leading to self-elimination.

- **Penalty for Losing Territory**:  
  Agents lose points equivalent to the number of cells lost to an opponent. Encourages active defense and maintaining control of territory.