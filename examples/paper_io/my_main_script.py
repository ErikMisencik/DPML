# my_main_script.py

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg
from make_env import make_env

arena = make_stem(make_env, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

# This is a list of assignments of entity (player) to policy
match_list = [[1, 2]]

# For each policy above, specify the type of policy
# You can specify a string name or a method to create a custom algorithm
policy_types = {1: "ppo", 2: "random"}

# Train with this configuration
arena.kickoff(match_list, policy_types, num_games=15000, render=True, scale=True)

