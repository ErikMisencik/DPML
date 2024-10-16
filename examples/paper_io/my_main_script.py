from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *

import my_config as cfg  # Configuration file for log paths and spaces
from make_env import make_env  # Environment creation method for Paper.io

# Create the Arena instance for Paper.io environment
arena = make_stem(make_env, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- Only the root process will get beyond this point ---

# Define the match assignments between entities (players) and policies
match_list = [[0, 1]]  # Entities 0 and 1

# Define what each policy does
policy_types = {0: "random", 1: "random"}

# Kick off the training session
arena.kickoff(match_list, policy_types, 15000, render=True, scale=True)

