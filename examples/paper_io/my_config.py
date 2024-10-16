# my_config.py

import os
from make_env import make_env

# Tell the arena where it can put log files that describe the results of
# specific policies. This is also used to pass results between root processes.

LOG_COMMS_DIR = os.getcwd() + "/log_comms/"

# Tell the arena what observation and action spaces to expect

temp_env = make_env()

OBS_SPACES = temp_env.observation_spaces
ACT_SPACES = temp_env.action_spaces

del temp_env
