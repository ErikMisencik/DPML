# make_env.py
# from examples.paper_io.Paper_io import PaperIoEnv
from examples.paper_io.test import PaperIoEnv

def make_env():
    return PaperIoEnv(grid_size=50, num_players=2)
