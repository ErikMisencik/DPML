# make_env.py
from examples.paper_io.Paper_io import PaperIoEnv #nema observation_spaces
# from examples.paper_io.test import PaperIoEnv # problem s teensorflow pri uceni od typka projektu

def make_env():
    return PaperIoEnv(grid_size=50, num_players=2)
