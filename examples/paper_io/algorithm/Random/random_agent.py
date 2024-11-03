import random

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation, index=None):
        # Return random actions for each alive player
        return [
            self.env.action_spaces[i].sample() if self.env.alive[i] else None
            for i in range(self.env.num_players)
        ]