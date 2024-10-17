import random

class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def get_actions(self, observation):
        # Return random actions for each alive player
        return [
            self.env.action_space.sample() if self.env.alive[i] else None
            for i in range(self.env.num_players)
        ]
