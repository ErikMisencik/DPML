import pickle

class BaseAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation, player_idx):
        """
        Get the action for a single agent based on its algorithm.
        To be implemented by each specific agent.
        """
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done, player_idx):
        """
        Update the agent based on its learning algorithm.
        To be implemented by each specific agent.
        """
        raise NotImplementedError

    def save(self, filepath):
        """
        Save the agent’s model or data.
        """
        raise NotImplementedError

    def load(self, filepath):
        """
        Load the agent’s model or data.
        """
        raise NotImplementedError
