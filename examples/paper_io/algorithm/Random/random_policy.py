import random

class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def run(self, num_steps):
        self.env.reset()
        
        # Track total rewards and episode lengths
        total_rewards = [0.0] * self.env.num_players  # Track rewards for each player
        episode_length = 0

        for step in range(num_steps):
            # Sample random actions for each player
            actions = [self.env.action_space.sample() for _ in range(self.env.num_players)]
            _, rewards, done, _ = self.env.step(actions)

            # Update total rewards and episode length
            for i, reward in enumerate(rewards):
                total_rewards[i] += reward
            episode_length += 1

            if done:
                print(f"Episode finished after {episode_length} steps")
                for i, total_reward in enumerate(total_rewards):
                    print(f"Player {i+1} total reward: {total_reward}")
                # Reset the environment for the next episode
                self.env.reset()
                total_rewards = [0.0] * self.env.num_players  # Reset rewards for each player
                episode_length = 0
