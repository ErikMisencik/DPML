from examples.paper_io.Paper_io_develop import PaperIoEnv
import pygame
# Assuming your code is in a file named PaperIoEnv.py

def main():
    env = PaperIoEnv(num_players=2)
    obs = env.reset()

    done = False
    while not done:
        actions = []
        for i in range(env.num_players):
            if env.alive[i]:
                # Sample an action from the player's action space
                actions.append(env.action_spaces[i].sample())
            else:
                actions.append(None)  # Placeholder for eliminated players

        obs, rewards, done, info = env.step(actions)
        env.render()

        # Add a short delay to allow the rendering window to update
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    env.close()

if __name__ == "__main__":
    main()
