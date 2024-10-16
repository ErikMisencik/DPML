import cv2

from examples.paper_io.Paper_io_develop import PaperIoEnv
# from examples.paper_io.Paper_io import PaperIoEnv


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
              # actions.append(env.action_space.sample())
            else:
                actions.append(None)  # Placeholder for eliminated players

        obs, rewards, done, info = env.step(actions)
        env.render()

        cv2.waitKey(1)  # Add a short delay to allow the rendering window to update

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
